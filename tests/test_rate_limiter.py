"""Tests for the token-bucket rate limiter in app.llm - added after a real live run against
the Groq free tier showed INGESTION_WORKERS concurrent page-extraction calls blowing through
the account's 8000 tokens-per-minute limit in under a second, then cascading into every other
concurrently-running page failing fast for the ~30s rate-limit window. The limiter's job is to
pace requests so that never happens in the first place.
"""

import threading
import time

from app.llm import _TokenBucket, _estimate_tokens


class TestTokenBucket:
    def test_starts_full_so_a_single_call_never_waits(self):
        bucket = _TokenBucket(tokens_per_minute=6000)
        start = time.monotonic()
        bucket.acquire(500)
        assert time.monotonic() - start < 0.05

    def test_blocks_once_the_budget_is_exhausted(self):
        bucket = _TokenBucket(tokens_per_minute=6000)  # 100 tokens/sec
        bucket.acquire(6000)  # drain the bucket completely
        start = time.monotonic()
        bucket.acquire(100)  # needs ~1s to refill 100 tokens at 100/sec
        elapsed = time.monotonic() - start
        assert 0.8 <= elapsed <= 1.5

    def test_a_request_larger_than_capacity_is_capped_not_stuck_forever(self):
        bucket = _TokenBucket(tokens_per_minute=6000)
        start = time.monotonic()
        bucket.acquire(999_999_999)  # would need forever to fully refill otherwise
        assert time.monotonic() - start < 0.5  # capped to capacity, granted immediately (bucket starts full)

    def test_concurrent_callers_are_serialized_not_double_spent(self):
        # Two threads pulling the entire budget at once must not both succeed immediately -
        # the second must wait for real refill, proving the lock actually gates shared state.
        bucket = _TokenBucket(tokens_per_minute=6000)  # capacity 6000; two callers ask for 4000 each (8000 total)
        results = []

        def worker():
            start = time.monotonic()
            bucket.acquire(4000)
            results.append(time.monotonic() - start)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 2
        assert min(results) < 0.1  # one gets served immediately from the full bucket
        assert max(results) > 0.1  # the other has to wait for refill


class TestEstimateTokens:
    def test_scales_with_content_length(self):
        short = _estimate_tokens([{"role": "user", "content": "hi"}])
        long = _estimate_tokens([{"role": "user", "content": "hi " * 5000}])
        assert long > short

    def test_ignores_non_string_content_without_crashing(self):
        # Vision messages carry a list of content blocks (text + image_url), not a plain string.
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert _estimate_tokens(messages) >= 0
