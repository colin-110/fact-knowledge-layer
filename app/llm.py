"""Thin provider abstraction around Groq's OpenAI-compatible chat API.

Kept deliberately small: one function for structured-JSON text completions and
one for vision (image + text) completions. Both validate the response against
a Pydantic model and retry once with a corrective prompt on malformed output,
so callers never have to deal with raw JSON parsing.

Supports multiple API keys (GROQ_API_KEYS, comma-separated) round-robined
across calls, with automatic failover to the next key when one is rate-limited
or a model is unavailable on it - a simple way to multiply daily quota and
concurrent throughput without any other infra.

Retries are reserved for genuinely transient errors (rate limits, 5xx,
network blips). A 404/400 (bad model name, malformed request) will never
succeed on retry, so it fails immediately instead of burning ~7s of sleep
per call - that distinction alone was most of the wall-clock time on a
100-page document where every visually-complex page hit a permanently
unavailable vision model three times before giving up.
"""

import base64
import itertools
import json
import re
import threading
import time
from typing import Type, TypeVar

import requests
from groq import APIStatusError, Groq
from pydantic import BaseModel, ValidationError

from app.config import (
    GEMINI_API_KEY,
    GEMINI_TEXT_MODEL,
    GEMINI_VISION_MODEL,
    GROQ_API_KEYS,
    GROQ_TEXT_MODEL,
    GROQ_TPM_LIMIT,
    GROQ_VISION_MODEL,
)

T = TypeVar("T", bound=BaseModel)

_clients: list[Groq] = []


class _TokenBucket:
    """Preemptive rate limiter: refills continuously at `tokens_per_minute / 60` per second,
    blocks a caller until enough budget exists rather than letting requests fire freely and
    finding out only after a 429. Shared across threads (one bucket per key+model), so
    concurrent ingestion workers queue for their share of the budget instead of all racing
    past it at once."""

    def __init__(self, tokens_per_minute: int):
        self._rate_per_sec = tokens_per_minute / 60.0
        self._tokens = float(tokens_per_minute)
        self._capacity = float(tokens_per_minute)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: float) -> None:
        estimated_tokens = min(estimated_tokens, self._capacity)  # a single call can't exceed the whole budget
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._last_refill = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
                if self._tokens >= estimated_tokens:
                    self._tokens -= estimated_tokens
                    return
                wait = (estimated_tokens - self._tokens) / self._rate_per_sec
            time.sleep(min(wait, 30.0))

    def true_up(self, estimated_tokens: float, actual_tokens: float) -> None:
        """Correct the running balance using the real token count the API reported for this
        call, instead of trusting the chars/4 estimate indefinitely. That estimate runs
        systematically low on dense, numeric/currency-heavy financial text (verified live: even
        with acquire() pacing every call, real 429s from the account's true TPM ceiling still
        occurred), and small per-call underestimates compound across hundreds of calls until the
        bucket's belief drifts far enough from the account's real usage to matter. A one-line
        correction after every call keeps the two in sync instead of letting that drift grow."""
        diff = actual_tokens - estimated_tokens
        if diff == 0:
            return
        with self._lock:
            self._tokens = min(self._capacity, self._tokens - diff)


_token_buckets: dict[tuple[int, str], _TokenBucket] = {}
_token_buckets_lock = threading.Lock()


def _bucket_for(key_index: int, model: str) -> _TokenBucket:
    cache_key = (key_index, model)
    bucket = _token_buckets.get(cache_key)
    if bucket is None:
        with _token_buckets_lock:
            bucket = _token_buckets.get(cache_key)
            if bucket is None:
                bucket = _TokenBucket(GROQ_TPM_LIMIT)
                _token_buckets[cache_key] = bucket
    return bucket


def _estimate_tokens(messages: list[dict]) -> float:
    # No tokenizer dependency - only used as a starting guess before the bucket's true_up()
    # corrects it with the API's own reported usage on the very first call. chars/3.6 (rather
    # than the more common chars/4) since this content is numeric/currency-heavy financial text,
    # which tokenizes less efficiently than plain English prose - verified live: chars/4 still
    # let real 429s through even with acquire() pacing every call, because the two-thirds of a
    # minute's budget a single ~20000-char page-extraction call can consume left no room for the
    # gap between the guess and the real count.
    chars = sum(len(m["content"]) if isinstance(m["content"], str) else 0 for m in messages)
    return chars / 3.6 + 500

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# A 429 with a long retry-after (e.g. a daily token-quota exhaustion, not a brief burst
# limit) will never succeed within our short backoff window - fail over to the next key
# (or fail fast if there is none) instead of retrying into a wall that won't move.
_RATE_LIMIT_FAIL_FAST_THRESHOLD_SECONDS = 20.0

# Per-key-index state: a model that 404s on key #0 might still work fine on key #1 (e.g.
# different account tiers), so dead/rate-limited status is tracked per (key_index, model).
_DEAD_MODELS: dict[int, set[str]] = {}
_RATE_LIMITED_UNTIL: dict[tuple[int, str], float] = {}

_round_robin = itertools.count()


def _get_clients() -> list[Groq]:
    global _clients
    if not _clients:
        if not GROQ_API_KEYS:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _clients = [Groq(api_key=key) for key in GROQ_API_KEYS]
    return _clients


def _parse_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    # Fall back to parsing Groq's own message, e.g. "Please try again in 9m37.152s."
    match = re.search(r"try again in(?:\s+(\d+)m)?\s+([\d.]+)s", str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None


class ModelUnavailableError(RuntimeError):
    """Raised when a model/key combination will never succeed (dead model, or all
    configured keys are dead/rate-limited for it) - never retried, never slept on."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        return not isinstance(exc, (ValueError, TypeError))
    return status in _RETRYABLE_STATUS_CODES


def _key_is_usable(key_index: int, model: str) -> bool:
    if model in _DEAD_MODELS.get(key_index, ()):
        return False
    until = _RATE_LIMITED_UNTIL.get((key_index, model))
    return not (until and time.time() < until)


def _call_on_key(key_index: int, messages: list[dict], model: str, max_retries: int, temperature: float) -> str:
    client = _get_clients()[key_index]
    bucket = _bucket_for(key_index, model)
    estimated_tokens = _estimate_tokens(messages)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        bucket.acquire(estimated_tokens)
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                response_format={"type": "json_object"},
            )
            usage = getattr(resp, "usage", None)
            actual_tokens = getattr(usage, "total_tokens", None) if usage else None
            if actual_tokens:
                bucket.true_up(estimated_tokens, float(actual_tokens))
            return resp.choices[0].message.content or ""
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code == 404 or not _is_retryable(exc):
                _DEAD_MODELS.setdefault(key_index, set()).add(model)
                raise ModelUnavailableError(
                    f"Model '{model}' is unavailable on key #{key_index} ({exc.status_code}): {exc}"
                ) from exc
            if exc.status_code == 429:
                wait = _parse_retry_after_seconds(exc)
                if wait is not None and wait > _RATE_LIMIT_FAIL_FAST_THRESHOLD_SECONDS:
                    _RATE_LIMITED_UNTIL[(key_index, model)] = time.time() + wait
                    raise ModelUnavailableError(
                        f"Key #{key_index} hit a rate/quota limit on '{model}' requiring a {wait:.0f}s wait: {exc}"
                    ) from exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
        except Exception as exc:  # noqa: BLE001 - network/timeout/etc, worth retrying a couple times
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Groq call failed after {max_retries} retries on key #{key_index}: {last_error}") from last_error


def _call_with_retry(messages: list[dict], model: str, max_retries: int = 3, temperature: float = 0.1) -> str:
    num_keys = len(_get_clients())
    start = next(_round_robin) % num_keys
    last_error: Exception | None = None
    tried_any = False

    for offset in range(num_keys):
        key_index = (start + offset) % num_keys
        if not _key_is_usable(key_index, model):
            continue
        tried_any = True
        try:
            return _call_on_key(key_index, messages, model, max_retries, temperature)
        except ModelUnavailableError as exc:
            last_error = exc
            continue  # try the next key

    if not tried_any:
        raise ModelUnavailableError(
            f"Model '{model}' is unavailable or rate-limited on all {num_keys} configured Groq key(s)."
        )
    raise ModelUnavailableError(
        f"Model '{model}' failed on all {num_keys} configured Groq key(s). Last error: {last_error}"
    ) from last_error


def complete_json(system_prompt: str, user_prompt: str, schema: Type[T], model: str | None = None) -> T:
    """Ask the text model for JSON matching `schema`, validating and repairing once if needed.

    Falls over to Gemini (if configured) when Groq is exhausted - the same fallback chain vision
    already used, extended to text completions since a real live run hit Groq's *daily* token
    quota mid-session, not just a missing vision model. Gemini's free tier is generous enough
    that this can keep fact extraction / relationship classification going for free rather than
    stalling until Groq's quota resets."""
    try:
        return _complete_json_groq(system_prompt, user_prompt, schema, model or GROQ_TEXT_MODEL)
    except ModelUnavailableError as exc:
        if not gemini_configured():
            raise
        try:
            return complete_json_gemini(system_prompt, user_prompt, schema)
        except GeminiUnavailableError:
            raise exc from None  # Groq is the primary provider - surface its failure, not Gemini's


def _complete_json_groq(system_prompt: str, user_prompt: str, schema: Type[T], model: str) -> T:
    schema_hint = (
        f"Respond with ONLY a single JSON object matching this shape "
        f"(omit fields you cannot determine, do not invent values):\n{json.dumps(schema.model_json_schema())}"
    )
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{schema_hint}"},
        {"role": "user", "content": user_prompt},
    ]
    raw = _call_with_retry(messages, model)
    try:
        return schema.model_validate(_extract_json(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": f"That was not valid JSON matching the schema ({exc}). Return ONLY the corrected JSON object.",
            }
        )
        raw2 = _call_with_retry(messages, model)
        return schema.model_validate(_extract_json(raw2))


def complete_json_with_image(
    system_prompt: str, user_prompt: str, image_bytes: bytes, schema: Type[T], model: str | None = None
) -> T:
    """Same as complete_json but attaches a single image (chart/figure crop) to the user turn."""
    model = model or GROQ_VISION_MODEL
    b64 = base64.b64encode(image_bytes).decode("ascii")
    schema_hint = (
        f"Respond with ONLY a single JSON object matching this shape "
        f"(omit fields you cannot determine; never invent numbers you cannot actually read):\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{schema_hint}"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]
    raw = _call_with_retry(messages, model)
    try:
        return schema.model_validate(_extract_json(raw))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": f"That was not valid JSON matching the schema ({exc}). Return ONLY the corrected JSON object.",
            }
        )
        raw2 = _call_with_retry(messages, model)
        return schema.model_validate(_extract_json(raw2))


def is_model_dead(model: str) -> bool:
    """True only once EVERY configured key has confirmed this model unusable (permanently or
    rate-limited) - used by callers that want to skip an API call entirely rather than pay for
    one more attempt-and-fail when they already know no key can serve it right now."""
    try:
        num_keys = len(_get_clients())
    except RuntimeError:
        return False
    return not any(_key_is_usable(i, model) for i in range(num_keys))


# ---------------------------------------------------------------------------
# Gemini - vision fallback for when Groq has no working vision model.
#
# A separate, deliberately simpler path (single key, no round-robin) since it
# only needs to serve as a fallback, not the primary provider. Plain REST via
# `requests` rather than the Gemini SDK, to avoid a heavy new dependency for
# one endpoint.
# ---------------------------------------------------------------------------

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_GEMINI_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini can't serve this call (no key configured, or the call failed)."""


def gemini_configured() -> bool:
    return bool(GEMINI_API_KEY)


def _gemini_generate(payload: dict, model: str, schema: Type[T], max_retries: int) -> T:
    """Shared request/retry/parse logic for both the text-only and vision Gemini calls."""
    if not GEMINI_API_KEY:
        raise GeminiUnavailableError("GEMINI_API_KEY is not set.")

    url = _GEMINI_ENDPOINT.format(model=model)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=payload, timeout=60,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
            continue

        if resp.status_code == 200:
            try:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return schema.model_validate(_extract_json(text))
            except (ValueError, KeyError, IndexError, ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    continue
                raise GeminiUnavailableError(f"Gemini returned unparseable output: {exc}") from exc

        last_error = RuntimeError(f"Gemini call failed ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code not in _GEMINI_RETRYABLE_STATUS or attempt == max_retries - 1:
            raise GeminiUnavailableError(str(last_error)) from last_error
        time.sleep(min(2**attempt, 8))

    raise GeminiUnavailableError(f"Gemini call failed after {max_retries} attempts: {last_error}")


def complete_json_with_image_gemini(
    system_prompt: str, user_prompt: str, image_bytes: bytes, schema: Type[T],
    model: str | None = None, max_retries: int = 2,
) -> T:
    model = model or GEMINI_VISION_MODEL
    b64 = base64.b64encode(image_bytes).decode("ascii")
    schema_hint = (
        f"Respond with ONLY a single JSON object matching this shape "
        f"(omit fields you cannot determine; never invent numbers you cannot actually read):\n"
        f"{json.dumps(schema.model_json_schema())}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": f"{system_prompt}\n\n{schema_hint}"}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ]
            }
        ],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
    }
    return _gemini_generate(payload, model, schema, max_retries)


def complete_json_gemini(
    system_prompt: str, user_prompt: str, schema: Type[T], model: str | None = None, max_retries: int = 2,
) -> T:
    """Text-only Gemini completion - used as a fallback for fact extraction and relationship
    classification when Groq is exhausted (daily quota or per-minute rate limit), not just for
    vision. Gemini's free tier is far more generous than Groq's (verified live: Groq's
    openai/gpt-oss-120b caps at 200K tokens/day on this account, exhausted well before a
    three-document corpus finished reprocessing), so this is a real second chance at completing
    a run for free rather than stalling until the next day."""
    model = model or GEMINI_TEXT_MODEL
    schema_hint = (
        f"Respond with ONLY a single JSON object matching this shape "
        f"(omit fields you cannot determine, do not invent values):\n{json.dumps(schema.model_json_schema())}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": f"{system_prompt}\n\n{schema_hint}"}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
    }
    return _gemini_generate(payload, model, schema, max_retries)
