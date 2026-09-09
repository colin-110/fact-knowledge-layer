"""Thin provider abstraction around Groq's OpenAI-compatible chat API.

Kept deliberately small: one function for structured-JSON text completions and
one for vision (image + text) completions. Both validate the response against
a Pydantic model and retry once with a corrective prompt on malformed output,
so callers never have to deal with raw JSON parsing.

Retries are reserved for genuinely transient errors (rate limits, 5xx,
network blips). A 404/400 (bad model name, malformed request) will never
succeed on retry, so it fails immediately instead of burning ~7s of sleep
per call - that distinction alone was most of the wall-clock time on a
100-page document where every visually-complex page hit a permanently
unavailable vision model three times before giving up.
"""

import base64
import json
import re
import time
from typing import Type, TypeVar

from groq import APIStatusError, Groq
from pydantic import BaseModel, ValidationError

from app.config import GROQ_API_KEY, GROQ_TEXT_MODEL, GROQ_VISION_MODEL

T = TypeVar("T", bound=BaseModel)

_client: Groq | None = None

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# A 429 with a long retry-after (e.g. a daily token-quota exhaustion, not a brief burst
# limit) will never succeed within our short backoff window - fail every concurrent call
# fast instead of each independently sleeping and retrying into a wall that won't move.
_RATE_LIMIT_FAIL_FAST_THRESHOLD_SECONDS = 20.0

# Once a model name proves permanently unusable (e.g. 404 model_not_found) for this
# process, stop trying it again - this is the circuit breaker that saves ~7s per page
# on a document where the vision model is unavailable for the whole run.
_DEAD_MODELS: set[str] = set()

# model -> unix timestamp before which calls to it should fail immediately rather than retry.
_RATE_LIMITED_UNTIL: dict[str, float] = {}


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
    """Raised immediately (no retry, no sleep) for a model that will never succeed."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


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
        # Not an API status error at all (e.g. a connection error) - worth one retry.
        return not isinstance(exc, (ValueError, TypeError))
    return status in _RETRYABLE_STATUS_CODES


def _call_with_retry(messages: list[dict], model: str, max_retries: int = 3, temperature: float = 0.1) -> str:
    if model in _DEAD_MODELS:
        raise ModelUnavailableError(f"Model '{model}' was already confirmed unavailable this run - not retrying.")

    rate_limited_until = _RATE_LIMITED_UNTIL.get(model)
    if rate_limited_until and time.time() < rate_limited_until:
        remaining = rate_limited_until - time.time()
        raise ModelUnavailableError(
            f"Model '{model}' hit a rate/quota limit; not retrying for another {remaining:.0f}s."
        )

    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code == 404 or not _is_retryable(exc):
                _DEAD_MODELS.add(model)
                raise ModelUnavailableError(f"Model '{model}' is unavailable ({exc.status_code}): {exc}") from exc
            if exc.status_code == 429:
                wait = _parse_retry_after_seconds(exc)
                if wait is not None and wait > _RATE_LIMIT_FAIL_FAST_THRESHOLD_SECONDS:
                    # A long wait (e.g. a daily token quota, not a brief burst limit) will never
                    # clear within our backoff window - stop every concurrent caller immediately
                    # instead of each burning its own retries against a wall that won't move.
                    _RATE_LIMITED_UNTIL[model] = time.time() + wait
                    raise ModelUnavailableError(
                        f"Model '{model}' hit a rate/quota limit requiring a {wait:.0f}s wait - failing fast "
                        f"instead of retrying: {exc}"
                    ) from exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
        except Exception as exc:  # noqa: BLE001 - network/timeout/etc, worth retrying a couple times
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Groq call failed after {max_retries} retries: {last_error}") from last_error


def complete_json(system_prompt: str, user_prompt: str, schema: Type[T], model: str | None = None) -> T:
    """Ask the text model for JSON matching `schema`, validating and repairing once if needed."""
    model = model or GROQ_TEXT_MODEL
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
    return model in _DEAD_MODELS
