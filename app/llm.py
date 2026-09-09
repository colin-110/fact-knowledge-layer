"""Thin provider abstraction around Groq's OpenAI-compatible chat API.

Kept deliberately small: one function for structured-JSON text completions and
one for vision (image + text) completions. Both validate the response against
a Pydantic model and retry once with a corrective prompt on malformed output,
so callers never have to deal with raw JSON parsing.
"""

import base64
import json
import time
from typing import Type, TypeVar

from groq import Groq
from pydantic import BaseModel, ValidationError

from app.config import GROQ_API_KEY, GROQ_TEXT_MODEL, GROQ_VISION_MODEL

T = TypeVar("T", bound=BaseModel)

_client: Groq | None = None


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


def _call_with_retry(messages: list[dict], model: str, max_retries: int = 3, temperature: float = 0.1) -> str:
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
        except Exception as exc:  # noqa: BLE001 - broad on purpose, this is a boundary
            last_error = exc
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
