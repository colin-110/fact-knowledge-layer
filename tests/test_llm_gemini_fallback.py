"""complete_json's Groq -> Gemini fallback for text completions - added after a live run hit
Groq's real daily token quota mid-session, stalling fact extraction and relationship
classification entirely even though a configured Gemini key (much more generous free tier)
could have kept going. Mirrors the fallback chain vision already had, extended to text.
"""

from unittest.mock import patch

from pydantic import BaseModel

from app import llm


class _Schema(BaseModel):
    value: str


class TestCompleteJsonFallback:
    def test_uses_groq_result_when_groq_succeeds(self):
        with patch("app.llm._complete_json_groq", return_value=_Schema(value="from-groq")) as mock_groq, \
             patch("app.llm.complete_json_gemini") as mock_gemini:
            result = llm.complete_json("sys", "user", _Schema)
        assert result.value == "from-groq"
        mock_groq.assert_called_once()
        mock_gemini.assert_not_called()

    def test_falls_over_to_gemini_when_groq_is_unavailable_and_gemini_is_configured(self):
        with patch("app.llm._complete_json_groq", side_effect=llm.ModelUnavailableError("groq dead")), \
             patch("app.llm.gemini_configured", return_value=True), \
             patch("app.llm.complete_json_gemini", return_value=_Schema(value="from-gemini")) as mock_gemini:
            result = llm.complete_json("sys", "user", _Schema)
        assert result.value == "from-gemini"
        mock_gemini.assert_called_once()

    def test_raises_the_original_groq_error_when_gemini_is_not_configured(self):
        with patch("app.llm._complete_json_groq", side_effect=llm.ModelUnavailableError("groq dead")), \
             patch("app.llm.gemini_configured", return_value=False):
            try:
                llm.complete_json("sys", "user", _Schema)
                assert False, "expected ModelUnavailableError"
            except llm.ModelUnavailableError as exc:
                assert "groq dead" in str(exc)

    def test_raises_the_original_groq_error_when_gemini_also_fails(self):
        # Groq is the primary provider - if both fail, the Groq failure is the more useful one
        # to surface (it's what a user configured first and expects to work).
        with patch("app.llm._complete_json_groq", side_effect=llm.ModelUnavailableError("groq dead")), \
             patch("app.llm.gemini_configured", return_value=True), \
             patch("app.llm.complete_json_gemini", side_effect=llm.GeminiUnavailableError("gemini also dead")):
            try:
                llm.complete_json("sys", "user", _Schema)
                assert False, "expected ModelUnavailableError"
            except llm.ModelUnavailableError as exc:
                assert "groq dead" in str(exc)
