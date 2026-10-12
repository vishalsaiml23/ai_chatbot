from datetime import datetime, timezone
from types import SimpleNamespace

import app


def test_default_model_uses_fastest_pool_member():
    assert app.DEFAULT_MODEL == "gemini-flash-lite-latest"
    assert app.MODEL_OPTIONS[0] == app.DEFAULT_MODEL
    assert app.MODEL_POOL[0] == app.DEFAULT_MODEL


def test_normalize_rewrites_legacy_model_names():
    assert app.normalize_model_name("gemini-2.0-flash") == app.DEFAULT_MODEL
    assert app.normalize_model_name("gemini-3.8-flash") == app.DEFAULT_MODEL
    assert app.normalize_model_name("models/gemini-2.0-flash") == app.DEFAULT_MODEL


def test_get_fallback_order_is_stable():
    assert app.get_fallback_models("gemini-flash-lite-latest") == [
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


def test_trim_chat_history_limits_size():
    app.st.session_state = {"chat_history": list(range(20))}
    app.trim_chat_history()
    assert len(app.st.session_state["chat_history"]) == app.MAX_HISTORY_MESSAGES


def test_quota_error_detection():
    message = "RESOURCE_EXHAUSTED: 429 quota exceeded"
    assert app.is_quota_exhausted_error(message)


def test_detect_weather_query():
    assert app.detect_weather_query("What is the weather today in Delhi?") is True
    assert app.detect_weather_query("Explain Python loops") is False


def test_format_weather_summary():
    data = {
        "city": "Delhi",
        "temperature": 31.5,
        "feels_like": 34.0,
        "humidity": 57,
        "wind_speed": 9.2,
        "precipitation": 0.1,
        "condition": "Sunny",
    }
    text = app.format_weather_summary(data)
    assert "Delhi" in text
    assert "31.5" in text or "32" in text
    assert "Sunny" in text


def test_extract_response_handles_various_formats():
    assert app.extract_response_text({"content": "hello"}) == "hello"
    assert app.extract_response_text([{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]) == "hello world"
    assert app.extract_response_text("hello") == "hello"


class RecordingLLM:
    def invoke(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return SimpleNamespace(content="A generated answer.", response_metadata={})


def test_current_officeholder_question_uses_live_search():
    llm = RecordingLLM()

    app.invoke_chat(llm, [], "current cm of tamilnadu")

    assert llm.kwargs == {"tools": [app.GOOGLE_SEARCH_TOOL]}
    assert isinstance(llm.messages[0], app.SystemMessage)
    assert "live Google Search results" in llm.messages[0].content
    assert "Today's date is " in llm.messages[0].content
    assert app.is_time_sensitive_query("current cm of tamilnadu")


def test_historical_officeholder_question_preserves_its_date():
    llm = RecordingLLM()

    response = app.invoke_chat(llm, [], "Who was the chief minister of Tamil Nadu in 2021?")

    assert llm.kwargs == {}
    assert app.is_historical_query("Who was the chief minister of Tamil Nadu in 2021?")
    assert not app.is_time_sensitive_query("Who was the chief minister of Tamil Nadu in 2021?")
    assert "do not replace a past answer with today's answer" in llm.messages[0].content
    assert app.format_answer(response) == "A generated answer."


def test_time_sensitive_answer_requires_grounding_sources():
    response = SimpleNamespace(
        content="The current officeholder is Example Person.",
        response_metadata={
            "grounding_metadata": {
                "grounding_chunks": [
                    {"web": {"title": "Official source", "uri": "https://example.gov/officeholder"}},
                    {"web": {"title": "Official source", "uri": "https://example.gov/officeholder"}},
                ]
            }
        },
    )

    answer = app.format_answer(response, require_live_sources=True)

    assert answer.startswith("The current officeholder is Example Person.")
    assert answer.count("https://example.gov/officeholder") == 1
    assert "[Official source](https://example.gov/officeholder)" in answer


def test_time_sensitive_answer_without_sources_abstains():
    response = SimpleNamespace(
        content="A stale generated answer.",
        response_metadata={"grounding_metadata": {"grounding_chunks": []}},
    )

    assert app.format_answer(response, require_live_sources=True) == app.NO_CURRENT_SOURCE_ANSWER
    assert "A stale generated answer." not in app.format_answer(response, require_live_sources=True)


def test_changing_prices_and_recent_events_use_live_search():
    assert app.is_time_sensitive_query("What is the current price of gold?")
    assert app.is_time_sensitive_query("What are the latest election results?")
    assert app.is_time_sensitive_query("Who won the election?")
    this_year = datetime.now(timezone.utc).year
    assert app.is_time_sensitive_query(f"What is the price of gold as of {this_year}?")
    assert app.is_historical_query(f"What was the price of gold in {this_year - 1}?")
