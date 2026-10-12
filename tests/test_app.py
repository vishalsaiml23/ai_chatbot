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
