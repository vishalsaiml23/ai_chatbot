import json
import os
import re
from urllib.parse import quote
from urllib.request import urlopen

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = "gemini-flash-lite-latest"
MODEL_POOL = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
MODEL_OPTIONS = MODEL_POOL
MAX_HISTORY_MESSAGES = 5
LEGACY_MODEL_NAMES = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
    "gemini-3.8-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    "models/gemini-3.8-flash",
}


def get_fallback_models(model_name):
    pool = list(MODEL_POOL)
    if model_name in pool:
        current_index = pool.index(model_name)
        return pool[current_index:] + pool[:current_index]
    return pool


def normalize_model_name(value):
    if value is None:
        return DEFAULT_MODEL

    value = str(value).strip()
    if value.startswith("models/"):
        value = value.replace("models/", "", 1)

    if not value:
        return DEFAULT_MODEL

    if value.lower() in {name.lower() for name in LEGACY_MODEL_NAMES}:
        return DEFAULT_MODEL

    return value


def is_quota_exhausted_error(message):
    if not message:
        return False
    value = str(message).lower()
    return "resource_exhausted" in value or "429" in value or "quota exceeded" in value


def detect_weather_query(prompt):
    if not prompt:
        return False
    text = str(prompt).lower()
    weather_terms = {
        "weather",
        "temperature",
        "temp",
        "rain",
        "sunny",
        "cloudy",
        "humid",
        "humidity",
        "wind",
        "forecast",
        "storm",
    }
    time_terms = {"today", "now", "current", "currently", "this morning", "this afternoon", "tonight"}
    has_weather_term = any(term in text for term in weather_terms)
    has_time_term = any(term in text for term in time_terms)
    return has_weather_term and has_time_term


def extract_city_name(prompt):
    if not prompt:
        return "Your city"

    text = str(prompt)
    match = re.search(r"(?:in|for|at)\s+([A-Za-z][A-Za-z\s'-]+?)(?:\?|$)", text, re.IGNORECASE)
    if match:
        city = match.group(1).strip()
        if city:
            return city

    if " in " in text.lower():
        tail = text.lower().split(" in ", 1)[1].strip()
        if tail:
            return tail.rstrip("?")

    return "Your city"


def weather_code_to_text(code):
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Freezing drizzle",
        57: "Heavy freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Rain showers",
        81: "Heavy rain showers",
        82: "Violent rain showers",
        85: "Snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Severe thunderstorm",
    }
    return mapping.get(code, "Current conditions")


def fetch_weather_data(city_name="Your city"):
    city = str(city_name or "Your city").strip() or "Your city"
    geocode_url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        f"name={quote(city)}&count=1&language=en&format=json"
    )
    try:
        with urlopen(geocode_url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        raise ValueError(f"Unable to find weather data for {city}.")

    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Unable to find weather data for {city}.")

    location = results[0]
    lat = location.get("latitude")
    lon = location.get("longitude")
    city_label = location.get("name") or city

    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m&timezone=auto"
    )
    try:
        with urlopen(forecast_url, timeout=10) as response:
            forecast = json.loads(response.read().decode("utf-8"))
    except Exception:
        raise ValueError(f"Unable to fetch live weather for {city_label}.")

    current = forecast.get("current") or {}
    return {
        "city": city_label,
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "precipitation": current.get("precipitation"),
        "condition": weather_code_to_text(current.get("weather_code")),
    }


def format_weather_summary(data):
    city = data.get("city", "Your city")
    temp = data.get("temperature")
    feels_like = data.get("feels_like")
    humidity = data.get("humidity")
    wind_speed = data.get("wind_speed")
    precipitation = data.get("precipitation")
    condition = data.get("condition", "Current conditions")

    parts = [f"{city}: {condition}"]
    if temp is not None:
        parts.append(f"Temperature: {temp}°C")
    if feels_like is not None:
        parts.append(f"Feels like: {feels_like}°C")
    if humidity is not None:
        parts.append(f"Humidity: {humidity}%")
    if wind_speed is not None:
        parts.append(f"Wind: {wind_speed} km/h")
    if precipitation is not None:
        parts.append(f"Precipitation: {precipitation} mm")

    return ". ".join(parts) + "."


def extract_response_text(response):
    if response is None:
        return ""

    if isinstance(response, dict):
        if "text" in response and response["text"]:
            return str(response["text"]).strip()
        if "content" in response:
            return extract_response_text(response["content"])
        if "parts" in response:
            return extract_response_text(response["parts"])

    if hasattr(response, "text"):
        text = response.text
        if isinstance(text, str):
            return text.strip()

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            item_text = extract_response_text(item)
            if item_text:
                parts.append(item_text)
        return " ".join(parts).strip()

    if isinstance(content, dict):
        if "text" in content and content["text"]:
            return str(content["text"]).strip()
        if "parts" in content:
            return extract_response_text(content["parts"])
        if "type" in content and content.get("type") == "text" and "text" in content:
            return str(content["text"]).strip()

    text = str(content).strip()
    return text if text not in {"None", "{}", "[]"} else ""

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f5f7ff 0%, #eef3ff 100%);
    }
    .welcome-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef4ff 100%);
        border: 1px solid #dfe8ff;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 25px rgba(88, 112, 255, 0.08);
    }
    .welcome-card h3 {
        margin: 0 0 0.5rem 0;
        color: #1f2a44;
        font-size: 1.2rem;
    }
    .welcome-card p {
        margin: 0 0 0.8rem 0;
        color: #46516d;
        line-height: 1.6;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .chip {
        display: inline-block;
        background: #ecf2ff;
        color: #3353d6;
        border: 1px solid #d6e2ff;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        background: #f8faff;
        border-right: 1px solid #e7ecff;
    }
    .sidebar-panel {
        background: #ffffff;
        border: 1px solid #e9efff;
        border-radius: 14px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        box-shadow: 0 4px 12px rgba(74, 102, 206, 0.04);
    }
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #ffd7d7;
        background: linear-gradient(180deg, #fff5f5 0%, #ffe9e9 100%);
        color: #a42929;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border: 1px solid #ffb7b7;
        background: linear-gradient(180deg, #fff1f1 0%, #ffdede 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything."}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model_name" not in st.session_state:
        st.session_state.model_name = normalize_model_name(os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    if "model_status" not in st.session_state:
        st.session_state.model_status = "Active"


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file and restart the app.")

    model_name = normalize_model_name(st.session_state.get("model_name", DEFAULT_MODEL))
    st.session_state.model_name = model_name

    for candidate in get_fallback_models(model_name):
        try:
            client = ChatGoogleGenerativeAI(
                model=candidate,
                temperature=0.4,
                google_api_key=api_key,
            )
            st.session_state.model_name = candidate
            st.session_state.model_status = "Active" if candidate == model_name else "Fallback"
            return client
        except Exception as exc:
            text = str(exc)
            if "NOT_FOUND" in text or "404" in text:
                continue
            if is_quota_exhausted_error(text):
                st.session_state.model_status = "Quota exhausted"
                continue
            raise

    fallback_model = DEFAULT_MODEL
    st.session_state.model_name = fallback_model
    st.session_state.model_status = "Fallback"
    return ChatGoogleGenerativeAI(
        model=fallback_model,
        temperature=0.4,
        google_api_key=api_key,
    )


def trim_chat_history():
    history = getattr(st.session_state, "chat_history", [])
    if isinstance(st.session_state, dict):
        history = st.session_state.get("chat_history", [])

    if not isinstance(history, list):
        return

    if len(history) > MAX_HISTORY_MESSAGES:
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        if isinstance(st.session_state, dict):
            st.session_state["chat_history"] = trimmed
        else:
            st.session_state.chat_history = trimmed


st.set_page_config(page_title="My Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 AI Chat Assistant")
st.caption("Your smart assistant for ideas, writing, coding, and quick answers.")

with st.sidebar:
    st.header("Settings")
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.session_state.model_name = st.selectbox(
        "Choose model",
        options=MODEL_OPTIONS,
        index=0,
        help="Fastest available default is gemini-flash-lite-latest; the app will auto-fallback when quota is exhausted.",
    )
    st.caption("Model: " + st.session_state.model_name)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
    st.caption("Quota status")
    status_value = st.session_state.get("model_status", "Active") if hasattr(st.session_state, "get") else "Active"
    model_value = st.session_state.get("model_name", normalize_model_name(os.getenv("GEMINI_MODEL", DEFAULT_MODEL))) if hasattr(st.session_state, "get") else normalize_model_name(os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    st.markdown(
        f"<div style='padding:0.5rem 0.75rem; border-radius:10px; background:#eef4ff; color:#1e3a8a; font-weight:600;'>"
        f"Model: {model_value}<br>"
        f"Status: {status_value}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-panel'>", unsafe_allow_html=True)
    st.caption("Chat controls")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything."}
        ]
        st.session_state.chat_history = []
        st.session_state.model_status = "Active"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

initialize_session_state()

if len(st.session_state.messages) == 1 and not st.session_state.chat_history:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>✨ Welcome</h3>
            <p>Ask me to write, summarize, explain, brainstorm, or help with coding tasks.</p>
            <div class="chip-row">
                <span class="chip">Writing</span>
                <span class="chip">Ideas</span>
                <span class="chip">Coding</span>
                <span class="chip">Summaries</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if detect_weather_query(prompt):
            city = extract_city_name(prompt)
            weather_data = fetch_weather_data(city)
            answer = format_weather_summary(weather_data)
        else:
            llm = get_llm()
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            trim_chat_history()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = llm.invoke(st.session_state.chat_history)

                answer = extract_response_text(response)
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_history.append(AIMessage(content=answer))
            trim_chat_history()
            raise SystemExit

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append(AIMessage(content=answer))
        trim_chat_history()
    except SystemExit:
        pass
    except Exception as e:
        message = str(e)
        if is_quota_exhausted_error(message):
            st.error(
                "The Gemini free-tier quota has been reached for this project. Please wait for the reset window or upgrade your plan."
            )
        else:
            st.error(f"Something went wrong: {e}")
        if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
            st.session_state.chat_history.pop()
