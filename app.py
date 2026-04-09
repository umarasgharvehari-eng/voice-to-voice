import os
import uuid
import json
import html
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Voice-to-Voice AI",
    page_icon="🎤",
    layout="wide",
)

APP_TITLE = "🎤 Voice-to-Voice AI Assistant"
APP_SUBTITLE = "Streamlit + Faster-Whisper + Groq + Browser Voice Output"
GROQ_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE = "small"

TEMP_DIR = Path(tempfile.gettempdir()) / "streamlit_voice_ai_app"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a helpful AI voice assistant.
Reply clearly, naturally, and concisely.
Keep answers easy to understand when spoken aloud.
Prefer short paragraphs.
""".strip()


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit secrets.")
    st.stop()


@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def load_whisper_model(model_size: str):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


client = load_groq_client()
whisper_model = load_whisper_model(WHISPER_MODEL_SIZE)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "auto_speak" not in st.session_state:
    st.session_state.auto_speak = True


def save_uploaded_audio(uploaded_audio) -> str:
    suffix = Path(uploaded_audio.name).suffix if uploaded_audio.name else ".wav"
    file_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(file_path, "wb") as f:
        f.write(uploaded_audio.read())

    return str(file_path)


def transcribe_audio(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
    )

    parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            parts.append(text)

    return " ".join(parts).strip()


def build_groq_messages(user_text: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in st.session_state.messages:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})
    return messages


def ask_groq(user_text: str) -> str:
    messages = build_groq_messages(user_text)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=700,
    )

    return chat_completion.choices[0].message.content.strip()


def add_user_message(text: str):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )


def add_assistant_message(text: str):
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )


def clear_chat():
    st.session_state.messages = []
    st.session_state.last_transcript = ""
    st.session_state.last_response = ""


def export_chat_as_text() -> str:
    lines = ["Voice AI Assistant Chat Export", "=" * 40]
    for msg in st.session_state.messages:
        role = msg.get("role", "unknown").upper()
        time = msg.get("time", "")
        content = msg.get("content", "")
        lines.append(f"[{time}] {role}:")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def export_chat_as_json() -> str:
    return json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)


def process_user_text(user_text: str):
    cleaned = (user_text or "").strip()
    if not cleaned:
        st.warning("Please enter some text.")
        return

    st.session_state.last_transcript = cleaned
    add_user_message(cleaned)

    with st.spinner("Thinking with Groq..."):
        assistant_reply = ask_groq(cleaned)

    st.session_state.last_response = assistant_reply
    add_assistant_message(assistant_reply)


def process_audio_input(uploaded_audio):
    if uploaded_audio is None:
        st.warning("Please record audio first.")
        return

    try:
        with st.spinner("Saving audio..."):
            audio_path = save_uploaded_audio(uploaded_audio)

        with st.spinner("Transcribing with Faster-Whisper..."):
            transcript = transcribe_audio(audio_path)

        if not transcript:
            st.warning("No speech detected. Please try again.")
            return

        st.session_state.last_transcript = transcript
        add_user_message(transcript)

        with st.spinner("Thinking with Groq..."):
            assistant_reply = ask_groq(transcript)

        st.session_state.last_response = assistant_reply
        add_assistant_message(assistant_reply)

    except Exception as e:
        st.error(f"Error: {e}")


def speak_text_browser(text: str, auto_play: bool = True):
    if not text:
        return

    safe_text = html.escape(text)
    autoplay_js = "speakNow();" if auto_play else ""

    components.html(
        f"""
        <div style="padding: 0.5rem 0;">
            <button onclick="speakNow()" style="
                background:#16a34a;
                color:white;
                border:none;
                padding:10px 16px;
                border-radius:8px;
                cursor:pointer;
                font-size:14px;
            ">
                🔊 Play Voice Reply
            </button>

            <button onclick="stopSpeak()" style="
                background:#dc2626;
                color:white;
                border:none;
                padding:10px 16px;
                border-radius:8px;
                cursor:pointer;
                font-size:14px;
                margin-left:8px;
            ">
                ⏹ Stop
            </button>
        </div>

        <script>
            const replyText = `{safe_text}`;

            function speakNow() {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(replyText);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                const voices = window.speechSynthesis.getVoices();

                const preferred =
                    voices.find(v => v.lang && v.lang.toLowerCase().startsWith("en")) ||
                    voices[0];

                if (preferred) {{
                    utterance.voice = preferred;
                }}

                window.speechSynthesis.speak(utterance);
            }}

            function stopSpeak() {{
                window.speechSynthesis.cancel();
            }}

            {autoplay_js}
        </script>
        """,
        height=80,
    )


with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Groq model:** `{GROQ_MODEL}`")
    st.write(f"**Whisper model:** `{WHISPER_MODEL_SIZE}`")

    st.session_state.auto_speak = st.toggle(
        "Auto play voice reply",
        value=st.session_state.auto_speak,
    )

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        clear_chat()
        st.success("Chat cleared.")

    st.divider()

    st.download_button(
        label="⬇️ Download Chat (.txt)",
        data=export_chat_as_text(),
        file_name="chat_export.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.download_button(
        label="⬇️ Download Chat (.json)",
        data=export_chat_as_json(),
        file_name="chat_export.json",
        mime="application/json",
        use_container_width=True,
    )

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])

with tab1:
    st.subheader("Speak to the assistant")
    audio_file = st.audio_input("Record your question")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Process Voice", use_container_width=True):
            process_audio_input(audio_file)

    with col2:
        if audio_file is not None:
            st.audio(audio_file)

with tab2:
    st.subheader("Type your message")
    text_input = st.text_area(
        "Ask anything",
        height=140,
        placeholder="Type your message here...",
    )

    if st.button("Send Text", use_container_width=True):
        process_user_text(text_input)

st.divider()
st.subheader("Latest Result")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Last Transcript / User Input**")
    if st.session_state.last_transcript:
        st.write(st.session_state.last_transcript)
    else:
        st.caption("No input processed yet.")

with col2:
    st.markdown("**Last Assistant Response**")
    if st.session_state.last_response:
        st.write(st.session_state.last_response)
    else:
        st.caption("No response generated yet.")

if st.session_state.last_response:
    st.subheader("🔊 Voice Output")
    speak_text_browser(
        st.session_state.last_response,
        auto_play=st.session_state.auto_speak,
    )

st.divider()
st.subheader("Conversation")

if not st.session_state.messages:
    st.info("No messages yet. Use voice input or text input to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(msg["time"])
