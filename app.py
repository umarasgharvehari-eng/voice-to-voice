import os
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
from groq import Groq
from faster_whisper import WhisperModel
from TTS.api import TTS

st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎤",
    layout="wide",
)

APP_TITLE = "🎤 Voice AI Assistant"
APP_SUBTITLE = "Streamlit + Groq + Faster-Whisper + Coqui TTS"
GROQ_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE = "small"
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

TEMP_DIR = Path(tempfile.gettempdir()) / "streamlit_voice_ai_app"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a helpful AI voice assistant.
Reply clearly, naturally, and concisely.
Keep answers easy to understand when read aloud.
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


@st.cache_resource
def load_tts_model():
    return TTS(model_name=TTS_MODEL_NAME, progress_bar=False)


client = load_groq_client()
whisper_model = load_whisper_model(WHISPER_MODEL_SIZE)
tts_model = load_tts_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = ""

if "processing_error" not in st.session_state:
    st.session_state.processing_error = ""


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

    transcript_parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript_parts.append(text)

    return " ".join(transcript_parts).strip()


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


def synthesize_speech(text: str) -> str:
    output_path = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    tts_model.tts_to_file(
        text=text,
        file_path=str(output_path),
    )
    return str(output_path)


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
    st.session_state.last_audio_path = ""
    st.session_state.processing_error = ""


def export_chat_as_text() -> str:
    lines = []
    lines.append("Voice AI Assistant Chat Export")
    lines.append("=" * 40)

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

    st.session_state.processing_error = ""
    st.session_state.last_transcript = cleaned
    st.session_state.last_audio_path = ""

    add_user_message(cleaned)

    with st.spinner("Thinking with Groq..."):
        assistant_reply = ask_groq(cleaned)

    st.session_state.last_response = assistant_reply
    add_assistant_message(assistant_reply)

    with st.spinner("Generating voice output..."):
        audio_path = synthesize_speech(assistant_reply)

    st.session_state.last_audio_path = audio_path


def process_audio_input(uploaded_audio):
    if uploaded_audio is None:
        st.warning("Please record audio first.")
        return

    st.session_state.processing_error = ""

    try:
        with st.spinner("Saving audio..."):
            audio_path = save_uploaded_audio(uploaded_audio)

        with st.spinner("Transcribing with Faster-Whisper..."):
            transcript = transcribe_audio(audio_path)

        if not transcript:
            st.warning("No speech detected. Please try again.")
            return

        st.session_state.last_transcript = transcript
        st.session_state.last_audio_path = ""
        add_user_message(transcript)

        with st.spinner("Thinking with Groq..."):
            assistant_reply = ask_groq(transcript)

        st.session_state.last_response = assistant_reply
        add_assistant_message(assistant_reply)

        with st.spinner("Generating voice output..."):
            assistant_audio_path = synthesize_speech(assistant_reply)

        st.session_state.last_audio_path = assistant_audio_path

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"Error: {e}")


with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Groq model:** `{GROQ_MODEL}`")
    st.write(f"**Whisper model:** `{WHISPER_MODEL_SIZE}`")
    st.write(f"**TTS model:** `{TTS_MODEL_NAME}`")

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

if st.session_state.last_audio_path:
    st.subheader("🔊 Voice Output")
    st.audio(st.session_state.last_audio_path)

    with open(st.session_state.last_audio_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Voice Reply",
            data=f,
            file_name="assistant_reply.wav",
            mime="audio/wav",
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
