import os
import uuid
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq
from faster_whisper import WhisperModel
from TTS.api import TTS

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Voice AI App", page_icon="🎤", layout="centered")

# =========================
# Load Groq API Key
# Priority:
# 1. Streamlit secrets
# 2. Environment variable
# =========================
GROQ_API_KEY = None

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit secrets.")
    st.stop()

# =========================
# Config
# =========================
GROQ_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE = "small"  # tiny, base, small, medium, large-v3
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

# Temp directory
TEMP_DIR = Path(tempfile.gettempdir()) / "streamlit_voice_ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Cached Models
# =========================
@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )

@st.cache_resource
def load_tts_model():
    return TTS(model_name=TTS_MODEL_NAME, progress_bar=False)

client = load_groq_client()
whisper_model = load_whisper_model()
tts_model = load_tts_model()

# =========================
# Session State
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Helper Functions
# =========================
def save_uploaded_audio(uploaded_file) -> str:
    """
    Save Streamlit uploaded audio to a temp file.
    """
    suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".wav"
    temp_audio_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    return str(temp_audio_path)

def transcribe_audio(audio_path: str) -> str:
    """
    Speech-to-text using faster-whisper.
    """
    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return transcript

def build_messages(user_text: str):
    """
    Build Groq messages with history.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Reply in a natural, concise, and spoken style. "
                "Keep answers clear and easy to listen to."
            ),
        }
    ]

    for item in st.session_state.chat_history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["assistant"]})

    messages.append({"role": "user", "content": user_text})
    return messages

def ask_groq(user_text: str) -> str:
    """
    Get response from Groq.
    """
    messages = build_messages(user_text)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
    )

    return chat_completion.choices[0].message.content.strip()

def synthesize_speech(text: str) -> str:
    """
    Text-to-speech using Coqui TTS.
    """
    output_path = TEMP_DIR / f"{uuid.uuid4().hex}.wav"
    tts_model.tts_to_file(text=text, file_path=str(output_path))
    return str(output_path)

# =========================
# UI
# =========================
st.title("🎤 Voice-to-Voice AI App")
st.write("Free open-source STT + TTS with Groq LLM and Streamlit frontend.")

with st.expander("Current Stack", expanded=True):
    st.markdown(
        f"""
        - **STT:** Faster-Whisper (`{WHISPER_MODEL_SIZE}`)
        - **LLM:** Groq (`{GROQ_MODEL}`)
        - **TTS:** Coqui TTS (`{TTS_MODEL_NAME}`)
        """
    )

audio_file = st.audio_input("Record your voice")

col1, col2 = st.columns(2)

with col1:
    process_btn = st.button("Process Voice", use_container_width=True)

with col2:
    clear_btn = st.button("Clear Chat", use_container_width=True)

if clear_btn:
    st.session_state.chat_history = []
    st.success("Chat cleared.")

if process_btn:
    if audio_file is None:
        st.warning("Pehle audio record karo.")
    else:
        try:
            with st.spinner("Saving audio..."):
                audio_path = save_uploaded_audio(audio_file)

            st.audio(audio_path)

            with st.spinner("Transcribing with Whisper..."):
                user_text = transcribe_audio(audio_path)

            st.subheader("Transcription")
            st.write(user_text if user_text else "_No speech detected._")

            if not user_text:
                st.warning("Speech detect nahi hui. Dobara try karo.")
            else:
                with st.spinner("Getting reply from Groq..."):
                    assistant_text = ask_groq(user_text)

                st.subheader("Assistant Reply")
                st.write(assistant_text)

                with st.spinner("Generating voice with Coqui TTS..."):
                    assistant_audio_path = synthesize_speech(assistant_text)

                st.subheader("Assistant Voice")
                st.audio(assistant_audio_path)

                st.session_state.chat_history.append(
                    {
                        "user": user_text,
                        "assistant": assistant_text,
                    }
                )

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# Chat History
# =========================
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.container():
            st.markdown(f"**User:** {item['user']}")
            st.markdown(f"**Assistant:** {item['assistant']}")
            st.divider()
