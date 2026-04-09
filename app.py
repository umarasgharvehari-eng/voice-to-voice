import os
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
from groq import Groq
from faster_whisper import WhisperModel

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎤",
    layout="wide",
)

# =====================================
# App Constants
# =====================================
APP_TITLE = "🎤 Voice AI Assistant"
APP_SUBTITLE = "Streamlit + Groq + Faster-Whisper (Free Open-Source STT)"
GROQ_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL_SIZE = "small"  # tiny, base, small, medium, large-v3

TEMP_DIR = Path(tempfile.gettempdir()) / "streamlit_voice_ai_app"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a helpful AI voice assistant.
Reply clearly, naturally, and concisely.
Keep answers easy to understand when read aloud.
When appropriate, use short paragraphs or bullets.
""".strip()

# =====================================
# Secrets / API Key
# =====================================
def get_groq_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")

GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it in Streamlit secrets.")
    st.info(
        "Create `.streamlit/secrets.toml` locally or add secrets in Streamlit Cloud."
    )
    st.stop()

# =====================================
# Cached Resources
# =====================================
@st.cache_resource
def load_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_whisper_model(model_size: str) -> WhisperModel:
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

client = load_groq_client()
whisper_model = load_whisper_model(WHISPER_MODEL_SIZE)

# =====================================
# Session State
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "processing_error" not in st.session_state:
    st.session_state.processing_error = ""

# =====================================
# Helpers
# =====================================
def save_uploaded_audio(uploaded_audio) -> str:
    """
    Save Streamlit audio input to a temporary file.
    """
    suffix = Path(uploaded_audio.name).suffix if uploaded_audio.name else ".wav"
    file_path = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    with open(file_path, "wb") as f:
        f.write(uploaded_audio.read())

    return str(file_path)

def transcribe_audio(audio_path: str) -> str:
    """
    Convert speech to text using Faster-Whisper.
    """
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

def build_groq_messages(user_text: str) -> list[dict]:
    """
    Build message list for Groq chat completion using chat history.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in st.session_state.messages:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text})
    return messages

def ask_groq(user_text: str) -> str:
    """
    Call Groq chat completion.
    """
    messages = build_groq_messages(user_text)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=700,
    )

    return chat_completion.choices[0].message.content.strip()

def add_user_message(text: str) -> None:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )

def add_assistant_message(text: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )

def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.last_transcript = ""
    st.session_state.last_response = ""
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

def process_user_text(user_text: str) -> None:
    """
    Process text input through Groq and update chat.
    """
    cleaned = (user_text or "").strip()
    if not cleaned:
        st.warning("Please enter some text.")
        return

    st.session_state.processing_error = ""
    st.session_state.last_transcript = cleaned

    add_user_message(cleaned)

    with st.spinner("Thinking with Groq..."):
        assistant_reply = ask_groq(cleaned)

    st.session_state.last_response = assistant_reply
    add_assistant_message(assistant_reply)

def process_audio_input(uploaded_audio) -> None:
    """
    Process voice input:
    1. Save audio
    2. Transcribe with Whisper
    3. Send transcript to Groq
    """
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
        add_user_message(transcript)

        with st.spinner("Thinking with Groq..."):
            assistant_reply = ask_groq(transcript)

        st.session_state.last_response = assistant_reply
        add_assistant_message(assistant_reply)

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"Error: {e}")

# =====================================
# Sidebar
# =====================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Groq model:** `{GROQ_MODEL}`")
    st.write(f"**Whisper model:** `{WHISPER_MODEL_SIZE}`")

    st.divider()

    st.subheader("📦 Stack")
    st.markdown(
        """
- **Frontend:** Streamlit  
- **LLM:** Groq  
- **STT:** Faster-Whisper  
- **TTS:** Not enabled in this stable cloud version
        """
    )

    st.divider()

    if st.button("🧹 Clear Chat", use_container_width=True):
        clear_chat()
        st.success("Chat cleared.")

    st.divider()

    txt_export = export_chat_as_text()
    json_export = export_chat_as_json()

    st.download_button(
        label="⬇️ Download Chat (.txt)",
        data=txt_export,
        file_name="chat_export.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.download_button(
        label="⬇️ Download Chat (.json)",
        data=json_export,
        file_name="chat_export.json",
        mime="application/json",
        use_container_width=True,
    )

# =====================================
# Main UI
# =====================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])

with tab1:
    st.subheader("Speak to the assistant")
    audio_file = st.audio_input("Record your question")

    col1, col2 = st.columns([1, 1])

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

# =====================================
# Last Interaction Summary
# =====================================
st.divider()
st.subheader("Latest Result")

result_col1, result_col2 = st.columns(2)

with result_col1:
    st.markdown("**Last Transcript / User Input**")
    if st.session_state.last_transcript:
        st.write(st.session_state.last_transcript)
    else:
        st.caption("No input processed yet.")

with result_col2:
    st.markdown("**Last Assistant Response**")
    if st.session_state.last_response:
        st.write(st.session_state.last_response)
    else:
        st.caption("No response generated yet.")

# =====================================
# Chat UI
# =====================================
st.divider()
st.subheader("Conversation")

if not st.session_state.messages:
    st.info("No messages yet. Use voice input or text input to start chatting.")
else:
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        time = msg.get("time", "")

        with st.chat_message(role):
            st.markdown(content)
            if time:
                st.caption(time)

# =====================================
# Footer
# =====================================
st.divider()
st.caption(
    "Built with Streamlit, Groq, and Faster-Whisper. "
    "This cloud-safe version skips TTS for deployment stability."
)
