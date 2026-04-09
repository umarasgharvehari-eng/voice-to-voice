import os
import uuid
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from faster_whisper import WhisperModel
from TTS.api import TTS

load_dotenv()

# =========================
# Configuration
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Whisper sizes: tiny, base, small, medium, large-v3
# small/base are lighter, medium gives better quality.
WHISPER_MODEL_SIZE = "small"

# Coqui TTS model:
# Simple and reliable English model.
TTS_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it as an environment variable.")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load STT model once
# device="cpu" works everywhere
# compute_type="int8" is lighter for CPU
stt_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

# Load TTS model once
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False)

# Temporary folder for generated audio
OUTPUT_DIR = Path(tempfile.gettempdir()) / "gradio_groq_voice_app"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helper Functions
# =========================
def transcribe_audio(audio_path: str) -> str:
    """
    Convert speech to text using faster-whisper.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Audio file not found.")

    segments, info = stt_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    return transcript


def ask_groq(user_text: str, history: list | None = None) -> str:
    """
    Send text to Groq chat completion API.
    """
    if not user_text.strip():
        raise ValueError("Transcript is empty.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful voice assistant. "
                "Reply clearly, naturally, and concisely. "
                "Keep spoken responses easy to understand."
            ),
        }
    ]

    if history:
        for human, assistant in history:
            if human:
                messages.append({"role": "user", "content": human})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": user_text})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
    )

    return chat_completion.choices[0].message.content.strip()


def synthesize_speech(text: str) -> str:
    """
    Convert assistant text reply to audio using Coqui TTS.
    Returns path of generated wav file.
    """
    if not text.strip():
        raise ValueError("Assistant reply is empty.")

    output_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    tts.tts_to_file(text=text, file_path=str(output_path))

    # Optional validation: ensure file is readable
    data, sr = sf.read(str(output_path))
    if data is None or len(data) == 0:
        raise RuntimeError("Generated audio file is empty.")

    return str(output_path)


def process_voice(audio, chat_history):
    """
    Full pipeline:
    1. STT
    2. Groq LLM
    3. TTS
    """
    if audio is None:
        return chat_history, "", "", None

    try:
        user_text = transcribe_audio(audio)
        assistant_text = ask_groq(user_text, chat_history)
        assistant_audio = synthesize_speech(assistant_text)

        chat_history = chat_history or []
        chat_history.append((user_text, assistant_text))

        return chat_history, user_text, assistant_text, assistant_audio

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return chat_history, error_message, error_message, None


def clear_all():
    return [], "", "", None


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Groq Voice-to-Voice Assistant") as demo:
    gr.Markdown("# Voice-to-Voice AI Assistant")
    gr.Markdown(
        """
        **Stack**
        - STT: Faster-Whisper
        - LLM: Groq (`llama-3.3-70b-versatile`)
        - TTS: Coqui TTS

        Record your voice, get a text answer, and hear the spoken response.
        """
    )

    chatbot = gr.Chatbot(label="Conversation", height=400)

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Speak or upload audio"
        )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    transcript_box = gr.Textbox(label="Transcription")
    response_box = gr.Textbox(label="Assistant Response", lines=6)
    audio_output = gr.Audio(label="Assistant Voice Response", type="filepath")

    submit_btn.click(
        fn=process_voice,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, transcript_box, response_box, audio_output]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, transcript_box, response_box, audio_output]
    )

if __name__ == "__main__":
    demo.launch()
