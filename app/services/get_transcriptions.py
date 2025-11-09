from app.core.config import settings
import openai
import google.generativeai as genai
import whisper

openai.api_key = settings.openai_api_key
genai.configure(api_key=settings.gemini_api_key)
WHISPER_MODEL = whisper.load_model("small", device="cpu")

def transcribe_with_whisper(audio_path: str, provider: str = "local") -> str:
    if provider.lower() == "openai":
        return __transcribe_openai(audio_path)
    return __transcribe_local_whisper(audio_path)

def __transcribe_openai(audio_path: str) -> str:
    """
    Transcribes audio from a given file path using OpenAI's Whisper API.

    Args:
        audio_path (str): Path to the audio file to transcribe.

    Returns:
        str: The transcribed text from the audio.

    Raises:
        openai.error.OpenAIError: If the transcription request fails.
    """
    resp = openai.audio.transcriptions.create(
        file=open(audio_path, "rb"),
        model="whisper-1",
        response_format="text",
    )
    return resp

def __transcribe_local_whisper(audio_path: str) -> str:
    """Transcribe audio using local Whisper model (free, CPU-compatible)."""
    result = WHISPER_MODEL.transcribe(audio_path)
    return result["text"]