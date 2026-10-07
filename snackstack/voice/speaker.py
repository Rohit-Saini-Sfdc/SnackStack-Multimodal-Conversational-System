import io
from snackstack.config import openai_client
from snackstack.logger import logger


def speak_text(text: str, voice: str = "alloy") -> None:
    """Converts text to speech using OpenAI TTS API and plays it through speaker.

    Args:
        text: Text to speak.
        voice: TTS voice (e.g. 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer').
    """
    if not text or not text.strip():
        return

    if not openai_client:
        logger.error("OpenAI client not configured for TTS.")
        return

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        logger.error(f"Audio dependencies missing: {e}. Voice output unavailable.")
        return

    logger.info("🔊 Generating TTS audio response...")
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="wav",
        )

        audio_bytes = response.content
        audio_stream = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(audio_stream)

        logger.info("🔊 Playing audio response...")
        sd.play(data, sample_rate)
        sd.wait()  # Wait until audio finishes playing

    except Exception as e:
        logger.error(f"Error during TTS playback: {e}")
