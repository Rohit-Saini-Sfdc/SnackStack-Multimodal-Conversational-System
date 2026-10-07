import tempfile
import numpy as np
from snackstack.config import openai_client
from snackstack.logger import logger


def record_and_transcribe(sample_rate: int = 16000) -> str:
    """Records audio from microphone starting on ENTER keypress and stopping on ENTER keypress.
    Transcribes recorded audio using OpenAI Whisper API.

    Args:
        sample_rate: Audio sampling rate in Hz (default 16000 Hz).

    Returns:
        Transcribed text string.
    """
    if not openai_client:
        logger.error("OpenAI client not configured for Whisper STT.")
        return ""

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        logger.error(f"Audio dependencies missing: {e}. Voice input unavailable.")
        return ""

    audio_frames = []

    def callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        audio_frames.append(indata.copy())

    logger.info("🎤 Starting audio stream... Speak now!")
    try:
        # Start non-blocking InputStream
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            input("🎙️ Recording active... [Press ENTER when done talking]\n")

        if not audio_frames:
            logger.warning("No audio frames were recorded.")
            return ""

        audio_data = np.concatenate(audio_frames, axis=0)
        logger.info("🎤 Recording finished. Transcribing with Whisper...")

        # Save to temporary WAV file for OpenAI Whisper API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            sf.write(temp_audio.name, audio_data, sample_rate)
            temp_audio.seek(0)

            with open(temp_audio.name, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
                text = transcript.text.strip()
                logger.info(f"Transcribed Text: '{text}'")
                return text

    except Exception as e:
        logger.error(f"Error during voice recording or STT: {e}")
        return ""
