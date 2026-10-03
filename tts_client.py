"""Text-to-Speech using Coqui XTTS v2 (BEST quality local TTS)."""
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from TTS.api import TTS
import torch

from config import (
    DEFAULT_TTS_SAMPLE_RATE,
    TTS_MODEL,
    TTS_LANGUAGE,
    WORDS_PER_SECOND,
    USE_GPU,
)
from utils.text_cleaning import estimate_reading_time

# Global TTS model (lazy loaded)
_tts_model = None


def get_tts_model():
    """Initialize TTS model once and reuse (faster)."""
    global _tts_model
    if _tts_model is None:
        print("[TTS] Loading Coqui XTTS v2 model (first time only, ~2GB download)...")
        print("[TTS] This will take 2-3 minutes on first run, then it's cached...")

        device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        print(f"[TTS] Using device: {device}")

        _tts_model = TTS(TTS_MODEL).to(device)
        print("[TTS] Model loaded successfully!")

    return _tts_model


def generate_tts_audio(script: str, output_path: str) -> str:
    """
    Generate natural human-like audio using Coqui XTTS v2.
    Quality comparable to ElevenLabs.
    """

    try:
        print(f"[TTS] Generating audio ({len(script)} chars, ~{len(script.split())} words)")

        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Get or load model
        tts = get_tts_model()

        # Split long text into chunks (XTTS works best with <500 chars)
        max_chunk_size = 500
        chunks = []

        if len(script) > max_chunk_size:
            # Split by sentences
            sentences = script.replace("!", ".").replace("?", ".").split(".")
            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 2 < max_chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [script]

        print(f"[TTS] Processing {len(chunks)} chunks...")

        # Generate audio for each chunk
        audio_segments = []
        for i, chunk in enumerate(chunks):
            print(f"[TTS] Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")

            # Generate audio (returns numpy array)
            wav = tts.tts(
                text=chunk,
                language=TTS_LANGUAGE,
            )
            audio_segments.append(wav)

        # Combine all audio segments
        if len(audio_segments) > 1:
            print("[TTS] Combining audio chunks...")
            # Add small silence between chunks (0.3 seconds)
            silence = np.zeros(int(0.3 * 24000))
            combined = []
            for seg in audio_segments:
                combined.append(seg)
                combined.append(silence)
            final_audio = np.concatenate(combined[:-1])  # Remove last silence
        else:
            final_audio = audio_segments[0]

        # Save as WAV
        print(f"[TTS] Saving audio to {output_path}")

        # XTTS outputs at 24kHz, convert to int16
        audio_int16 = (final_audio * 32767).astype(np.int16)
        wavfile.write(output_path, 24000, audio_int16)

        duration = len(final_audio) / 24000
        print(f"[TTS] Audio generated: {duration:.1f} seconds")

        return output_path

    except Exception as e:
        print(f"[TTS ERROR] Coqui TTS failed: {e}")
        print(f"[TTS ERROR] Falling back to silent audio")

        # Fallback to silent audio
        duration_seconds = estimate_reading_time(script, WORDS_PER_SECOND)
        sample_rate = DEFAULT_TTS_SAMPLE_RATE
        num_samples = int(sample_rate * duration_seconds)
        audio_data = np.zeros(num_samples, dtype=np.int16)

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, sample_rate, audio_data)

        print(f"[TTS] Generated {duration_seconds:.1f}s of silent audio (fallback)")
        return output_path


def estimate_audio_duration(script: str) -> float:
    """Estimate audio duration."""
    return estimate_reading_time(script, WORDS_PER_SECOND)
