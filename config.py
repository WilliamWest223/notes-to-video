"""Configuration settings for the Notes-to-Video backend."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEO_DIR = OUTPUT_DIR / "videos"
TEMP_DIR = OUTPUT_DIR / "temp"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, AUDIO_DIR, VIDEO_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Video generation defaults
DEFAULT_STYLE = "study"
DEFAULT_DURATION_SECONDS = 90
MIN_DURATION_SECONDS = 30
MAX_DURATION_SECONDS = 300

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
AUDIO_CODEC = "aac"

# Text rendering settings
FONT_SIZE_TITLE = 72
FONT_SIZE_BODY = 48
FONT_SIZE_CAPTION = 36
TEXT_COLOR = "white"
CAPTION_BG_COLOR = (0, 0, 0, 180)

# TTS settings
DEFAULT_TTS_SAMPLE_RATE = 24000
WORDS_PER_SECOND = 2.5

# LLM settings - TODO: Add your API keys
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4000

# TTS settings - TODO: Add your TTS API keys
TTS_API_KEY = os.getenv("TTS_API_KEY", "")
TTS_PROVIDER = "openai"
TTS_VOICE = "alloy"

# Style configurations
STYLE_CONFIGS = {
    "brainrot": {
        "description": "High-energy, meme-heavy, Gen-Z style",
        "background_colors": ["#FF6B9D", "#C44569", "#FFA07A", "#FFD93D"],
        "words_per_second": 3.0,
        "caption_style": "frequent",
        "image_style": "energetic, colorful, meme-inspired, vibrant gradients"
    },
    "study": {
        "description": "Clean, focused, educational style",
        "background_colors": ["#4A90E2", "#50C878", "#9B59B6", "#3498DB"],
        "words_per_second": 2.5,
        "caption_style": "moderate",
        "image_style": "clean, professional, educational diagrams"
    },
    "professor": {
        "description": "Formal, academic style",
        "background_colors": ["#2C3E50", "#34495E", "#7F8C8D", "#95A5A6"],
        "words_per_second": 2.0,
        "caption_style": "minimal",
        "image_style": "academic, scholarly, textbook-style"
    }
}

VIDEO_FILENAME_TEMPLATE = "{timestamp}_notes_video.mp4"
AUDIO_FILENAME_TEMPLATE = "{timestamp}_audio.wav"

# Local AI Model Settings
OLLAMA_MODEL = "llama3.1:8b"  # Best reasoning model
OLLAMA_BASE_URL = "http://localhost:11434"

# TTS Model Settings
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"  # Best quality
TTS_LANGUAGE = "en"
TTS_SPEAKER_WAV = None  # Use default voice (or path to your voice sample)

# Image Generation Settings
IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"  # Best quality SD model
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_INFERENCE_STEPS = 30  # Higher = better quality, slower (20-50)
IMAGE_GUIDANCE_SCALE = 7.5  # How closely to follow prompt (5-15)

# Performance Settings
USE_GPU = True  # Set False if no GPU (will be slower but still works)
