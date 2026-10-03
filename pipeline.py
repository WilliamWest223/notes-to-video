"""Pipeline orchestrator - runs the full notes-to-video workflow."""
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from config import (
    AUDIO_DIR,
    VIDEO_DIR,
    AUDIO_FILENAME_TEMPLATE,
    VIDEO_FILENAME_TEMPLATE,
    STYLE_CONFIGS
)
from models import VideoScreen, StructuredNotes
from utils.text_cleaning import clean_notes, extract_topic_from_notes
from llm_client import organize_notes_with_llm, generate_screens_with_llm
from tts_client import generate_tts_audio, estimate_audio_duration
from video_assembler import assemble_video
from image_generator import generate_images_for_screens


def run_organize_notes_pipeline(
    notes: str,
    topic: Optional[str],
    style: str
) -> StructuredNotes:
    """
    Run Stage A: Organize raw notes into structured format.
    
    Args:
        notes: Raw notes text
        topic: Optional topic name
        style: Video style
        
    Returns:
        StructuredNotes object
    """
    print(f"[PIPELINE] Stage A: Organizing notes (style: {style})")
    
    # Clean the input notes
    cleaned_notes = clean_notes(notes)
    print(f"[PIPELINE] Cleaned notes: {len(cleaned_notes)} characters")
    
    # Extract or use provided topic
    final_topic = extract_topic_from_notes(cleaned_notes, topic)
    print(f"[PIPELINE] Topic: {final_topic}")
    
    # Use LLM to organize notes
    structured_dict = organize_notes_with_llm(cleaned_notes, final_topic, style)
    
    # Convert to Pydantic model
    structured_notes = StructuredNotes(**structured_dict)
    
    print(f"[PIPELINE] Organized into {len(structured_notes.sections)} sections")
    
    return structured_notes


def run_generate_screens_pipeline(
    structured_notes: StructuredNotes,
    style: str,
    target_duration_seconds: int
) -> List[VideoScreen]:
    """
    Run Stage B: Generate video screens from structured notes.
    
    Args:
        structured_notes: Organized notes
        style: Video style
        target_duration_seconds: Target video duration
        
    Returns:
        List of VideoScreen objects
    """
    print(f"[PIPELINE] Stage B: Generating video screens (target: {target_duration_seconds}s)")
    
    # Use LLM to generate screens
    screens_dict_list = generate_screens_with_llm(
        structured_notes.model_dump(),
        style,
        target_duration_seconds
    )
    
    # Convert to Pydantic models
    screens = [VideoScreen(**screen_dict) for screen_dict in screens_dict_list]
    
    total_duration = sum(screen.approx_duration_seconds for screen in screens)
    print(f"[PIPELINE] Generated {len(screens)} screens, total duration: {total_duration:.1f}s")
    
    return screens


def run_generate_audio_pipeline(
    screens: List[VideoScreen],
    timestamp: str
) -> Tuple[str, str]:
    """
    Run Stage C: Generate TTS audio from screens.
    
    Args:
        screens: List of video screens
        timestamp: Timestamp for filename
        
    Returns:
        Tuple of (audio_path, full_script)
    """
    print(f"[PIPELINE] Stage C: Generating TTS audio")
    
    # Combine all voiceover text into one script
    full_script = " ".join(screen.voiceover_text for screen in screens)
    
    print(f"[PIPELINE] Full script: {len(full_script)} characters")
    
    # Generate audio file
    audio_filename = AUDIO_FILENAME_TEMPLATE.format(timestamp=timestamp)
    audio_path = str(AUDIO_DIR / audio_filename)
    
    audio_path = generate_tts_audio(full_script, audio_path)
    
    print(f"[PIPELINE] Audio generated: {audio_path}")
    
    return audio_path, full_script


def run_assemble_video_pipeline(
    screens: List[VideoScreen],
    audio_path: str,
    style: str,
    timestamp: str
) -> str:
    """
    Run Stage D: Assemble final video.
    
    Args:
        screens: List of video screens
        audio_path: Path to audio file
        style: Video style
        timestamp: Timestamp for filename
        
    Returns:
        Path to generated video
    """
    print(f"[PIPELINE] Stage D: Assembling video")
    
    # Generate video file
    video_filename = VIDEO_FILENAME_TEMPLATE.format(timestamp=timestamp)
    video_path = str(VIDEO_DIR / video_filename)
    
    video_path = assemble_video(screens, audio_path, video_path, style)
    
    print(f"[PIPELINE] Video assembled: {video_path}")
    
    return video_path


def run_full_pipeline(
    notes: str,
    topic: Optional[str],
    style: str,
    target_duration_seconds: int
) -> Tuple[str, StructuredNotes, List[VideoScreen], Dict[str, Any]]:
    """
    Run the complete notes-to-video pipeline.
    
    Args:
        notes: Raw notes text
        topic: Optional topic name
        style: Video style (brainrot, study, professor)
        target_duration_seconds: Target video duration
        
    Returns:
        Tuple of (video_path, structured_notes, screens, metadata)
    """
    print(f"[PIPELINE] ========================================")
    print(f"[PIPELINE] Starting full pipeline")
    print(f"[PIPELINE] Style: {style}, Target duration: {target_duration_seconds}s")
    print(f"[PIPELINE] ========================================")
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    try:
        # Stage A: Organize notes
        structured_notes = run_organize_notes_pipeline(notes, topic, style)
        
        # Stage B: Generate screens
        screens = run_generate_screens_pipeline(
            structured_notes,
            style,
            target_duration_seconds
        )

        # Stage B.5: Generate images for screens
        print(f"[PIPELINE] Stage B.5: Generating AI images for screens")
        screens = generate_images_for_screens(screens, style)
        
        # Stage C: Generate audio
        audio_path, full_script = run_generate_audio_pipeline(screens, timestamp)
        
        # Stage D: Assemble video
        video_path = run_assemble_video_pipeline(screens, audio_path, style, timestamp)
        
        # Calculate metadata
        estimated_duration = sum(screen.approx_duration_seconds for screen in screens)
        metadata = {
            "video_style": style,
            "target_duration_seconds": target_duration_seconds,
            "estimated_total_duration_seconds": estimated_duration,
            "screen_count": len(screens),
            "generated_at": timestamp
        }
        
        print(f"[PIPELINE] ========================================")
        print(f"[PIPELINE] Pipeline completed successfully!")
        print(f"[PIPELINE] Video: {video_path}")
        print(f"[PIPELINE] Screens: {len(screens)}, Duration: {estimated_duration:.1f}s")
        print(f"[PIPELINE] ========================================")
        
        return video_path, structured_notes, screens, metadata
        
    except Exception as e:
        print(f"[PIPELINE] ========================================")
        print(f"[PIPELINE] Pipeline failed: {e}")
        print(f"[PIPELINE] ========================================")
        raise
