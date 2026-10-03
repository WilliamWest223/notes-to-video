"""Video assembly using MoviePy - creates video from screens, images, and audio."""
from pathlib import Path
from typing import List
import numpy as np

from moviepy.editor import (
    VideoClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    ColorClip,
    concatenate_videoclips
)
from moviepy.video.fx.all import fadein, fadeout

from config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    VIDEO_CODEC,
    AUDIO_CODEC,
    FONT_SIZE_TITLE,
    FONT_SIZE_CAPTION,
    TEXT_COLOR,
    CAPTION_BG_COLOR,
    STYLE_CONFIGS
)
from models import VideoScreen


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def create_screen_clip(screen: VideoScreen, style: str, screen_index: int) -> VideoClip:
    """
    Create a video clip for a single screen.
    
    Args:
        screen: Screen data
        style: Video style (determines colors, etc.)
        screen_index: Index for color selection
        
    Returns:
        MoviePy VideoClip
    """
    duration = screen.approx_duration_seconds
    
    # Get background color for this style
    style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["study"])
    colors = style_config["background_colors"]
    bg_color = colors[screen_index % len(colors)]
    
    # Create base background - use generated image if available
    if screen.background_image_path and Path(screen.background_image_path).exists():
        print(f"[VIDEO] Using generated image: {screen.background_image_path}")
        from moviepy.editor import ImageClip

        background = ImageClip(screen.background_image_path).set_duration(duration)

        # Resize to fit if needed
        if background.w != VIDEO_WIDTH or background.h != VIDEO_HEIGHT:
            background = background.resize((VIDEO_WIDTH, VIDEO_HEIGHT))
    else:
        print(f"[VIDEO] Using color background (no image generated)")
        # Fallback to colored background
        style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["study"])
        colors = style_config["background_colors"]
        bg_color = colors[screen_index % len(colors)]

        background = ColorClip(
            size=(VIDEO_WIDTH, VIDEO_HEIGHT),
            color=hex_to_rgb(bg_color)
        ).set_duration(duration)
    
    # Create main text overlay
    try:
        text_clip = TextClip(
            screen.on_screen_text,
            fontsize=FONT_SIZE_TITLE,
            color=TEXT_COLOR,
            font='Arial-Bold',
            size=(VIDEO_WIDTH - 200, None),
            method='caption',
            align='center'
        ).set_duration(duration).set_position('center')
        
        # Add fade effects
        text_clip = fadein(text_clip, 0.3)
        text_clip = fadeout(text_clip, 0.3)
        
    except Exception as e:
        print(f"[WARNING] Failed to create text clip: {e}")
        text_clip = None
    
    # Create caption overlays
    caption_clips = []
    for caption in screen.captions:
        try:
            # Calculate when this caption appears and disappears
            start_time = caption.time
            # Show each caption for 3 seconds or until next caption
            next_caption_time = next(
                (c.time for c in screen.captions if c.time > caption.time),
                duration
            )
            caption_duration = min(3.0, next_caption_time - start_time)
            
            # Create semi-transparent background for caption
            caption_bg = ColorClip(
                size=(VIDEO_WIDTH, 100),
                color=CAPTION_BG_COLOR[:3]
            ).set_opacity(0.7).set_duration(caption_duration).set_start(start_time)
            caption_bg = caption_bg.set_position(('center', VIDEO_HEIGHT - 150))
            
            # Create caption text
            caption_text = TextClip(
                caption.text,
                fontsize=FONT_SIZE_CAPTION,
                color=TEXT_COLOR,
                font='Arial',
                size=(VIDEO_WIDTH - 100, None),
                method='caption',
                align='center'
            ).set_duration(caption_duration).set_start(start_time)
            caption_text = caption_text.set_position(('center', VIDEO_HEIGHT - 150))
            
            caption_clips.extend([caption_bg, caption_text])
            
        except Exception as e:
            print(f"[WARNING] Failed to create caption clip: {e}")
            continue
    
    # Composite all elements
    elements = [background]
    if text_clip:
        elements.append(text_clip)
    elements.extend(caption_clips)
    
    return CompositeVideoClip(elements, size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def assemble_video(
    screens: List[VideoScreen],
    audio_path: str,
    output_path: str,
    style: str
) -> str:
    """
    Assemble final video from screens and audio.
    
    Args:
        screens: List of video screens
        audio_path: Path to audio file
        output_path: Where to save video
        style: Video style
        
    Returns:
        Path to generated video file
    """
    print(f"[VIDEO] Assembling video with {len(screens)} screens")
    print(f"[VIDEO] Audio: {audio_path}")
    print(f"[VIDEO] Output: {output_path}")
    
    try:
        # Create clip for each screen
        screen_clips = []
        for idx, screen in enumerate(screens):
            print(f"[VIDEO] Creating clip for screen {screen.id}")
            clip = create_screen_clip(screen, style, idx)
            screen_clips.append(clip)
        
        # Concatenate all screens
        print("[VIDEO] Concatenating screen clips")
        video = concatenate_videoclips(screen_clips, method="compose")
        
        # Load and attach audio
        print("[VIDEO] Attaching audio track")
        audio = AudioFileClip(audio_path)
        
        # Adjust video duration to match audio if needed
        if abs(video.duration - audio.duration) > 1.0:
            print(f"[VIDEO] Duration mismatch: video={video.duration:.1f}s, audio={audio.duration:.1f}s")
            # Use audio duration as the target
            if video.duration < audio.duration:
                # Extend last screen
                print("[VIDEO] Extending video to match audio")
            else:
                # Speed up video slightly
                print("[VIDEO] Video will be trimmed to audio length")
            video = video.set_duration(audio.duration)
        
        video = video.set_audio(audio)
        
        # Write final video
        print("[VIDEO] Writing final video file...")
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        video.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            temp_audiofile=str(output_path_obj.parent / 'temp_audio.m4a'),
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Clean up
        video.close()
        audio.close()
        for clip in screen_clips:
            clip.close()
        
        print(f"[VIDEO] Video saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Video assembly failed: {e}")
        raise RuntimeError(f"Failed to assemble video: {str(e)}")
