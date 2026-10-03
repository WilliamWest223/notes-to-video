"""Generate images using local Stable Diffusion."""
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from config import (
    IMAGE_MODEL,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_INFERENCE_STEPS,
    IMAGE_GUIDANCE_SCALE,
    USE_GPU,
    TEMP_DIR,
)

# Global pipeline (lazy loaded)
_sd_pipeline = None


def get_image_pipeline():
    """Initialize Stable Diffusion pipeline once and reuse."""
    global _sd_pipeline
    if _sd_pipeline is None:
        print("[IMAGE] Loading Stable Diffusion model (first time only, ~4GB download)...")
        print("[IMAGE] This will take 3-5 minutes on first run, then it's cached...")

        device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"[IMAGE] Using device: {device}")

        # Load model with optimizations
        _sd_pipeline = StableDiffusionPipeline.from_pretrained(
            IMAGE_MODEL,
            torch_dtype=torch_dtype,
            safety_checker=None,  # Disable for speed (we control prompts)
        )

        # Use faster scheduler
        _sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            _sd_pipeline.scheduler.config
        )

        _sd_pipeline = _sd_pipeline.to(device)

        # Enable memory optimizations
        if device == "cuda":
            _sd_pipeline.enable_attention_slicing()
            _sd_pipeline.enable_vae_slicing()

        print("[IMAGE] Model loaded successfully!")

    return _sd_pipeline


def generate_image(prompt: str, output_path: str, screen_id: int) -> str:
    """
    Generate an image using Stable Diffusion.
    
    Args:
        prompt: Image description from screen.image_prompt
        output_path: Where to save the image
        screen_id: Screen number for logging
        
    Returns:
        Path to generated image
    """

    try:
        print(f"[IMAGE] Generating image for screen {screen_id}")
        print(f"[IMAGE] Prompt: {prompt[:100]}...")

        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Get pipeline
        pipeline = get_image_pipeline()

        # Enhance prompt for educational content
        enhanced_prompt = f\"{prompt}, high quality, detailed, professional, educational, clean, 4k\"
        negative_prompt = \"blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature\"

        # Generate image
        print(f\"[IMAGE] Generating (this takes ~10-30 seconds)...\")

        with torch.inference_mode():
            result = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=IMAGE_INFERENCE_STEPS,
                guidance_scale=IMAGE_GUIDANCE_SCALE,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )

        image = result.images[0]

        # Save image
        image.save(output_path, format="PNG", quality=95)

        print(f"[IMAGE] Image saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"[IMAGE ERROR] Failed to generate image: {e}")
        print(f"[IMAGE ERROR] Creating placeholder instead")

        # Create a simple colored placeholder
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color="#4A90E2")
        draw = ImageDraw.Draw(img)

        # Add text
        try:
            text = f"Screen {screen_id}"
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("Arial.ttf", 100)
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            position = ((IMAGE_WIDTH - text_width) // 2, (IMAGE_HEIGHT - text_height) // 2)
            draw.text(position, text, fill="white", font=font)
        except Exception:
            pass

        img.save(output_path, format="PNG")
        return output_path


def generate_images_for_screens(screens: list, style: str) -> list:
    """
    Generate images for all screens in parallel or sequentially.
    
    Args:
        screens: List of VideoScreen objects
        style: Video style (for context)
        
    Returns:
        Updated screens list with background_image_path set
    """
    print(f"[IMAGE] Generating images for {len(screens)} screens...")

    for i, screen in enumerate(screens):
        # Generate unique filename
        image_filename = f"screen_{screen.id}_{style}.png"
        image_path = TEMP_DIR / image_filename

        # Generate image
        generated_path = generate_image(
            prompt=screen.image_prompt,
            output_path=str(image_path),
            screen_id=screen.id,
        )

        # Update screen with image path
        screen.background_image_path = generated_path

        print(f"[IMAGE] Progress: {i+1}/{len(screens)} images generated")

    print(f"[IMAGE] All images generated successfully!")
    return screens
