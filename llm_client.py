"""LLM client abstraction layer with prompts for note organization and screen generation."""
import json
from typing import Dict, Any, Optional
import ollama
from config import (
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    STYLE_CONFIGS,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


# TODO: Uncomment when you have a real LLM API key
# Example for OpenAI:
# import openai
# openai.api_key = LLM_API_KEY


def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = LLM_TEMPERATURE
) -> str:
    """
    Call LOCAL Ollama with Llama 3.1 8B (BEST free model).
    Quality comparable to GPT-4 for most tasks.
    """
    try:
        print(f"[LLM] Calling {OLLAMA_MODEL} locally...")

        # Enhance prompt for JSON mode
        if json_mode:
            system_prompt += "\n\nCRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations, just pure JSON."

        # Call local Ollama with optimized settings
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": 4000,  # Allow longer responses
                "top_p": 0.9,
                "top_k": 40,
            },
        )

        content = response["message"]["content"].strip()

        # Clean up JSON responses
        if json_mode:
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Validate JSON
            try:
                json.loads(content)  # Test if valid
            except json.JSONDecodeError:
                print("[LLM WARNING] Invalid JSON, attempting to fix...")
                # Try to extract JSON object
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

        print(f"[LLM] Generated {len(content)} characters")
        return content

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        print("[LLM ERROR] Make sure Ollama is running: ollama serve")
        print(f"[LLM ERROR] And model is installed: ollama pull {OLLAMA_MODEL}")

        # Fallback to stubs
        if "organize" in system_prompt.lower():
            return _stub_organize_notes_response(user_prompt)
        elif "screens" in system_prompt.lower():
            return _stub_generate_screens_response(user_prompt)
        else:
            return '{"error": "LLM unavailable"}'


def _stub_organize_notes_response(notes: str) -> str:
    """Stub response for organizing notes."""
    return json.dumps({
        "topic": "Study Notes",
        "sections": [
            {
                "title": "Introduction",
                "summary": "Overview of the main concepts covered in these notes.",
                "bullets": [
                    {"text": "Key concept #1 from your notes", "importance": "high"},
                    {"text": "Key concept #2 from your notes", "importance": "medium"}
                ]
            },
            {
                "title": "Main Content",
                "summary": "Detailed explanation of the core topics.",
                "bullets": [
                    {"text": "Important detail from notes", "importance": "high"},
                    {"text": "Supporting information", "importance": "medium"},
                    {"text": "Additional context", "importance": "low"}
                ]
            }
        ]
    })


def _stub_generate_screens_response(context: str) -> str:
    """Stub response for generating video screens."""
    return json.dumps({
        "screens": [
            {
                "id": 1,
                "section_title": "Introduction",
                "on_screen_text": "Welcome to Study Notes",
                "voiceover_text": "Hey there! Let's dive into these study notes and break down the key concepts.",
                "image_prompt": "Clean, professional background with open book and light bulb, soft blue tones",
                "approx_duration_seconds": 5.0,
                "captions": [
                    {"time": 0.0, "text": "Welcome"},
                    {"time": 2.5, "text": "Study Notes"}
                ]
            },
            {
                "id": 2,
                "section_title": "Main Content",
                "on_screen_text": "Key Concepts",
                "voiceover_text": "Let's break down the main concepts. First, we have this important idea that connects to everything else.",
                "image_prompt": "Educational diagram with connected nodes, vibrant colors, modern design",
                "approx_duration_seconds": 7.0,
                "captions": [
                    {"time": 0.0, "text": "Key Concepts"},
                    {"time": 3.5, "text": "Important Ideas"}
                ]
            }
        ]
    })


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

ORGANIZE_NOTES_SYSTEM_PROMPT = """You are an expert educational content organizer. Transform raw notes into clean, structured format.

You MUST respond with valid JSON only, following this schema:

{
  "topic": "string - the main topic/subject",
  "sections": [
    {
      "title": "string - section title",
      "summary": "string - 1-2 sentence summary",
      "bullets": [
        {
          "text": "string - bullet point content",
          "importance": "high" | "medium" | "low"
        }
      ]
    }
  ]
}

Guidelines:
- Preserve all important information from original notes
- Organize logically into 2-6 sections
- Each section should have 2-8 bullet points
- Mark importance based on how critical the information is
- Clean up formatting but preserve technical terms, formulas, special characters
- Extract clear topic name even if not explicitly stated"""


def get_organize_notes_user_prompt(notes: str, topic: Optional[str], style: str) -> str:
    """Generate user prompt for organizing notes."""
    style_desc = STYLE_CONFIGS.get(style, STYLE_CONFIGS["study"])["description"]
    topic_instruction = f"\n\nThe user indicated this is about: {topic}" if topic else ""
    
    return f"""Please organize these raw notes into structured format suitable for a {style} style video ({style_desc}).

{topic_instruction}

Raw notes:
{notes}

Remember to output ONLY valid JSON following the schema provided."""


GENERATE_SCREENS_SYSTEM_PROMPT = """You are an expert educational video scriptwriter. Convert structured notes into video "screens".

You MUST respond with valid JSON only, following this schema:

{
  "screens": [
    {
      "id": integer - sequential starting from 1,
      "section_title": "string - which section this relates to",
      "on_screen_text": "string - concise text to display (5-10 words max)",
      "voiceover_text": "string - natural, conversational script for TTS",
      "image_prompt": "string - detailed prompt for AI image generation",
      "approx_duration_seconds": float - estimated time for this screen,
      "captions": [
        {
          "time": float - seconds from screen start,
          "text": "string - caption text"
        }
      ]
    }
  ]
}

Critical guidelines:
- on_screen_text: Short, punchy, easy to read at a glance
- voiceover_text: Natural spoken language, appropriate to the style
- image_prompt: Detailed visual description including colors, mood, style, specific elements
- Duration: Estimate based on voiceover length and pacing
- Captions: Break voiceover into 2-4 timed chunks per screen
- Create enough screens to cover all content within target duration"""


def get_generate_screens_user_prompt(
    structured_notes: Dict[str, Any],
    style: str,
    target_duration: int
) -> str:
    """Generate user prompt for creating video screens."""
    style_config = STYLE_CONFIGS.get(style, STYLE_CONFIGS["study"])
    
    return f"""Create video screens from these structured notes.

Style: {style} - {style_config['description']}
Target total duration: {target_duration} seconds
Image style guidance: {style_config['image_style']}
Pacing: {style_config['words_per_second']} words per second
Caption frequency: {style_config['caption_style']}

Structured notes:
{json.dumps(structured_notes, indent=2)}

Create a sequence of screens that:
1. Covers all key information from the notes
2. Flows naturally from one screen to the next
3. Matches the {style} style in tone and energy
4. Fits within approximately {target_duration} seconds total
5. Has engaging, descriptive image prompts for each screen

Remember to output ONLY valid JSON following the schema provided."""


def organize_notes_with_llm(notes: str, topic: Optional[str], style: str) -> Dict[str, Any]:
    """Use LLM to organize raw notes into structured format."""
    system_prompt = ORGANIZE_NOTES_SYSTEM_PROMPT
    user_prompt = get_organize_notes_user_prompt(notes, topic, style)
    
    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True
    )
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response: {e}")
        print(f"[ERROR] Response was: {response[:500]}")
        # Return minimal valid structure
        return {
            "topic": topic or "Study Notes",
            "sections": [
                {
                    "title": "Content",
                    "summary": "Notes content",
                    "bullets": [{"text": "Information from notes", "importance": "medium"}]
                }
            ]
        }


def generate_screens_with_llm(
    structured_notes: Dict[str, Any],
    style: str,
    target_duration: int
) -> list[Dict[str, Any]]:
    """Use LLM to generate video screens from structured notes."""
    system_prompt = GENERATE_SCREENS_SYSTEM_PROMPT
    user_prompt = get_generate_screens_user_prompt(structured_notes, style, target_duration)
    
    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True
    )
    
    try:
        data = json.loads(response)
        screens = data.get("screens", []) if isinstance(data, dict) else []
        # Fallback to stub screens if none were returned
        if not screens:
            return json.loads(_stub_generate_screens_response("fallback")).get("screens", [])
        return screens
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse LLM response: {e}")
        print(f"[ERROR] Response was: {response[:500]}")
        # Return minimal valid screens
        return [
            {
                "id": 1,
                "section_title": "Content",
                "on_screen_text": "Study Notes",
                "voiceover_text": "Here are your study notes.",
                "image_prompt": "Clean educational background",
                "approx_duration_seconds": 5.0,
                "captions": [{"time": 0.0, "text": "Study Notes"}]
            }
        ]
