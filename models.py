"""Pydantic models for request/response validation."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class BulletPoint(BaseModel):
    """A single bullet point within a section."""
    text: str
    importance: Literal["high", "medium", "low"] = "medium"


class Section(BaseModel):
    """A section of organized notes."""
    title: str
    summary: str
    bullets: List[BulletPoint] = []


class StructuredNotes(BaseModel):
    """Organized notes with sections and structure."""
    topic: str
    sections: List[Section]


class Caption(BaseModel):
    """A timed caption for video overlay."""
    time: float = Field(..., description="Time in seconds when caption appears")
    text: str


class VideoScreen(BaseModel):
    """A single screen/scene in the video."""
    id: int
    section_title: str
    on_screen_text: str = Field(..., description="Main text displayed on screen")
    voiceover_text: str = Field(..., description="Script for text-to-speech")
    image_prompt: str = Field(..., description="Prompt for AI image generation")
    approx_duration_seconds: float
    captions: List[Caption] = []
    background_image_path: Optional[str] = None


class OrganizeNotesRequest(BaseModel):
    """Request to organize raw notes."""
    notes: str
    topic: Optional[str] = None
    style: Optional[Literal["brainrot", "study", "professor"]] = "study"


class OrganizeNotesResponse(BaseModel):
    """Response from organizing notes."""
    status: Literal["ok", "error"]
    structured_notes: Optional[StructuredNotes] = None
    error_message: Optional[str] = None


class GenerateVideoRequest(BaseModel):
    """Request to generate a complete video from notes."""
    notes: str
    topic: Optional[str] = None
    style: Literal["brainrot", "study", "professor"] = "study"
    target_duration_seconds: int = Field(default=90, ge=30, le=300)


class VideoMetadata(BaseModel):
    """Metadata about the generated video."""
    video_style: str
    target_duration_seconds: int
    estimated_total_duration_seconds: float
    screen_count: int
    generated_at: str


class GenerateVideoResponse(BaseModel):
    """Response from video generation."""
    status: Literal["completed", "error"]
    video_path: Optional[str] = None
    screens: Optional[List[VideoScreen]] = None
    structured_notes: Optional[StructuredNotes] = None
    metadata: Optional[VideoMetadata] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "1.0.0"
