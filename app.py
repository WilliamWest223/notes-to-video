"""FastAPI application for Notes-to-Video backend."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import traceback

from models import (
    HealthResponse,
    OrganizeNotesRequest,
    OrganizeNotesResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    VideoMetadata
)
from pipeline import run_organize_notes_pipeline, run_full_pipeline


# Initialize FastAPI app
app = FastAPI(
    title="Notes-to-Video API",
    description="Transform study notes into AI-generated educational videos",
    version="1.0.0"
)

# Add CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns server status and timestamp.
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/api/organize-notes", response_model=OrganizeNotesResponse)
async def organize_notes(request: OrganizeNotesRequest):
    """
    Organize raw notes into structured format.
    
    This endpoint runs Stage A of the pipeline:
    - Cleans and normalizes the input text
    - Uses LLM to organize into sections with bullets
    - Returns structured JSON
    """
    try:
        print(f"[API] Received organize-notes request")
        print(f"[API] Notes length: {len(request.notes)} characters")
        print(f"[API] Topic: {request.topic or 'Not provided'}")
        print(f"[API] Style: {request.style}")
        
        # Run the organize notes pipeline
        structured_notes = run_organize_notes_pipeline(
            notes=request.notes,
            topic=request.topic,
            style=request.style or "study"
        )
        
        return OrganizeNotesResponse(
            status="ok",
            structured_notes=structured_notes
        )
        
    except Exception as e:
        print(f"[API ERROR] Failed to organize notes: {e}")
        print(traceback.format_exc())
        return OrganizeNotesResponse(
            status="error",
            error_message=f"Failed to organize notes: {str(e)}"
        )


@app.post("/api/generate-video", response_model=GenerateVideoResponse)
async def generate_video(request: GenerateVideoRequest):
    """
    Generate complete video from raw notes.
    
    This endpoint runs the full pipeline (Stages A-D):
    - Stage A: Organize notes into structured format
    - Stage B: Generate video screens with scripts and image prompts
    - Stage C: Generate TTS audio
    - Stage D: Assemble final video with captions
    
    Returns video path, screens, structured notes, and metadata.
    """
    try:
        print(f"[API] Received generate-video request")
        print(f"[API] Notes length: {len(request.notes)} characters")
        print(f"[API] Topic: {request.topic or 'Not provided'}")
        print(f"[API] Style: {request.style}")
        print(f"[API] Target duration: {request.target_duration_seconds}s")
        
        # Run the full pipeline
        video_path, structured_notes, screens, metadata_dict = run_full_pipeline(
            notes=request.notes,
            topic=request.topic,
            style=request.style,
            target_duration_seconds=request.target_duration_seconds
        )
        
        # Create metadata object
        metadata = VideoMetadata(**metadata_dict)
        
        return GenerateVideoResponse(
            status="completed",
            video_path=video_path,
            screens=screens,
            structured_notes=structured_notes,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"[API ERROR] Failed to generate video: {e}")
        print(traceback.format_exc())
        return GenerateVideoResponse(
            status="error",
            error_message=f"Failed to generate video: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Notes-to-Video API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /api/health",
            "organize_notes": "POST /api/organize-notes",
            "generate_video": "POST /api/generate-video",
            "docs": "GET /docs (Swagger UI)",
            "redoc": "GET /redoc (ReDoc)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Notes-to-Video API server...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
