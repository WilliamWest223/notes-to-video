"""Simple test script for the API."""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_organize_notes():
    """Test organize notes endpoint."""
    print("\n=== Testing Organize Notes Endpoint ===")
    payload = {
        "notes": """
        Photosynthesis Notes
        
        Process where plants convert light into energy
        - Happens in chloroplasts
        - Uses CO2 and water
        - Produces glucose and oxygen
        
        Two stages:
        1. Light-dependent reactions
        2. Calvin cycle (light-independent)
        """,
        "topic": "Biology - Photosynthesis",
        "style": "study"
    }
    response = requests.post(f"{BASE_URL}/api/organize-notes", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_generate_video():
    """Test generate video endpoint."""
    print("\n=== Testing Generate Video Endpoint ===")
    payload = {
        "notes": """
        Newton's Laws of Motion
        
        First Law: An object at rest stays at rest, object in motion stays in motion
        unless acted upon by external force (inertia)
        
        Second Law: F = ma (force equals mass times acceleration)
        
        Third Law: For every action, there is an equal and opposite reaction
        """,
        "topic": "Physics - Newton's Laws",
        "style": "study",
        "target_duration_seconds": 60
    }
    response = requests.post(f"{BASE_URL}/api/generate-video", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Status: {result.get('status')}")
    if result.get("status") != "completed":
        print(f"Error: {result.get('error_message')}")
        return
    print(f"Video Path: {result.get('video_path')}")
    print(f"Screen Count: {result.get('metadata', {}).get('screen_count')}")
    print(f"Duration: {result.get('metadata', {}).get('estimated_total_duration_seconds')}s")


if __name__ == "__main__":
    print("Testing Notes-to-Video API")
    print("Make sure the server is running first!")
    
    test_health()
    test_organize_notes()
    test_generate_video()
