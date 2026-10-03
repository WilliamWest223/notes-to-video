"""Text cleaning and normalization utilities."""
import re
import unicodedata
from typing import Optional


def clean_notes(raw_notes: str) -> str:
    """Clean and normalize raw notes while preserving important characters."""
    if not raw_notes:
        return ""
    
    # Normalize Unicode characters (NFC form)
    text = unicodedata.normalize('NFC', raw_notes)
    
    # Replace multiple consecutive whitespace with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def sanitize_for_filename(text: str, max_length: int = 50) -> str:
    """Sanitize text for use in filenames."""
    # Remove forbidden characters: / \ : * ? " < > |
    text = re.sub(r'[/\\:*?"<>|]', '_', text)
    text = text.replace(' ', '_')
    
    # Remove non-ASCII or control characters
    text = ''.join(char for char in text if ord(char) >= 32 and ord(char) < 127)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove trailing dots or spaces
    text = text.rstrip('. ')
    
    return text or "untitled"


def estimate_reading_time(text: str, words_per_second: float = 2.5) -> float:
    """Estimate how long it takes to read/speak text."""
    words = len(text.split())
    duration = words / words_per_second
    duration *= 1.1  # Add 10% buffer for pauses
    return max(duration, 1.0)


def extract_topic_from_notes(notes: str, provided_topic: Optional[str] = None) -> str:
    """Extract or infer the topic from notes."""
    if provided_topic:
        return provided_topic.strip()
    
    # Try to extract from first line if it looks like a title
    lines = notes.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        
        # If first line is short and looks like a title
        if len(first_line) < 100 and len(first_line.split()) < 15:
            first_line = re.sub(r'^(notes?|topic|subject|title):\s*', '', 
                              first_line, flags=re.IGNORECASE)
            if first_line:
                return first_line
    
    return "Study Notes"
