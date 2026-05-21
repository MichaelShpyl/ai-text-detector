"""
Shared preprocessing utilities used by both the training pipeline and the
serving layer. Keeping these in one module is what eliminates training-serving
skew (Padlet stage 4 risk, cf. John et al. 2021 Release Pipeline).
"""
import re
from typing import List

MAX_TOKENS = 512
MIN_CHARS = 20


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove zero-width and control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\ufeff]", "", text)
    return text.strip()


def is_valid_length(text: str) -> bool:
    """Reject inputs that are too short to classify reliably."""
    return len(text.strip()) >= MIN_CHARS


def chunk_text(text: str, chunk_chars: int = 1500) -> List[str]:
    """Split very long inputs into overlapping chunks for aggregation at inference time.
    Each chunk is roughly ~512 tokens for English text."""
    text = clean_text(text)
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    overlap = 200
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks
