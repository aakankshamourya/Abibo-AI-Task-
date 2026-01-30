"""
Lightweight image captioning for Telegram bot (/image command).
Uses Hugging Face BLIP for caption + simple keyword extraction for tags.
"""
import os
import re
from pathlib import Path
from typing import List, Tuple

# Lazy-load PyTorch BLIP only (avoids TensorFlow/Keras import)
_model = None
_processor = None
_device = None

def _get_captioner():
    """Load BLIP model and processor with PyTorch only (no TF/Keras). Kept on CPU to avoid .device setter issues in some transformers versions."""
    global _model, _processor, _device
    if _model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _device = "cpu"
    return _model, _processor, _device

def caption_image(image_path: str) -> Tuple[str, List[str]]:
    """
    Generate a short caption and 3 tags for an image.
    Returns (caption_string, list_of_3_keyword_strings).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    from PIL import Image
    import torch
    model, processor, device = _get_captioner()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True).strip() or "No caption generated."
    tags = _extract_tags(caption)
    return caption, tags

def _extract_tags(caption: str, n: int = 3) -> List[str]:
    """Extract up to n keyword-like tokens from caption (nouns/short phrases)."""
    # Remove punctuation, lowercase, split
    text = re.sub(r"[^\w\s]", " ", caption).lower()
    words = text.split()
    # Stopwords to drop
    stop = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "can", "this", "that",
        "these", "those", "it", "its", "there", "here",
    }
    # Prefer longer words as "tags"
    candidates = [w for w in words if w not in stop and len(w) > 1]
    if not candidates:
        # Fallback: first 3 words
        candidates = words[:n]
    # Deduplicate preserving order, take first n
    seen = set()
    tags = []
    for w in candidates:
        if w not in seen and len(tags) < n:
            seen.add(w)
            tags.append(w)
    return tags[:n]
