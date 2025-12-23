from pathlib import Path
from typing import List, Dict


def download_french_poetry() -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")
    
    ds = load_dataset("manu/french_poetry")
    dataset = ds["train"]
    
    poems_data = []
    for item in dataset:
        poem_text = item.get("text", "").strip()
        if poem_text and len(poem_text) > 50:
            poems_data.append({
                "title": item.get("title", ""),
                "poet": item.get("poet", ""),
                "content": poem_text,
                "link": item.get("link", ""),
                "id": item.get("id", "")
            })
    
    return poems_data

