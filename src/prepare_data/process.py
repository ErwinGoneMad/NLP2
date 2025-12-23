import re
from pathlib import Path
from typing import List, Dict
import json


def process_corpus(raw_data: List[Dict], output_path: Path = None) -> List[str]:
    if output_path is None:
        output_path = Path("./data/processed/french_poetry_processed.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cleaned_poems = []
    
    for item in raw_data:
        content = item.get("content", "")
        if not content:
            continue
        
        lines = content.split("\n")
        cleaned_lines = []
        in_poem = False
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if in_poem:
                    cleaned_lines.append("")
                continue
            
            if not in_poem:
                if (line_stripped.startswith("Poésie :") or 
                    line_stripped.startswith("Titre :") or 
                    line_stripped.startswith("Poète :") or 
                    line_stripped.startswith("Recueil :") or
                    line_stripped == "Sonnet." or
                    line_stripped.startswith("Ode ") or
                    line_stripped.startswith("Chanson") or
                    len(line_stripped) < 3):
                    continue
                else:
                    in_poem = True
            
            if in_poem:
                cleaned_lines.append(line_stripped)
        
        poem_text = "\n".join(cleaned_lines) if cleaned_lines else content
        lines = [line.strip() for line in poem_text.split("\n") if line.strip()]
        if len(lines) >= 4:
            cleaned_poems.append("\n".join(lines))
    
    output_data = {
        "poems": cleaned_poems,
        "count": len(cleaned_poems)
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return cleaned_poems

