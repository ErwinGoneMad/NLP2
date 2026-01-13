"""Module pour l'évaluation des poèmes par un LLM juge (Gemini 3)."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import re
from datetime import datetime
import requests
import os


def load_poems_from_results(results_dir: Path = Path("results")) -> Dict[str, List[Dict[str, Any]]]:
    poems = {
        "comparison": [],
        "iterative": []
    }
    
    comparison_dir = results_dir / "comparison"
    if comparison_dir.exists():
        for json_file in comparison_dir.glob("comparison_*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", json_file.stem)
                    for approach_name, approach_data in data.get("approaches", {}).items():
                        poems["comparison"].append({
                            "source_file": str(json_file),
                            "timestamp": timestamp,
                            "approach": approach_name,
                            "description": approach_data.get("description", ""),
                            "poem": approach_data.get("poem", ""),
                            "metrics": approach_data.get("metrics", {})
                        })
            except Exception:
                pass
    
    iterative_dir = results_dir / "iterative"
    if iterative_dir.exists():
        for json_file in iterative_dir.glob("iterative_*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", json_file.stem)
                    results = data.get("results", {})
                    for approach_name, approach_data in results.items():
                        if "final" in approach_data:
                            poems["iterative"].append({
                                "source_file": str(json_file),
                                "timestamp": timestamp,
                                "approach": approach_name,
                                "version": "final",
                                "poem": approach_data.get("final", ""),
                                "metrics": approach_data.get("final_metrics", {})
                            })
            except Exception:
                pass
    
    return poems


def evaluate_poem_with_gemini(
    poem: str,
    topic_graph: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY non trouvée")
    
    topics = ""
    if topic_graph:
        topics = f"Thèmes attendus: {', '.join(topic_graph.get('nodes', []))}\n\n"
    
    evaluation_prompt = f"""Tu es un expert en poésie française. Évalue ce poème selon les critères suivants :

{topics}Poème à évaluer :
{poem}

Évalue ce poème sur une échelle de 0 à 10 pour chaque critère :
1. Qualité poétique globale (rythme, musicalité, images)
2. Cohérence thématique (respect des thèmes demandés)
3. Originalité et créativité
4. Maîtrise de la langue française

Réponds UNIQUEMENT au format JSON suivant (sans texte supplémentaire) :
{{
    "overall_score": <score 0-10>,
    "coherence_score": <score 0-10>,
    "poetic_quality": <score 0-10>,
    "theme_adherence": <score 0-10>,
    "originality": <score 0-10>,
    "language_mastery": <score 0-10>,
    "feedback": "<commentaire détaillé en français>"
}}"""
    
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:streamGenerateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": evaluation_prompt
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
        stream=True
    )
    
    if response.status_code != 200:
        raise Exception(f"Erreur API: {response.status_code} - {response.text}")
    
    full_text = ""
    buffer = ""
    
    for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
        if chunk:
            buffer += chunk
    
    buffer = buffer.strip()
    if buffer.startswith('['):
        buffer = buffer[1:]
    if buffer.endswith(']'):
        buffer = buffer[:-1]
    
    objects = []
    current_obj = ""
    depth = 0
    in_string = False
    escape_next = False
    
    for char in buffer:
        if escape_next:
            current_obj += char
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            current_obj += char
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            current_obj += char
            continue
        
        if not in_string:
            if char == '{':
                depth += 1
                current_obj += char
            elif char == '}':
                depth -= 1
                current_obj += char
                if depth == 0:
                    objects.append(current_obj.strip().rstrip(','))
                    current_obj = ""
            elif char == ',' and depth == 0:
                if current_obj.strip():
                    objects.append(current_obj.strip())
                current_obj = ""
            else:
                current_obj += char
        else:
            current_obj += char
    
    if current_obj.strip():
        objects.append(current_obj.strip().rstrip(','))
    
    for obj_str in objects:
        obj_str = obj_str.strip().rstrip(',')
        if not obj_str:
            continue
        try:
            chunk_data = json.loads(obj_str)
            if "candidates" in chunk_data and len(chunk_data["candidates"]) > 0:
                if "content" in chunk_data["candidates"][0]:
                    parts = chunk_data["candidates"][0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        full_text += parts[0]["text"]
        except (json.JSONDecodeError, KeyError):
            continue
    
    if not full_text:
        raise Exception("Aucune réponse de l'API")
    
    response_text = full_text.strip()
    
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    try:
        evaluation = json.loads(response_text)
    except json.JSONDecodeError as e:
        # Essayer de nettoyer le JSON en extrayant seulement la partie JSON valide
        try:
            # Extraire le JSON entre les premières { et dernières }
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response_text[start:end+1]
                # Nettoyer les échappements problématiques : remplacer \ suivi d'un caractère non-échappable par \\
                json_str = re.sub(r'\\(?![nrtbf"\\/u0123456789abcdefABCDEF])', r'\\\\', json_str)
                evaluation = json.loads(json_str)
            else:
                raise e
        except (json.JSONDecodeError, ValueError):
            # Si ça échoue encore, essayer d'extraire les valeurs avec regex
            evaluation = {
                "overall_score": 0,
                "coherence_score": 0,
                "poetic_quality": 0,
                "theme_adherence": 0,
                "originality": 0,
                "language_mastery": 0,
                "feedback": "Erreur lors du parsing de la réponse JSON"
            }
            # Essayer d'extraire les scores avec regex
            for key in ["overall_score", "coherence_score", "poetic_quality", "theme_adherence", "originality", "language_mastery"]:
                match = re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', response_text)
                if match:
                    try:
                        evaluation[key] = float(match.group(1))
                    except (ValueError, AttributeError):
                        pass
            # Extraire le feedback (gérer les échappements)
            feedback_match = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', response_text, re.DOTALL)
            if feedback_match:
                feedback_text = feedback_match.group(1)
                feedback_text = feedback_text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                evaluation["feedback"] = feedback_text
    
    return evaluation


def evaluate_all_poems_from_results(
    results_dir: Path = Path("results"),
    api_key: Optional[str] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    poems = load_poems_from_results(results_dir)
    
    all_evaluations = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "comparison_evaluations": [],
        "iterative_evaluations": []
    }
    
    for poem_data in poems["comparison"]:
        topic_graph = None
        try:
            with open(poem_data["source_file"], "r", encoding="utf-8") as f:
                source_data = json.load(f)
                topic_graph = source_data.get("config", {}).get("topic_graph")
        except:
            pass
        
        evaluation = evaluate_poem_with_gemini(
            poem_data["poem"],
            topic_graph,
            api_key
        )
        
        all_evaluations["comparison_evaluations"].append({
            **poem_data,
            "llm_evaluation": evaluation
        })
    
    for poem_data in poems["iterative"]:
        evaluation = evaluate_poem_with_gemini(
            poem_data["poem"],
            None,
            api_key
        )
        
        all_evaluations["iterative_evaluations"].append({
            **poem_data,
            "llm_evaluation": evaluation
        })
    
    if save_results:
        output_dir = results_dir / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"llm_evaluations_{all_evaluations['timestamp']}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_evaluations, f, ensure_ascii=False, indent=2)
    
    return all_evaluations
