from typing import Any, Dict, Optional


def generate_revision_llama_cpp(llm: Any, original_poem: str, critique: str, 
                                original_prompt: str, form_spec: Dict[str, Any]) -> str:
    revision_prompt = f"""Voici un poème que tu as écrit, suivi d'une critique. 
Réécris le poème en tenant compte de la critique pour améliorer :
- La cohérence thématique
- La qualité des images poétiques
- La structure formelle ({form_spec.get('form', 'poème')})

Poème original :
{original_poem}

Critique :
{critique}

Génère une version révisée améliorée qui répond aux suggestions de la critique :"""

    messages = [
        {"role": "system", "content": "You are a skilled poet. Revise your work based on constructive feedback. Write in French."},
        {"role": "user", "content": revision_prompt}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=1000,
        temperature=0.8,
        top_p=0.95,
    )

    revised_text = output["choices"][0]["message"]["content"].strip()
    
    lines = revised_text.split("\n")
    cleaned_lines = []
    skip_intro = True
    skip_patterns = ["voici", "version révisée", "poème réécrit", "génère", "réécris", "tenant compte"]
    stop_patterns = ["critique:", "**", "points forts", "faiblesses", "suggestions"]
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        if skip_intro:
            if any(pattern in line_lower for pattern in skip_patterns):
                if ":" in line_stripped:
                    continue
            if line_stripped and not line_stripped.startswith("-") and len(line_stripped) > 15:
                if not any(pattern in line_lower for pattern in skip_patterns):
                    skip_intro = False
        
        if not skip_intro:
            if any(pattern in line_lower for pattern in stop_patterns):
                break
            if line_stripped.startswith("-") and len(line_stripped) < 50:
                continue
            cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    
    if not result or len(result) < 50:
        return revised_text
    
    return result


def generate_revision_unsloth(model: Any, tokenizer: Any, original_poem: str, 
                              critique: str, original_prompt: str, 
                              form_spec: Dict[str, Any]) -> Optional[str]:
    from src.comparison.helpers import generate_with_unsloth

    revision_prompt = f"""Voici un poème que tu as écrit, suivi d'une critique. 
Réécris le poème en tenant compte de la critique pour améliorer :
- La cohérence thématique
- La qualité des images poétiques
- La structure formelle ({form_spec.get('form', 'poème')})

Poème original :
{original_poem}

Critique :
{critique}

Génère une version révisée améliorée qui répond aux suggestions de la critique :"""

    revision = generate_with_unsloth(
        model, tokenizer, revision_prompt, max_tokens=1000, temperature=0.8
    )
    
    if revision:
        lines = revision.split("\n")
        cleaned_lines = []
        skip_intro = True
        skip_patterns = ["voici", "version révisée", "poème réécrit", "génère", "réécris", "tenant compte"]
        stop_patterns = ["critique:", "**", "points forts", "faiblesses", "suggestions"]
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            if skip_intro:
                if any(pattern in line_lower for pattern in skip_patterns):
                    if ":" in line_stripped:
                        continue
                if line_stripped and not line_stripped.startswith("-") and len(line_stripped) > 15:
                    if not any(pattern in line_lower for pattern in skip_patterns):
                        skip_intro = False
            
            if not skip_intro:
                if any(pattern in line_lower for pattern in stop_patterns):
                    break
                if line_stripped.startswith("-") and len(line_stripped) < 50:
                    continue
                cleaned_lines.append(line)
        
        result = "\n".join(cleaned_lines).strip()
        
        if not result or len(result) < 50:
            return revision
        
        return result
    
    return revision


def generate_revision(model: Any, original_poem: str, critique: str, 
                     original_prompt: str, form_spec: Dict[str, Any],
                     model_type: str = "generic", tokenizer: Any = None) -> str:
    if model_type == "generic":
        return generate_revision_llama_cpp(model, original_poem, 
                                          critique, original_prompt, 
                                          form_spec)
    elif model_type == "finetuned":
        if tokenizer is None:
            raise ValueError("Tokenizer required for finetuned model")
        result = generate_revision_unsloth(model, tokenizer, 
                                          original_poem, critique,
                                          original_prompt, form_spec)
        if result is None:
            raise RuntimeError("Failed to generate revision with finetuned model")
        return result
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

