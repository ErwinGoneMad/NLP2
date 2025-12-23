from typing import Any, Dict, Optional


def generate_critique_llama_cpp(llm: Any, poem: str, form_spec: Dict[str, Any]) -> str:
    critique_prompt = f"""Tu es un critique poétique expert. Analyse ce poème et identifie :
1. Les points forts
2. Les faiblesses (cohérence, imagerie, structure formelle)
3. Des suggestions d'amélioration concrètes

Poème à analyser :
{poem}

Fournis une critique constructive et détaillée :"""

    messages = [
        {"role": "system", "content": "You are an expert poetry critic. Provide detailed, constructive feedback in French."},
        {"role": "user", "content": critique_prompt}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
    )

    return output["choices"][0]["message"]["content"].strip()


def generate_critique_unsloth(model: Any, tokenizer: Any, poem: str, form_spec: Dict[str, Any]) -> Optional[str]:
    from src.comparison.helpers import generate_with_unsloth

    critique_prompt = f"""Tu es un critique poétique expert. Analyse ce poème et identifie :
1. Les points forts
2. Les faiblesses (cohérence, imagerie, structure formelle)
3. Des suggestions d'amélioration concrètes

Poème à analyser :
{poem}

Fournis une critique constructive et détaillée :"""

    critique = generate_with_unsloth(
        model, tokenizer, critique_prompt, max_tokens=500, temperature=0.7
    )

    return critique


def generate_critique(model: Any, poem: str, form_spec: Dict[str, Any], 
                     model_type: str = "generic", tokenizer: Any = None) -> str:
    if model_type == "generic":
        return generate_critique_llama_cpp(model, poem, form_spec)
    elif model_type == "finetuned":
        if tokenizer is None:
            raise ValueError("Tokenizer required for finetuned model")
        result = generate_critique_unsloth(model, tokenizer, poem, form_spec)
        if result is None:
            raise RuntimeError("Failed to generate critique with finetuned model")
        return result
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

