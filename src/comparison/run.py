from typing import Dict, Any, Optional
from .helpers import (
    graph_to_prompt,
    load_finetuned_model,
    generate_with_unsloth,
    compute_metrics
)


def run_comparison(
    llm: Any,
    topic_graph: Dict[str, Any],
    form_spec: Dict[str, Any],
    generation_params: Dict[str, Any],
    finetuned_model_path: str = "./models/finetuned"
) -> Dict[str, Any]:
    finetuned_model, finetuned_tokenizer = load_finetuned_model(finetuned_model_path)
    
    prompt_baseline = f"Écris un {form_spec['form']} en français sur les thèmes de {', '.join(topic_graph['nodes'])}."
    structured_instructions = graph_to_prompt(topic_graph, form_spec)
    
    results = {}
    
    messages_baseline = [
        {"role": "system", "content": "You are a skilled poet."},
        {"role": "user", "content": prompt_baseline}
    ]
    
    output_baseline = llm.create_chat_completion(
        messages=messages_baseline,
        max_tokens=generation_params["max_tokens"],
        temperature=generation_params["temperature"],
        top_p=generation_params["top_p"],
    )
    poem_baseline = output_baseline["choices"][0]["message"]["content"]
    results["baseline"] = {
        "description": "Modèle générique + prompt simple",
        "poem": poem_baseline,
        "metrics": compute_metrics(poem_baseline, topic_graph)
    }
    
    messages_structure = [
        {"role": "system", "content": "You are a skilled poet. Write in French."},
        {"role": "user", "content": structured_instructions}
    ]
    
    output_structure = llm.create_chat_completion(
        messages=messages_structure,
        max_tokens=generation_params["max_tokens"],
        temperature=generation_params["temperature"],
        top_p=generation_params["top_p"],
    )
    poem_structure = output_structure["choices"][0]["message"]["content"]
    results["structure_only"] = {
        "description": "Modèle générique + graphe thématique",
        "poem": poem_structure,
        "metrics": compute_metrics(poem_structure, topic_graph)
    }
    
    if finetuned_model is not None:
        poem_specialization = generate_with_unsloth(
            finetuned_model,
            finetuned_tokenizer,
            prompt_baseline,
            max_tokens=generation_params["max_tokens"],
            temperature=generation_params["temperature"]
        )
        if not poem_specialization:
            poem_specialization = poem_baseline
    else:
        poem_specialization = poem_baseline
    
    results["specialization_only"] = {
        "description": "Modèle fine-tuné + prompt simple",
        "poem": poem_specialization,
        "metrics": compute_metrics(poem_specialization, topic_graph)
    }
    
    if finetuned_model is not None:
        poem_combined = generate_with_unsloth(
            finetuned_model,
            finetuned_tokenizer,
            structured_instructions,
            max_tokens=generation_params["max_tokens"],
            temperature=generation_params["temperature"]
        )
        if not poem_combined:
            poem_combined = poem_structure
    else:
        poem_combined = poem_structure
    
    results["structure_specialization"] = {
        "description": "Modèle fine-tuné + graphe thématique",
        "poem": poem_combined,
        "metrics": compute_metrics(poem_combined, topic_graph)
    }
    
    return results

