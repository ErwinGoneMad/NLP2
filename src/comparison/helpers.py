from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def graph_to_prompt(graph: Dict[str, Any], form_spec: Dict[str, Any]) -> str:
    nodes = graph["nodes"]
    edges = graph["edges"]
    form = form_spec["form"]
    total_lines = form_spec["total_lines"]
    
    num_sections = len(nodes) + len(edges) + 1
    lines_per_section = total_lines // num_sections
    remainder = total_lines % num_sections
    
    prompt_parts = [
        f"You will write a {total_lines}-line {form} in French.",
        "Follow this structured plan for each section:",
        ""
    ]
    
    line_num = 1
    
    for i, node in enumerate(nodes):
        end_line = line_num + lines_per_section - 1
        if i < remainder:
            end_line += 1
        prompt_parts.append(f"Lines {line_num}-{end_line}: Focus on '{node}' imagery and emotions.")
        line_num = end_line + 1
    
    for i, (source, target) in enumerate(edges):
        end_line = line_num + lines_per_section - 1
        if len(nodes) + i < remainder:
            end_line += 1
        prompt_parts.append(f"Lines {line_num}-{end_line}: Transition from '{source}' to '{target}'.")
        line_num = end_line + 1
    
    if line_num <= total_lines:
        remaining_lines = total_lines - line_num + 1
        all_nodes = ", ".join(nodes)
        prompt_parts.append(f"Lines {line_num}-{total_lines}: Synthesize all themes ({all_nodes}) in a concluding reflection.")
    
    return "\n".join(prompt_parts)


def load_finetuned_model(model_path: str = "./models/finetuned") -> Tuple[Optional[Any], Optional[Any]]:
    try:
        from unsloth import FastLanguageModel
        path = Path(model_path)
        if not path.exists():
            return None, None
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        return model, tokenizer
    except Exception:
        return None, None


def generate_with_unsloth(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 1000, temperature: float = 0.8) -> Optional[str]:
    if model is None or tokenizer is None:
        return None
    
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=eos_token_id,
        eos_token_id=eos_token_id,
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if assistant_start in generated_text:
        poem = generated_text.split(assistant_start)[1]
        poem = poem.split("<|eot_id|>")[0].strip()
        return poem
    return generated_text


def compute_metrics(
    poem_text: str, 
    topic_graph: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Calcule les métriques de base et avancées pour un poème.
    
    Args:
        poem_text: Le texte du poème
        topic_graph: Dictionnaire avec 'nodes' et 'edges' (optionnel, pour métriques avancées)
    
    Returns:
        Dictionnaire avec toutes les métriques (base + avancées si topic_graph fourni)
    """
    if not poem_text:
        return {}
    
    # Métriques de base (toujours calculées)
    lines = [line.strip() for line in poem_text.split("\n") if line.strip()]
    words = poem_text.split()
    chars = len(poem_text)
    unique_words = len(set(word.lower().strip(".,!?;:()[]{}") for word in words))
    
    base_metrics = {
        "word_count": len(words),
        "line_count": len(lines),
        "char_count": chars,
        "vocab_size": unique_words,
        "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
    }
    
    # Si topic_graph fourni, calculer les métriques avancées
    if topic_graph is not None:
        try:
            from src.metrics.advanced_metrics import compute_all_advanced_metrics
            
            # Charger le modèle d'embeddings (singleton)
            from src.metrics.embeddings import load_embedding_model
            embedding_model = load_embedding_model()
            
            # Calculer toutes les métriques avancées
            advanced_metrics = compute_all_advanced_metrics(
                poem_text, 
                topic_graph, 
                embedding_model
            )
            
            # Fusionner avec les métriques de base
            base_metrics.update(advanced_metrics)
        except Exception as e:
            print(f"Erreur lors du calcul des métriques avancées: {e}")
            # En cas d'erreur, retourner seulement les métriques de base
    
    return base_metrics

