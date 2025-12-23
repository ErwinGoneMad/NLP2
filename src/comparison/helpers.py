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


def compute_metrics(poem_text: str) -> Dict[str, float]:
    if not poem_text:
        return {}
    
    lines = [line.strip() for line in poem_text.split("\n") if line.strip()]
    words = poem_text.split()
    chars = len(poem_text)
    unique_words = len(set(word.lower().strip(".,!?;:()[]{}") for word in words))
    
    return {
        "word_count": len(words),
        "line_count": len(lines),
        "char_count": chars,
        "vocab_size": unique_words,
        "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
    }

