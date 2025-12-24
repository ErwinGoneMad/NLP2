from typing import Any, Dict, List, Optional
from .critique import generate_critique
from .revision import generate_revision
from src.comparison.helpers import compute_metrics


def run_iterative_generation(
    llm: Any,
    poem: str,
    prompt: str,
    form_spec: Dict[str, Any],
    generation_params: Dict[str, Any],
    num_iterations: int = 2,
    model_type: str = "generic",
    finetuned_model: Any = None,
    finetuned_tokenizer: Any = None,
    topic_graph: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    result = {
        "original": poem,
        "original_metrics": compute_metrics(poem, topic_graph),
        "iterations": []
    }

    current_poem = poem
    model = llm if model_type == "generic" else finetuned_model
    tokenizer = finetuned_tokenizer if model_type == "finetuned" else None

    for i in range(num_iterations):
        try:
            critique = generate_critique(
                model=model,
                poem=current_poem,
                form_spec=form_spec,
                model_type=model_type,
                tokenizer=tokenizer
            )

            revised_poem = generate_revision(
                model=model,
                original_poem=current_poem,
                critique=critique,
                original_prompt=prompt,
                form_spec=form_spec,
                model_type=model_type,
                tokenizer=tokenizer
            )

            iteration_result = {
                "iteration": i + 1,
                "critique": critique,
                "revised": revised_poem,
                "metrics": compute_metrics(revised_poem, topic_graph)
            }

            result["iterations"].append(iteration_result)
            current_poem = revised_poem

        except Exception as e:
            iteration_result = {
                "iteration": i + 1,
                "error": str(e),
                "critique": None,
                "revised": None
            }
            result["iterations"].append(iteration_result)
            break

    if result["iterations"]:
        result["final"] = result["iterations"][-1]["revised"]
        result["final_metrics"] = result["iterations"][-1]["metrics"]
    else:
        result["final"] = poem
        result["final_metrics"] = result["original_metrics"]

    return result

