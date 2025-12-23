import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from typing import Dict, Any


def visualize_iterative_results(results: Dict[str, Any], approach_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Évolution Itérative - {approach_name}", fontsize=16, fontweight="bold")

    original = results["original"]
    iterations = results.get("iterations", [])

    poems_data = [
        (original, "Poème Original"),
    ]

    for iteration in iterations[:2]:
        if iteration.get("revised"):
            poems_data.append((iteration["revised"], f"Itération {iteration['iteration']}"))
        else:
            poems_data.append((f"Erreur: {iteration.get('error', 'Unknown')}", f"Itération {iteration['iteration']} - Erreur"))

    while len(poems_data) < 4:
        poems_data.append(("", ""))

    for idx, (poem, title) in enumerate(poems_data):
        ax = axes[idx // 2, idx % 2]
        
        if poem and not poem.startswith("Erreur"):
            poem_clean = poem.split("Critique:")[0].split("**")[0].strip()
            poem_clean = "\n".join([line for line in poem_clean.split("\n") 
                                   if not (line.strip().startswith("-") and len(line.strip()) < 50)
                                   and not any(keyword in line.lower() for keyword in ["voici", "version révisée", "points forts", "faiblesses"])])
            poem_clean = "\n".join([line for line in poem_clean.split("\n") if line.strip()])
            wrapped_poem = "\n".join([textwrap.fill(line, width=60) if len(line) > 60 else line 
                                      for line in poem_clean.split("\n")])
            ax.text(0.05, 0.95, wrapped_poem, transform=ax.transAxes, 
                    fontsize=9, verticalalignment="top", 
                    family="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        elif poem and poem.startswith("Erreur"):
            ax.text(0.5, 0.5, poem, transform=ax.transAxes, ha="center", va="center",
                   fontsize=10, color="red")
        else:
            ax.axis("off")
        
        ax.set_title(title, fontsize=12, fontweight="bold")
        if poem:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    metrics_data = []
    metrics_data.append({
        "Version": "Original",
        "Mots": results["original_metrics"]["word_count"],
        "Lignes": results["original_metrics"]["line_count"],
        "Vocabulaire": results["original_metrics"]["vocab_size"]
    })

    for iteration in iterations:
        if iteration.get("metrics"):
            metrics_data.append({
                "Version": f"Itération {iteration['iteration']}",
                "Mots": iteration["metrics"]["word_count"],
                "Lignes": iteration["metrics"]["line_count"],
                "Vocabulaire": iteration["metrics"]["vocab_size"]
            })

    if metrics_data:
        df = pd.DataFrame(metrics_data)
        print(f"\nÉvolution des métriques - {approach_name}:")
        print(df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(metrics_data))
        width = 0.25

        ax.bar([i - width for i in x], [m["Mots"] for m in metrics_data], 
               width, label="Mots", color="#1f77b4")
        ax.bar(x, [m["Lignes"] for m in metrics_data], 
               width, label="Lignes", color="#ff7f0e")
        ax.bar([i + width for i in x], [m["Vocabulaire"] for m in metrics_data], 
               width, label="Vocabulaire", color="#2ca02c")

        ax.set_xlabel("Version")
        ax.set_ylabel("Valeur")
        ax.set_title(f"Évolution des Métriques - {approach_name}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m["Version"] for m in metrics_data], rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def visualize_all_iterative_results(all_results: Dict[str, Dict[str, Any]]):
    for approach_name, results in all_results.items():
        visualize_iterative_results(results, approach_name)

