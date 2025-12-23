from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def analyze_corpus(poems: List[str]) -> Dict:
    word_counts = []
    line_counts = []
    char_counts = []
    
    for poem in poems:
        words = poem.split()
        lines = [line.strip() for line in poem.split("\n") if line.strip()]
        
        word_counts.append(len(words))
        line_counts.append(len(lines))
        char_counts.append(len(poem))
    
    stats = {
        "total_poems": len(poems),
        "word_stats": {
            "mean": float(np.mean(word_counts)),
            "median": float(np.median(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "std": float(np.std(word_counts))
        },
        "line_stats": {
            "mean": float(np.mean(line_counts)),
            "median": float(np.median(line_counts)),
            "min": int(np.min(line_counts)),
            "max": int(np.max(line_counts)),
            "std": float(np.std(line_counts))
        },
        "char_stats": {
            "mean": float(np.mean(char_counts)),
            "median": float(np.median(char_counts)),
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
            "std": float(np.std(char_counts))
        },
        "raw_data": {
            "word_counts": word_counts,
            "line_counts": line_counts,
            "char_counts": char_counts
        }
    }
    
    return stats


def plot_distributions(stats: Dict, save_path: str = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    word_counts = stats["raw_data"]["word_counts"]
    line_counts = stats["raw_data"]["line_counts"]
    char_counts = stats["raw_data"]["char_counts"]
    
    axes[0].hist(word_counts, bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(stats["word_stats"]["mean"], color="red", linestyle="--", label=f"Moyenne: {stats['word_stats']['mean']:.1f}")
    axes[0].axvline(stats["word_stats"]["median"], color="green", linestyle="--", label=f"Médiane: {stats['word_stats']['median']:.1f}")
    axes[0].set_xlabel("Nombre de mots")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution du nombre de mots par poème")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(line_counts, bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[1].axvline(stats["line_stats"]["mean"], color="red", linestyle="--", label=f"Moyenne: {stats['line_stats']['mean']:.1f}")
    axes[1].axvline(stats["line_stats"]["median"], color="green", linestyle="--", label=f"Médiane: {stats['line_stats']['median']:.1f}")
    axes[1].set_xlabel("Nombre de lignes")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution du nombre de lignes par poème")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(char_counts, bins=30, edgecolor="black", alpha=0.7, color="purple")
    axes[2].axvline(stats["char_stats"]["mean"], color="red", linestyle="--", label=f"Moyenne: {stats['char_stats']['mean']:.1f}")
    axes[2].axvline(stats["char_stats"]["median"], color="green", linestyle="--", label=f"Médiane: {stats['char_stats']['median']:.1f}")
    axes[2].set_xlabel("Nombre de caractères")
    axes[2].set_ylabel("Fréquence")
    axes[2].set_title("Distribution du nombre de caractères par poème")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()

