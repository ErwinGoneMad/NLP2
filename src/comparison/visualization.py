import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from typing import Dict, Any


def visualize_comparison(results: Dict[str, Any]):
    poems_data = [
        (results["baseline"]["poem"], "1. Baseline\n(Modèle générique + prompt simple)"),
        (results["structure_only"]["poem"], "2. Structure only\n(Modèle générique + graphe)"),
        (results["specialization_only"]["poem"], "3. Specialization only\n(Modèle fine-tuné + prompt simple)"),
        (results["structure_specialization"]["poem"], "4. Structure + Specialization\n(Modèle fine-tuné + graphe)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Comparaison des 4 Approches", fontsize=16, fontweight="bold")
    
    for idx, (poem, title) in enumerate(poems_data):
        ax = axes[idx // 2, idx % 2]
        
        # Wrapper le texte avec une largeur appropriée
        wrapped_lines = []
        for line in poem.split("\n"):
            if len(line) > 70:
                wrapped_lines.extend(textwrap.wrap(line, width=70))
            else:
                wrapped_lines.append(line)
        
        # Limiter le nombre de lignes pour éviter la superposition (max ~30 lignes)
        max_lines = 30
        if len(wrapped_lines) > max_lines:
            wrapped_lines = wrapped_lines[:max_lines]
            wrapped_lines.append(f"\n... (texte tronqué, {len(poem.split())} mots au total)")
        
        wrapped_poem = "\n".join(wrapped_lines)
        
        # Positionner le texte avec un espace pour le titre
        ax.text(0.02, 0.98, wrapped_poem, transform=ax.transAxes, 
                fontsize=8, verticalalignment="top", horizontalalignment="left",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3, pad=0.5))
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97], h_pad=2.0, w_pad=2.0)
    plt.show()
    
    metrics_list = []
    approach_names = ["Baseline", "Structure only", "Specialization only", "Structure + Specialization"]
    approach_keys = ["baseline", "structure_only", "specialization_only", "structure_specialization"]
    
    for name, key in zip(approach_names, approach_keys):
        m = results[key]["metrics"].copy()
        m["Approche"] = name
        metrics_list.append(m)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Approche", "word_count", "line_count", "char_count", "vocab_size", "avg_line_length"]]
    df_metrics.columns = ["Approche", "Mots", "Lignes", "Caractères", "Vocabulaire unique", "Longueur moyenne ligne"]
    
    print("Tableau des métriques :")
    print(df_metrics.to_string(index=False))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Comparaison des Métriques", fontsize=16, fontweight="bold")
    
    metrics_to_plot = ["Mots", "Lignes", "Caractères", "Vocabulaire unique", "Longueur moyenne ligne"]
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(df_metrics["Approche"], df_metrics[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("Valeur")
        ax.tick_params(axis="x", rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{int(height)}", ha="center", va="bottom", fontsize=9)
    
    axes[1, 2].axis("off")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

