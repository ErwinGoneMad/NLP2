from pathlib import Path
from typing import Optional
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(history_file: str = "./results/training_history.json", save_path: Optional[str] = None):
    history_path = Path(history_file)
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_file}")
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    steps = []
    losses = []
    learning_rates = []
    
    for entry in history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
            if "learning_rate" in entry:
                learning_rates.append((entry["step"], entry["learning_rate"]))
    
    if not steps:
        raise ValueError("No loss data found in history")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Évolution de la Loss pendant l'entraînement")
    ax1.grid(True, alpha=0.3)
    
    if len(steps) > 1:
        z = np.polyfit(steps, losses, 1)
        p = np.poly1d(z)
        ax1.plot(steps, p(steps), "r--", alpha=0.5, label=f"Tendance (pente: {z[0]:.4f})")
        ax1.legend()
    
    if learning_rates:
        lr_steps, lr_values = zip(*learning_rates)
        ax2.plot(lr_steps, lr_values, marker="s", markersize=3, linewidth=1.5, color="green")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Évolution du Learning Rate")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Pas de données de learning rate", 
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Learning Rate (non disponible)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()

