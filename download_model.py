#!/usr/bin/env python3
"""
Script pour télécharger le modèle Llama-3.2-1B-Instruct quantifié (GGUF)
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

# Modèle SOTA 1B, quantifié à 4-bits (Q4_K_M)
REPO_ID = "MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF"
FILENAME = "Llama-3.2-1B-Instruct.Q4_K_M.gguf"

# Créer le dossier models s'il n'existe pas
models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

print(f"Téléchargement de {FILENAME}...")
print(f"Depuis le dépôt: {REPO_ID}")
print("Cela peut prendre quelques minutes selon votre connexion...\n")

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(models_dir.absolute())
    )
    print(f"\n✅ Succès! Modèle sauvegardé dans: {model_path}")
except Exception as e:
    print(f"\nerreur lors du téléchargement: {e}")
    exit(1)

