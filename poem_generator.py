#!/usr/bin/env python3
"""
Phase 0 - Minimal Sandbox: Génération de poèmes avec un petit modèle local
"""

import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration du modèle
# TinyLlama-1.1B est un bon choix pour CPU (~1.1B paramètres)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model():
    """Charge le modèle et le tokenizer"""
    print(f"Chargement du modèle {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
        print("Utilisation du CPU (peut être lent)")
    else:
        print("Utilisation du GPU")
    
    return tokenizer, model

def generate_poem(topic, form, tokenizer, model):
    """Génère un poème basé sur le sujet et la forme"""
    # Construction du prompt
    prompt = f"Write a {form} about {topic}.\n\n"
    
    print(f"\nGénération du poème...")
    print(f"Sujet: {topic}")
    print(f"Forme: {form}")
    print(f"Prompt: {prompt}")
    
    # Tokenisation
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if not torch.cuda.is_available():
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    # Génération
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Décodage
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraction du poème (enlever le prompt)
    poem = generated_text[len(prompt):].strip()
    
    return poem

def main():
    parser = argparse.ArgumentParser(
        description="Générateur de poèmes minimal (Phase 0)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="grief and the ocean",
        help="Sujet du poème (ex: 'grief and the ocean')"
    )
    parser.add_argument(
        "--form",
        type=str,
        default="14-line sonnet",
        help="Forme du poème (ex: '14-line sonnet')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="poem.txt",
        help="Fichier de sortie (défaut: poem.txt)"
    )
    
    args = parser.parse_args()
    
    # Chargement du modèle
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        sys.exit(1)
    
    # Génération
    try:
        poem = generate_poem(args.topic, args.form, tokenizer, model)
    except Exception as e:
        print(f"Erreur lors de la génération: {e}")
        sys.exit(1)
    
    # Sauvegarde
    output_path = Path(args.output)
    output_path.write_text(poem, encoding="utf-8")
    
    print(f"\n{'='*60}")
    print("POÈME GÉNÉRÉ:")
    print(f"{'='*60}")
    print(poem)
    print(f"{'='*60}")
    print(f"\nPoème sauvegardé dans: {output_path.absolute()}")

if __name__ == "__main__":
    main()

