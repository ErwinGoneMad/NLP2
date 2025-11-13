# NLP2 - Générateur de poèmes

Générateur de poèmes utilisant des modèles de langage locaux.

## Installation avec uv

Ce projet utilise [uv](https://github.com/astral-sh/uv), un gestionnaire de paquets Python rapide et moderne.

### Prérequis

Installez `uv` si ce n'est pas déjà fait :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation des dépendances

```bash
# Créer un environnement virtuel et installer les dépendances
uv venv

# Activer l'environnement virtuel
source .venv/bin/activate  # Sur macOS/Linux
# ou
.venv\Scripts\activate  # Sur Windows

# Installer les dépendances
uv pip install -e .
```

Ou en une seule commande :

```bash
uv sync
```

## Utilisation

```bash
python poem_generator.py --topic "grief and the ocean" --form "14-line sonnet" --output poem.txt
```

## Dépendances

- torch>=2.0.0
- transformers>=4.35.0
- accelerate>=0.24.0

Note: `bitsandbytes` a été retiré car il n'est pas disponible sur macOS ARM64. Il n'est pas nécessaire pour le fonctionnement de base du générateur.

