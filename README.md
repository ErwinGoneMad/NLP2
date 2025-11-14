# Project 1 — "Compressed Creativity: Can a Small LLM Generate Poetic Coherence and Originality?"

## Résumé

Ce projet explore si un modèle de langage léger et open-source (CPU-bound) peut générer de la poésie à la fois sémantiquement cohérente et stylistiquement inventive, en utilisant des structures de feedback et des ressources minimales.

## Question de Recherche

**Peut-on obtenir de la créativité poétique avec un petit modèle LLM ?**

La question centrale de ce projet est de déterminer si un modèle LLM léger (par exemple, TinyLLaMA, Pythia-1B, ou Llama-3.2-1B) peut produire de la poésie qui soit à la fois :

- **Sémantiquement cohérente** : avec un sens et une continuité thématique
- **Stylistiquement inventive** : avec originalité et impact émotionnel

Tout cela en utilisant des techniques de structuration, de feedback itératif et des ressources minimales (CPU uniquement).

## Motivation

La créativité est souvent considérée comme émergeant d'un raisonnement contextuel massif — quelque chose que les grands modèles maîtrisent. Ce projet pose la question suivante : **la structure peut-elle remplacer l'échelle ?**

Peut-on guider un petit modèle vers l'artistique grâce à :

- Des contraintes formelles
- Des boucles de feedback
- Des représentations structurées (graphes sémantiques)

**Idée centrale** : Remplacer la créativité basée sur la taille par une créativité basée sur la structure.

## Méthodologie

### 1. Configuration du Modèle

- **Modèle utilisé** : Llama-3.2-1B-Instruct (quantifié en Q4_K_M)
- **Exécution** : Locale, CPU-bound
- **Format** : GGUF (via `llama-cpp-python`)

### 2. Échafaudage Poétique

Fournir des structures formelles et sémantiques :

- **Contraintes formelles** : sonnet, haïku, vers libres, limerick
- **Graphes thématiques** : nœuds représentant thèmes, émotions, images
- **Proximité sémantique** : arêtes encodant les relations (ex: deuil → mémoire → océan)

### 3. Génération Itérative

#### Étape 1 : Génération initiale

- Générer un premier jet sous contraintes formelles
- Utiliser des prompts structurés avec rôle système/utilisateur

#### Étape 2 : Critique et révision

- Re-prompt le modèle pour critiquer sa propre génération
- Évaluer la cohérence et la qualité des images
- Générer une version révisée basée sur le feedback

### 4. Évaluation

#### Quantitative

- **Cohérence basée sur embeddings** : similarité sémantique entre strophes
- **Diversité lexicale** : mesure de la richesse du vocabulaire
- **Perplexité** : mesure de la qualité linguistique

#### Qualitative

- **Évaluation humaine** : impact émotionnel et originalité
- **Comparaison** : contrastes interprétatifs avec des sorties de modèles plus grands (ex: GPT-4)
