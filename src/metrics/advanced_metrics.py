"""Module pour les métriques avancées d'évaluation des poèmes."""

from typing import Dict, Any, Optional
import numpy as np


def compute_perplexity(poem: str) -> float:
    """
    Calcule une métrique de qualité linguistique basée sur FlauBERT.
    Utilise une approximation via la cohérence des embeddings.
    
    Args:
        poem: Le texte du poème
    
    Returns:
        Métrique de qualité linguistique (float) ou 0.0 si erreur
    """
    if not poem or not poem.strip():
        return 0.0
    
    try:
        from transformers import FlaubertModel, FlaubertTokenizer
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        model_name = "flaubert/flaubert_base_uncased"
        
        tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lowercase=True)
        model = FlaubertModel.from_pretrained(model_name)
        model.eval()
        
        sentences = [s.strip() for s in poem.replace('\n', ' ').split('.') if s.strip() and len(s.strip()) > 3]
        if not sentences:
            sentences = [poem]
        
        total_score = 0.0
        count = 0
        
        with torch.no_grad():
            for sentence in sentences[:10]:
                try:
                    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
                    
                    if inputs["input_ids"].size(1) < 2:
                        continue
                    
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state
                    
                    if embeddings.size(1) > 1:
                        consecutive_similarities = []
                        for i in range(embeddings.size(1) - 1):
                            sim = F.cosine_similarity(
                                embeddings[0, i:i+1],
                                embeddings[0, i+1:i+2],
                                dim=1
                            ).item()
                            consecutive_similarities.append(sim)
                        
                        avg_similarity = np.mean(consecutive_similarities) if consecutive_similarities else 0
                        
                        embedding_variance = torch.var(embeddings.view(-1)).item()
                        
                        coherence = avg_similarity / (embedding_variance + 1e-6)
                        
                        perplexity_approx = 50 / (coherence + 0.1)
                        
                        total_score += perplexity_approx
                        count += 1
                        
                except Exception:
                    continue
        
        if count == 0:
            return 0.0
        
        return float(total_score / count)
    
    except Exception as e:
        print(f"Erreur lors du calcul de la métrique de qualité avec FlauBERT: {e}")
        return 0.0


def compute_lexical_diversity(poem: str) -> Dict[str, float]:
    """
    Calcule diverses métriques de diversité lexicale.
    
    Args:
        poem: Le texte du poème
    
    Returns:
        Dictionnaire avec :
        - type_token_ratio: vocab_size / word_count (TTR classique)
        - unique_word_ratio: proportion de mots uniques
        - lexical_richness: nombre de mots uniques par 100 mots
    """
    if not poem or not poem.strip():
        return {
            "type_token_ratio": 0.0,
            "unique_word_ratio": 0.0,
            "lexical_richness": 0.0,
        }
    
    words = poem.split()
    cleaned_words = [
        word.lower().strip(".,!?;:()[]{}«»\"'") 
        for word in words 
        if word.strip()
    ]
    
    cleaned_words = [w for w in cleaned_words if w]
    
    if not cleaned_words:
        return {
            "type_token_ratio": 0.0,
            "unique_word_ratio": 0.0,
            "lexical_richness": 0.0,
        }
    
    word_count = len(cleaned_words)
    unique_words = len(set(cleaned_words))
    
    type_token_ratio = unique_words / word_count if word_count > 0 else 0.0
    unique_word_ratio = unique_words / word_count if word_count > 0 else 0.0
    lexical_richness = (unique_words / word_count * 100) if word_count > 0 else 0.0
    
    return {
        "type_token_ratio": float(type_token_ratio),
        "unique_word_ratio": float(unique_word_ratio),
        "lexical_richness": float(lexical_richness),
    }


def compute_all_advanced_metrics(
    poem: str, 
    topic_graph: Optional[Dict[str, Any]] = None,
    embedding_model=None
) -> Dict[str, float]:
    """
    Calcule toutes les métriques avancées (embeddings + autres).
    
    Args:
        poem: Le texte du poème
        topic_graph: Dictionnaire avec 'nodes' et 'edges'
        embedding_model: Le modèle d'embeddings
    
    Returns:
        Dictionnaire avec toutes les métriques avancées
    """
    metrics = {}
    
    try:
        lexical_metrics = compute_lexical_diversity(poem)
        metrics.update(lexical_metrics)
    except Exception as e:
        print(f"Erreur lors du calcul de la diversité lexicale: {e}")
        metrics.update({
            "type_token_ratio": 0.0,
            "unique_word_ratio": 0.0,
            "lexical_richness": 0.0,
        })
    
    try:
        perplexity = compute_perplexity(poem)
        metrics["perplexity"] = perplexity
    except Exception as e:
        print(f"Erreur lors du calcul de la métrique de qualité linguistique: {e}")
        metrics["perplexity"] = 0.0
    
    if topic_graph is not None:
        try:
            from .embeddings import compute_graph_similarity, compute_stanza_coherence
            
            graph_metrics = compute_graph_similarity(poem, topic_graph, embedding_model)
            metrics.update(graph_metrics)
            
            coherence_metrics = compute_stanza_coherence(poem, embedding_model)
            metrics.update(coherence_metrics)
        except Exception as e:
            print(f"Erreur lors du calcul des métriques d'embeddings: {e}")
            metrics.update({
                "graph_mean_similarity": 0.0,
                "graph_max_similarity": 0.0,
                "graph_min_similarity": 0.0,
                "graph_coverage": 0.0,
                "mean_consecutive_similarity": 0.0,
                "overall_coherence": 0.0,
                "coherence_std": 0.0,
            })
    
    return metrics

