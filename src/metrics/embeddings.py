"""Module pour le calcul d'embeddings et de similarités sémantiques."""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


# Singleton pour le modèle d'embeddings
_embedding_model = None


def load_embedding_model():
    """
    Charge le modèle d'embeddings (singleton pattern).
    
    Returns:
        Le modèle sentence-transformers ou None si erreur.
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Modèle français optimisé et léger
        model_name = "dangvantuan/sentence-camembert-base"
        _embedding_model = SentenceTransformer(model_name)
        return _embedding_model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle d'embeddings: {e}")
        return None


def encode_text(text: str, embedding_model=None) -> Optional[np.ndarray]:
    """
    Encode un texte en vecteur d'embeddings.
    
    Args:
        text: Le texte à encoder
        embedding_model: Le modèle d'embeddings (optionnel, chargé automatiquement si None)
    
    Returns:
        Vecteur d'embeddings ou None si erreur
    """
    if not text or not text.strip():
        return None
    
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return None
    
    try:
        # Normalisation automatique par sentence-transformers
        embedding = embedding_model.encode(text, normalize_embeddings=True)
        return embedding
    except Exception as e:
        print(f"Erreur lors de l'encodage du texte: {e}")
        return None


def encode_batch(texts: List[str], embedding_model=None) -> Optional[np.ndarray]:
    """
    Encode plusieurs textes en batch (optimisation).
    
    Args:
        texts: Liste de textes à encoder
        embedding_model: Le modèle d'embeddings (optionnel)
    
    Returns:
        Matrice d'embeddings (n_texts, embedding_dim) ou None si erreur
    """
    if not texts:
        return None
    
    # Filtrer les textes vides
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return None
    
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return None
    
    try:
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings
    except Exception as e:
        print(f"Erreur lors de l'encodage batch: {e}")
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs d'embeddings.
    
    Args:
        emb1: Premier vecteur d'embeddings
        emb2: Deuxième vecteur d'embeddings
    
    Returns:
        Similarité cosinus (entre -1 et 1, généralement entre 0 et 1 pour embeddings normalisés)
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    try:
        # Utiliser sklearn pour le calcul efficace
        # Reshape pour avoir la bonne forme (1, n_features)
        emb1_reshaped = emb1.reshape(1, -1) if emb1.ndim == 1 else emb1
        emb2_reshaped = emb2.reshape(1, -1) if emb2.ndim == 1 else emb2
        
        similarity = sklearn_cosine_similarity(emb1_reshaped, emb2_reshaped)[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Erreur lors du calcul de similarité: {e}")
        return 0.0


def compute_graph_similarity(
    poem: str, 
    topic_graph: Dict[str, Any], 
    embedding_model=None
) -> Dict[str, float]:
    """
    Calcule la similarité entre un poème et le graphe thématique.
    
    Args:
        poem: Le texte du poème
        topic_graph: Dictionnaire avec 'nodes' (liste de thèmes) et 'edges' (liste de transitions)
        embedding_model: Le modèle d'embeddings (optionnel)
    
    Returns:
        Dictionnaire avec les métriques de similarité :
        - graph_mean_similarity: moyenne des similarités poème ↔ tous les nœuds
        - graph_max_similarity: similarité maximale (thème le plus présent)
        - graph_min_similarity: similarité minimale (thème le moins présent)
        - graph_coverage: proportion de nœuds avec similarité > 0.5
    """
    if not poem or not poem.strip():
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    nodes = topic_graph.get("nodes", [])
    if not nodes:
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    # Encoder le poème
    poem_embedding = encode_text(poem, embedding_model)
    if poem_embedding is None:
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    # Encoder tous les nœuds du graphe
    node_embeddings = encode_batch(nodes, embedding_model)
    if node_embeddings is None:
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    # Calculer les similarités entre le poème et chaque nœud
    similarities = []
    for node_emb in node_embeddings:
        sim = cosine_similarity(poem_embedding, node_emb)
        similarities.append(sim)
    
    if not similarities:
        return {
            "graph_mean_similarity": 0.0,
            "graph_max_similarity": 0.0,
            "graph_min_similarity": 0.0,
            "graph_coverage": 0.0,
        }
    
    similarities_array = np.array(similarities)
    
    return {
        "graph_mean_similarity": float(np.mean(similarities_array)),
        "graph_max_similarity": float(np.max(similarities_array)),
        "graph_min_similarity": float(np.min(similarities_array)),
        "graph_coverage": float(np.mean(similarities_array > 0.5)),
    }


def compute_stanza_coherence(
    poem: str, 
    embedding_model=None
) -> Dict[str, float]:
    """
    Calcule la cohérence sémantique entre les strophes d'un poème.
    
    Args:
        poem: Le texte du poème
        embedding_model: Le modèle d'embeddings (optionnel)
    
    Returns:
        Dictionnaire avec les métriques de cohérence :
        - mean_consecutive_similarity: moyenne des similarités entre strophes consécutives
        - overall_coherence: moyenne des similarités entre toutes les paires de strophes
        - coherence_std: écart-type des similarités (mesure de variation)
    """
    if not poem or not poem.strip():
        return {
            "mean_consecutive_similarity": 0.0,
            "overall_coherence": 0.0,
            "coherence_std": 0.0,
        }
    
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return {
            "mean_consecutive_similarity": 0.0,
            "overall_coherence": 0.0,
            "coherence_std": 0.0,
        }
    
    # Découper le poème en strophes
    lines = [line.strip() for line in poem.split("\n")]
    
    # Regrouper en strophes : par lignes vides ou par groupes de 4 lignes
    stanzas = []
    current_stanza = []
    
    for line in lines:
        if line == "":
            if current_stanza:
                stanzas.append("\n".join(current_stanza))
                current_stanza = []
        else:
            current_stanza.append(line)
    
    # Ajouter la dernière strophe si elle existe
    if current_stanza:
        stanzas.append("\n".join(current_stanza))
    
    # Si pas de strophes détectées (pas de lignes vides), essayer de regrouper par 4 lignes
    if len(stanzas) == 1 and len(lines) > 4:
        # Regrouper par groupes de 4 lignes
        stanzas = []
        for i in range(0, len(lines), 4):
            stanza_lines = lines[i:i+4]
            if stanza_lines:
                stanzas.append("\n".join(stanza_lines))
    
    # Si toujours une seule strophe, retourner des métriques par défaut
    if len(stanzas) < 2:
        return {
            "mean_consecutive_similarity": 1.0,  # Cohérence parfaite si une seule strophe
            "overall_coherence": 1.0,
            "coherence_std": 0.0,
        }
    
    # Encoder toutes les strophes
    stanza_embeddings = encode_batch(stanzas, embedding_model)
    if stanza_embeddings is None:
        return {
            "mean_consecutive_similarity": 0.0,
            "overall_coherence": 0.0,
            "coherence_std": 0.0,
        }
    
    # Calculer les similarités entre strophes consécutives
    consecutive_similarities = []
    for i in range(len(stanzas) - 1):
        sim = cosine_similarity(stanza_embeddings[i], stanza_embeddings[i + 1])
        consecutive_similarities.append(sim)
    
    # Calculer toutes les similarités entre paires de strophes
    all_similarities = []
    for i in range(len(stanzas)):
        for j in range(i + 1, len(stanzas)):
            sim = cosine_similarity(stanza_embeddings[i], stanza_embeddings[j])
            all_similarities.append(sim)
    
    if not consecutive_similarities:
        return {
            "mean_consecutive_similarity": 0.0,
            "overall_coherence": 0.0,
            "coherence_std": 0.0,
        }
    
    consecutive_array = np.array(consecutive_similarities)
    all_array = np.array(all_similarities) if all_similarities else consecutive_array
    
    return {
        "mean_consecutive_similarity": float(np.mean(consecutive_array)),
        "overall_coherence": float(np.mean(all_array)),
        "coherence_std": float(np.std(all_array)),
    }

