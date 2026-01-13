"""Module pour l'enrichissement des graphes sémantiques avec embeddings."""

from typing import Dict, Any, List, Optional
import numpy as np


def expand_graph_with_embeddings(
    topic_graph: Dict[str, Any], 
    embedding_model=None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Enrichit un graphe thématique en trouvant les mots sémantiquement proches pour chaque nœud.
    
    Cette fonction utilise les embeddings pour trouver des mots proches de chaque thème,
    créant ainsi un champ lexical enrichi. Cette expansion est utilisée uniquement pour
    améliorer les métriques de similarité, pas pour modifier la génération.
    
    Args:
        topic_graph: Dictionnaire avec 'nodes' (liste de thèmes) et 'edges' (liste de transitions)
        embedding_model: Le modèle d'embeddings
        top_k: Nombre de mots proches à trouver pour chaque nœud
    
    Returns:
        Dictionnaire enrichi avec 'expanded_nodes' contenant le champ lexical pour chaque nœud
    """
    if embedding_model is None:
        from src.metrics.embeddings import load_embedding_model
        embedding_model = load_embedding_model()
    
    if embedding_model is None:
        return topic_graph.copy()
    
    nodes = topic_graph.get("nodes", [])
    if not nodes:
        return topic_graph.copy()
    
    expanded_graph = topic_graph.copy()
    expanded_graph["expanded_nodes"] = {}
    
    try:
        from src.metrics.embeddings import encode_text, cosine_similarity
        
        common_french_words = [
            "amour", "douleur", "joie", "tristesse", "mélancolie", "bonheur",
            "souvenir", "mémoire", "oubli", "passé", "présent", "futur",
            "mer", "océan", "vague", "eau", "lac", "rivière", "plage",
            "ciel", "étoile", "lune", "soleil", "nuage", "vent",
            "nuit", "jour", "aube", "crépuscule", "matin", "soir",
            "fleur", "rose", "arbre", "forêt", "jardin", "nature",
            "mort", "vie", "naissance", "fin", "début", "existence",
            "silence", "bruit", "voix", "chant", "musique", "son",
            "lumière", "ombre", "ténèbres", "clarté", "obscurité",
            "cœur", "âme", "esprit", "corps", "pensée", "rêve",
            "espoir", "désespoir", "peur", "courage", "force", "faiblesse",
            "infini", "espace", "cosmos", "galaxie", "planète", "univers",
            "exploration", "découverte", "aventure", "voyage", "chemin",
            "solitude", "isolement", "rencontre", "union", "séparation",
            "émerveillement", "admiration", "surprise", "étonnement", "merveille",
            "mystère", "secret", "énigme", "inconnu", "caché",
            "temps", "durée", "moment", "instant", "éternité",
            "nostalgie", "regret", "souhait", "désir", "aspiration",
            "terre", "sol", "paysage", "horizon", "montagne", "vallée",
            "profondeur", "hauteur", "largeur", "distance", "proximité",
            "chaleur", "froid", "douceur", "dureté", "tendresse",
            "liberté", "contrainte", "libération", "captivité", "envol"
        ]
        
        vocab_embeddings = None
        try:
            from src.metrics.embeddings import encode_batch
            vocab_embeddings = encode_batch(common_french_words, embedding_model)
        except Exception:
            pass
        
        for node in nodes:
            node_embedding = encode_text(node, embedding_model)
            if node_embedding is None:
                expanded_graph["expanded_nodes"][node] = {
                    "lexical_field": [],
                    "similarities": []
                }
                continue
            
            if vocab_embeddings is not None:
                similarities = []
                node_lower = node.lower().strip()
                
                for i, word_emb in enumerate(vocab_embeddings):
                    word = common_french_words[i]
                    word_lower = word.lower().strip()
                    
                    if word_lower == node_lower or word_lower in node_lower or node_lower in word_lower:
                        continue
                    
                    sim = cosine_similarity(node_embedding, word_emb)
                    
                    if sim < 0.98:
                        similarities.append((word, sim))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_words = similarities[:top_k]
                
                if len(top_words) < top_k and len(similarities) > 0:
                    top_words = similarities[:min(top_k, len(similarities))]
                
                expanded_graph["expanded_nodes"][node] = {
                    "lexical_field": [word for word, _ in top_words],
                    "similarities": [sim for _, sim in top_words]
                }
            else:
                expanded_graph["expanded_nodes"][node] = {
                    "lexical_field": [],
                    "similarities": []
                }
    
    except Exception as e:
        print(f"Erreur lors de l'expansion du graphe: {e}")
        return topic_graph.copy()
    
    return expanded_graph

