"""Module de métriques avancées pour l'évaluation des poèmes générés."""

from .embeddings import (
    load_embedding_model,
    encode_text,
    encode_batch,
    cosine_similarity,
    compute_graph_similarity,
    compute_stanza_coherence,
)

from .advanced_metrics import (
    compute_perplexity,
    compute_lexical_diversity,
    compute_all_advanced_metrics,
)

__all__ = [
    "load_embedding_model",
    "encode_text",
    "encode_batch",
    "cosine_similarity",
    "compute_graph_similarity",
    "compute_stanza_coherence",
    "compute_perplexity",
    "compute_lexical_diversity",
    "compute_all_advanced_metrics",
]

