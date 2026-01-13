"""Module pour la visualisation des graphes sémantiques."""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import networkx as nx


def visualize_semantic_graph(
    topic_graph: Dict[str, Any], 
    save_path: Optional[str] = None
):
    """
    Visualise un graphe thématique avec networkx et matplotlib.
    
    Args:
        topic_graph: Dictionnaire avec 'nodes' (liste de thèmes) et 'edges' (liste de transitions)
        save_path: Chemin pour sauvegarder la figure (optionnel)
    """
    nodes = topic_graph.get("nodes", [])
    edges = topic_graph.get("edges", [])
    expanded_nodes = topic_graph.get("expanded_nodes", {})
    
    if not nodes:
        print("Aucun nœud dans le graphe à visualiser")
        return
    
    G = nx.DiGraph()
    
    for node in nodes:
        G.add_node(node)
    
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
            if source in nodes and target in nodes:
                G.add_edge(source, target)
    
    plt.figure(figsize=(12, 8))
    
    if len(nodes) <= 5:
        pos = nx.spring_layout(G, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color='lightblue',
        node_size=2000,
        alpha=0.9
    )
    
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        arrowstyle='->',
        arrowsize=20,
        edge_color='gray',
        width=2,
        alpha=0.6
    )
    
    labels = {}
    for node in nodes:
        if node in expanded_nodes and expanded_nodes[node].get("lexical_field"):
            lexical_field = ", ".join(expanded_nodes[node]["lexical_field"][:3])
            labels[node] = f"{node}\n({lexical_field})"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=10,
        font_weight='bold'
    )
    
    plt.title("Graphe Thématique Sémantique", fontsize=16, fontweight='bold', pad=20)
    
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphe sauvegardé dans {save_path}")
    else:
        plt.show()
    
    plt.close()

