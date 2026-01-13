"""Module pour recalculer les métriques avec topic_graph dans les fichiers de résultats."""

from pathlib import Path
from typing import Optional
import json
from src.comparison.helpers import compute_metrics


def recalculate_metrics(
    results_dir: Path = Path("results"),
    file_pattern: str = "*.json",
    subdir: Optional[str] = None
) -> None:
    """
    Recalcule les métriques avec topic_graph pour tous les fichiers de résultats.
    
    Cette fonction parcourt les fichiers JSON dans le dossier de résultats et recalcule
    les métriques en utilisant le topic_graph stocké dans la config de chaque fichier.
    
    Args:
        results_dir: Dossier racine contenant les résultats (défaut: "results")
        file_pattern: Pattern pour filtrer les fichiers (défaut: "*.json")
        subdir: Sous-dossier spécifique à traiter (ex: "iterative", "comparison")
                 Si None, traite tous les sous-dossiers
    
    Exemples:
        # Recalculer toutes les métriques itératives
        recalculate_metrics(subdir="iterative")
        
        # Recalculer toutes les métriques de comparaison
        recalculate_metrics(subdir="comparison")
        
        # Recalculer tous les fichiers dans results/
        recalculate_metrics()
    """
    if subdir:
        search_dir = results_dir / subdir
        if not search_dir.exists():
            print(f"⚠ Dossier {search_dir} n'existe pas")
            return
        json_files = sorted(search_dir.glob(file_pattern), 
                           key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        # Chercher dans tous les sous-dossiers
        json_files = []
        for subdir_path in results_dir.iterdir():
            if subdir_path.is_dir() and not subdir_path.name.startswith('.'):
                json_files.extend(sorted(subdir_path.glob(file_pattern),
                                        key=lambda p: p.stat().st_mtime, reverse=True))
    
    if not json_files:
        print(f"Aucun fichier trouvé dans {results_dir}")
        if subdir:
            print(f"  (sous-dossier: {subdir})")
        return
    
    print(f"Recalcul des métriques avec topic_graph...")
    print(f"({len(json_files)} fichier(s) trouvé(s))\n")
    
    for json_file in json_files:
        print(f"\nTraitement de {json_file.name}...")
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠ Erreur lors de la lecture: {e}")
            continue
        
        # Extraire le topic_graph de la config
        topic_graph = data.get("config", {}).get("topic_graph")
        
        if not topic_graph:
            print(f"  ⚠ Pas de topic_graph trouvé dans la config")
            continue
        
        print(f"  ✓ Topic graph trouvé: {len(topic_graph.get('nodes', []))} nœuds")
        
        updated = False
        
        # Traiter les fichiers de comparaison
        if "approaches" in data:
            print(f"  → Type: Comparaison")
            for approach_name, approach_data in data.get("approaches", {}).items():
                if "poem" in approach_data:
                    poem = approach_data["poem"]
                    new_metrics = compute_metrics(poem, topic_graph)
                    approach_data["metrics"] = new_metrics
                    updated = True
                    print(f"    ✓ {approach_name} recalculé")
        
        # Traiter les fichiers itératifs
        elif "results" in data:
            print(f"  → Type: Itératif")
            for approach_name, approach_data in data.get("results", {}).items():
                # Recalculer original_metrics
                if "original" in approach_data:
                    original_poem = approach_data["original"]
                    new_metrics = compute_metrics(original_poem, topic_graph)
                    approach_data["original_metrics"] = new_metrics
                    updated = True
                    print(f"    ✓ {approach_name} - Original recalculé")
                
                # Recalculer les métriques pour chaque itération
                if "iterations" in approach_data:
                    for iteration in approach_data["iterations"]:
                        if "revised" in iteration and iteration["revised"]:
                            revised_poem = iteration["revised"]
                            new_metrics = compute_metrics(revised_poem, topic_graph)
                            iteration["metrics"] = new_metrics
                            updated = True
                            print(f"    ✓ {approach_name} - Itération {iteration.get('iteration', '?')} recalculée")
                
                # Recalculer final_metrics
                if "final" in approach_data:
                    final_poem = approach_data["final"]
                    new_metrics = compute_metrics(final_poem, topic_graph)
                    approach_data["final_metrics"] = new_metrics
                    updated = True
                    print(f"    ✓ {approach_name} - Final recalculé")
        
        # Sauvegarder le fichier mis à jour
        if updated:
            # Créer une sauvegarde
            backup_file = json_file.with_suffix('.json.backup')
            try:
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # Sauvegarder le fichier mis à jour
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                print(f"  ✓ Fichier mis à jour (backup: {backup_file.name})")
            except Exception as e:
                print(f"  ⚠ Erreur lors de la sauvegarde: {e}")
        else:
            print(f"  ⚠ Aucune mise à jour nécessaire")
    
    print(f"\n✓ Recalcul terminé !")

