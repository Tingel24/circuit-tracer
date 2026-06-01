import json
from typing import Dict, Any, List, Set


def load_graph(path: str) -> Dict[str, Any]:
    """Load a graph JSON file into a dict."""
    with open(path, "r") as f:
        return json.load(f)


def save_graph(data: Dict[str, Any], path: str) -> None:
    """Save a graph JSON dict back to a file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_node_ids(data: Dict[str, Any]) -> Set[str]:
    """Extract the set of node_ids from a graph JSON dict."""
    return {node["node_id"] for node in data.get("nodes", [])}


def mark_overlaps(graphs: List[Dict[str, Any]], flag: str = "has_overlap") -> None:
    """
    Add a boolean flag to nodes in all graphs that overlap by node_id.
    Modifies the dicts in place.
    """
    # Collect counts of node_ids across all graphs
    from collections import Counter

    all_ids = Counter()
    for g in graphs:
        all_ids.update(get_node_ids(g))

    overlap_ids = {nid for nid, count in all_ids.items() if count > 1}

    # Mark nodes in each graph
    for g in graphs:
        for node in g.get("nodes", []):
            if node["node_id"] in overlap_ids:
                node[flag] = True


def main(files: List[str], outs: List[str] = None) -> None:
    """
    Load N graphs, mark overlaps, and save them back out.
    If outs is not provided, overwrite the input files.
    """
    graphs = [load_graph(f) for f in files]

    mark_overlaps(graphs, flag="has_overlap")

    if outs is None:
        outs = files
    elif len(outs) != len(files):
        raise ValueError("Number of output files must match number of input files")

    for g, out in zip(graphs, outs):
        save_graph(g, out)

    print("✅ Overlaps marked. Updated files:")
    for out in outs:
        print(f" - {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mark overlapping nodes in multiple graph JSON files.")
    parser.add_argument("files", nargs="+", help="Paths to graph JSON files")
    parser.add_argument("--outs", nargs="*", help="Optional output paths (defaults to overwrite input files)")

    args = parser.parse_args()
    main(args.files, args.outs)
