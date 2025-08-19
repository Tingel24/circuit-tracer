import json
from pathlib import Path
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


def mark_overlaps(data1: Dict[str, Any], data2: Dict[str, Any], flag: str = "overlap") -> None:
    """
    Add a boolean flag to nodes in both graphs that overlap by node_id.
    Modifies the dicts in place.
    """
    overlap_ids = get_node_ids(data1) & get_node_ids(data2)

    for node in data1.get("nodes", []):
        if node["node_id"] in overlap_ids:
            node[flag] = True

    for node in data2.get("nodes", []):
        if node["node_id"] in overlap_ids:
            node[flag] = True


def main(file1: str, file2: str, out1: str = None, out2: str = None) -> None:
    """
    Load two graphs, mark overlaps, and save them back out.
    If out1/out2 are not provided, overwrite the input files.
    """
    data1 = load_graph(file1)
    data2 = load_graph(file2)

    mark_overlaps(data1, data2, flag="has_overlap")

    out1 = out1 or file1
    out2 = out2 or file2

    save_graph(data1, out1)
    save_graph(data2, out2)

    print(f"✅ Overlaps marked. Updated files:\n - {out1}\n - {out2}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mark overlapping nodes in two graph JSON files.")
    parser.add_argument("file1", help="Path to first graph JSON file")
    parser.add_argument("file2", help="Path to second graph JSON file")
    parser.add_argument("--out1", help="Optional output path for first graph (defaults to overwrite)")
    parser.add_argument("--out2", help="Optional output path for second graph (defaults to overwrite)")

    args = parser.parse_args()
    main(args.file1, args.file2, args.out1, args.out2)
