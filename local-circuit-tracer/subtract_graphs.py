import json
from pathlib import Path
from collections import defaultdict
import argparse

NODE_NUMERIC_FIELDS = {"influence", "activation"}
LINK_NUMERIC_FIELDS = {"weight"}

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    metadata = data.get("metadata", {})
    return nodes, links, metadata

def subtract_nodes(node_lists):
    node_accumulator = defaultdict(list)
    all_layers = set()

    for nodes in node_lists:
        filtered_nodes = [node for node in nodes if node.get("feature_type") != "mlp reconstruction error"]
        for node in filtered_nodes:
            layer = node.get("layer")
            if layer != "E":
                all_layers.add(int(layer))
            node_accumulator[node["node_id"]].append(node)

    if not all_layers:
        raise ValueError("No numeric layers found in input data.")

    output_layer = max(all_layers)
    print(f"📤 Using output layer: {output_layer}")

    # Only keep nodes present in all files OR in the output or input layer
    common_node_ids = {
        node_id for node_id, instances in node_accumulator.items()
        if (
            len(instances) == len(node_lists) or
            all(str(n.get("layer")) == str(output_layer) or n.get("layer") == "E" for n in instances)
        )
    }
    # Include input and output nodes of first graph
    fixed_nodeids = [node["node_id"] for node in node_lists[0] if str(node.get("layer")) == str(output_layer) or node.get("layer") == "E"]
    common_node_ids.update(fixed_nodeids)
    averaged_nodes = [node_accumulator[node_id][0] for node_id in common_node_ids]

    return averaged_nodes, common_node_ids


def average_links(link_lists, valid_targets):
    link_accumulator = defaultdict(list)

    for links in link_lists:
        for link in links:
            if link["target"] in valid_targets:
                key = (link["source"], link["target"])
                link_accumulator[key].append(link)

    averaged_links = []
    for (src, tgt), instances in link_accumulator.items():
        base = dict(instances[0])  # Copy fields from first
        for field in LINK_NUMERIC_FIELDS:
            base[field] = sum(link.get(field, 0.0) for link in instances) / len(instances)
        averaged_links.append(base)

    return averaged_links

def main(files, output_path=None, graph_metadata_path=None):
    if len(files) < 2:
        print("Please provide at least two .json files to intersect.")
        return

    node_lists = []
    link_lists = []
    for file in files:
        nodes, links, _ = load_data(file)
        node_lists.append(nodes)
        link_lists.append(links)

    averaged_nodes, common_node_ids = subtract_nodes(node_lists)
    averaged_links = average_links(link_lists, common_node_ids)

    # Use metadata from the first file, but update slug
    _, _, metadata = load_data(files[0])
    metadata["slug"] = "subtraction_" + "_".join([Path(f).stem for f in files])

    # Try to copy qParams from the first file
    nodes0, _, metadata0 = load_data(files[0])
    with open(files[0]) as f:
        original_data = json.load(f)
    qParams = original_data.get("qParams", {
        "pinnedIds": [],
        "supernodes": [],
        "linkType": "both",
        "clickedId": "",
        "sg_pos": ""
    })
    result = {
        "metadata": metadata,
        "nodes": averaged_nodes,
        "links": averaged_links,
        "qParams": qParams
    }

    # Determine output file name from slug
    slug = metadata.get("slug", "intersected_graph")
    output_dir = Path(output_path).parent if output_path else Path(".")
    output_file = Path(output_dir) / f"{slug}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Output written to {output_file}")

    print(f"🔗 Intersected Nodes: {len(averaged_nodes)} | Links: {len(averaged_links)}")

    if graph_metadata_path:
        update_graph_metadata(graph_metadata_path, metadata)


def update_graph_metadata(graph_metadata_path, new_metadata):
    try:
        with open(graph_metadata_path, "r") as f:
            graph_meta = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: {graph_metadata_path} not found. Creating a new one.")
        graph_meta = {"graphs": []}

    # Try to copy a base graph entry to clone
    base_entry = next((g for g in graph_meta["graphs"] if g.get("slug") == new_metadata["slug"]), None)
    if base_entry is None and graph_meta["graphs"]:
        base_entry = graph_meta["graphs"][0]  # fallback
    elif base_entry is None:
        print("❌ Could not find a graph to clone for metadata. Aborting update.")
        return

    new_entry = dict(base_entry)
    new_entry["slug"] = new_metadata["slug"]

    if "prompt" in new_metadata:
        new_entry["prompt"] = "subtraction:" + new_metadata["prompt"]
    if "prompt_tokens" in new_metadata:
        new_entry["prompt_tokens"] = new_metadata["prompt_tokens"]

    if "scan" in new_metadata:
        new_entry["scan"] = new_metadata["scan"]
    if "node_threshold" in new_metadata:
        new_entry["node_threshold"] = new_metadata["node_threshold"]

    graph_meta["graphs"].append(new_entry)

    with open(graph_metadata_path, "w") as f:
        json.dump(graph_meta, f, indent=2)

    print(f"🧩 Appended new graph entry to: {graph_metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract from the first _clerps.json files the nodes it does not have in common with the rest.")
    parser.add_argument("files", nargs="+", help="List of .json files to intersect")
    parser.add_argument("--output", "-o",
                        help="Optional output file or directory. Defaults to slug.json in current dir.")
    parser.add_argument("--graph-metadata", "-g", help="Path to graph-metadata.json to update")
    args = parser.parse_args()

    main(args.files, args.output, args.graph_metadata)


