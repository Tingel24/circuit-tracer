import os
import re
import gzip
import json
import argparse
import requests
from tqdm import tqdm
from urllib.parse import urljoin
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# 📡 Step 1: Download explanation files from Neuronpedia's public S3 bucket
# ---------------------------------------------------------------------------
def download_transcoder_explanations(destination_dir: str):
    """
    Downloads transcoder-16k explanation files for layers 0–25
    from Neuronpedia's S3 bucket using known folder/file patterns.
    """
    base_url = "https://neuronpedia-datasets.s3.amazonaws.com/v1/gemma-2-2b/"
    layer_range = range(26)  # Layers 0–25
# https://neuronpedia-datasets.s3.amazonaws.com/v1/gemma-2-2b/0-gemmascope-transcoder-16k/explanations/batch-0.jsonl.gz
    for layer in layer_range:
        folder = f"{layer}-gemmascope-transcoder-16k"
        explanation_url = urljoin(base_url, f"{folder}/explanations/")
        print(f"\n🔍 Scanning: {explanation_url}")

        for part in range(500):  # Try up to 100 parts per layer
            file_name = f"batch-{part}.jsonl.gz"
            file_url = urljoin(explanation_url, file_name)

            local_dir = os.path.join(destination_dir, "gemma-2-2b", folder, "explanations")
            os.makedirs(local_dir, exist_ok=True)
            out_path = os.path.join(local_dir, file_name)

            if os.path.exists(out_path):
                continue  # Already downloaded

            # Check if file exists before downloading
            head = requests.head(file_url)
            if head.status_code == 404:
                if part == 0:
                    print(f"⚠️  No files in {folder}/explanations/")
                break  # No more files in this folder
            elif head.status_code != 200:
                print(f"❌ Error {head.status_code} for {file_url}")
                break

            print(f"⬇️  Downloading Batch {part}")
            try:
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            except Exception as e:
                print(f"❌ Download failed: {file_url} → {e}")
                break


# ---------------------------------------------------------------------------
# 📚 Step 2: Build in-memory explanation index from downloaded files
# ---------------------------------------------------------------------------
def build_index_from_jsonl(explanation_base_path: str) -> Dict[Tuple[int, int], str]:
    """
    Walks the explanation folder, builds an in-memory index of:
    (layer, feature_id) → explanation text
    """
    index = {}
    print(f"\n🧠 Building explanation index from: {explanation_base_path}")

    for root, _, files in os.walk(explanation_base_path):
        for file in files:
            if not file.endswith(".jsonl.gz"):
                continue

            path = os.path.join(root, file)
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            feature_index = int(record.get("index"))
                            layer_index = int(record.get("layer").split("-")[0])
                            description = record.get("description", "No explanation found.")
                            if feature_index:
                                index[(layer_index, feature_index)] = description
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"❌ Failed to read {path}: {e}")

    print(f"✅ Index built with {len(index):,} entries.")
    return index


# ---------------------------------------------------------------------------
# ✍️ Step 3: Populate missing 'clerp' values in graph nodes
# ---------------------------------------------------------------------------
def populate_clerps_from_index(nodes: list, index: Dict[Tuple[int,int], str]):
    """
    For each node of type 'transcoder' with no 'clerp', look up its explanation
    and populate it using the index.
    """
    count_filled = 0
    for node in nodes:
        if "transcoder" in node.get("feature_type", "").lower() and not node.get("clerp"):
            layer = int(node['layer'])
            feature_index = node["feature"] % 100000
            clerp = index.get((layer,feature_index), "Explanation not found in index.")
            node["clerp"] = clerp
            count_filled += 1
    print(f"✅ Populated clerps for {count_filled:,} nodes.")


# ---------------------------------------------------------------------------
# 🚀 CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Populate missing 'clerp' fields in multiple graph JSON files using Neuronpedia explanations.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to one or more input graph JSON files.")
    parser.add_argument("--explanations", required=True, help="Directory to store/fetch explanation files.")
    parser.add_argument('--update', default=True, action=argparse.BooleanOptionalAction, help="Download explanation files if missing.")
    args = parser.parse_args()

    if args.update:
        print("🌐 Checking and downloading explanation files if needed...")
        download_transcoder_explanations(args.explanations)

    index = build_index_from_jsonl(args.explanations)

    for input_path in args.inputs:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get("metadata", {}).get("clerps_generated") is True:
                print(f"✅ Skipping {input_path} (clerps already generated)")
                continue

            nodes = data.get("nodes", [])
            print(f"\n📦 Processing {len(nodes)} nodes in {input_path}")

            populate_clerps_from_index(nodes, index)

            # Add clerps_generated flag
            if "metadata" not in data or not isinstance(data["metadata"], dict):
                data["metadata"] = {}
            data["metadata"]["clerps_generated"] = True

            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"💾 Updated: {input_path}")

        except Exception as e:
            print(f"❌ Failed to process {input_path}: {e}")



if __name__ == "__main__":
    main()
