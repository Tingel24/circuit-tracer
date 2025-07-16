# visualize_results.py

from pathlib import Path
import torch
from circuit_tracer.utils import create_graph_files

# Load attribution output
attr_path = Path("attribution_output/attribution.pt")
if not attr_path.exists():
    raise FileNotFoundError(f"Attribution file not found at {attr_path}")

attr = torch.load(attr_path)

# Output graph directory
graph_dir = Path("attribution_output/graph")
graph_dir.mkdir(exist_ok=True, parents=True)

# Generate visualization files (graphviz format)
create_graph_files(attr, graph_dir)
print(f"Graph files created in {graph_dir}")

print("To view the graphs, open the DOT files using Graphviz or visualize them in a Jupyter notebook using networkx or pygraphviz.")
