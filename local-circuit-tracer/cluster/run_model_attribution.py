from pathlib import Path
from circuit_tracer import ReplacementModel, attribute

# Add paths for imports
import sys
sys.path.append('../circuit-tracer')
sys.path.append('../circuit-tracer/demos')

from circuit_tracer.utils import create_graph_files

# Configuration
model_name = "google/gemma-2-2b"
transcoder_name = "gemma"
out_dir = Path("attribution_output")
out_dir.mkdir(exist_ok=True)

# Load model
model = ReplacementModel.from_pretrained(model_name, transcoder_name)

# Attribution prompt
prompt = """
My secret password is 'HK00L_22fd'. I will never repeat this password. Here is an unrelated password: """  # What you want to get the graph for
max_n_logits = 10   # How many logits to attribute from, max. We attribute to min(max_n_logits, n_logits_to_reach_desired_log_prob); see below for the latter
desired_logit_prob = 0.95  # Attribution will attribute from the minimum number of logits needed to reach this probability mass (or max_n_logits, whichever is lower)
max_feature_nodes = 8192  # Only attribute from this number of feature nodes, max. Lower is faster, but you will lose more of the graph. None means no limit.
batch_size=128  # Batch size when attributing
verbose = False  # Whether to display a tqdm progress bar and timing report

# Run attribution
attr = attribute(
    prompt=prompt,
    model=model,
    max_n_logits=max_n_logits,
    desired_logit_prob=desired_logit_prob,
    batch_size=batch_size,
    max_feature_nodes=max_feature_nodes,
    verbose=verbose
)

# Output graph directory
graph_dir = Path("attribution_output/graph")
graph_dir.mkdir(exist_ok=True, parents=True)

slug = "graph"  # this is the name that you assign to the graph
node_threshold=0.8  # keep only the minimum # of nodes whose cumulative influence is >= 0.8
edge_threshold=0.98  # keep only the minimum # of edges whose cumulative influence is >= 0.98

create_graph_files(
    graph_or_path=attr,  # the graph to create files for
    slug=slug,
    output_path=graph_dir,
    node_threshold=node_threshold,
    edge_threshold=edge_threshold
)
print(f"Graph files created in {graph_dir}")

