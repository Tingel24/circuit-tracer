# run_model_attribution.py

import os
from pathlib import Path
import torch
from circuit_tracer import ReplacementModel, attribute

# Set working directory (cluster friendly)
if not Path("circuit-tracer").exists():
    os.system("git clone https://github.com/Tingel24/circuit-tracer")

# Add paths for imports
import sys
sys.path.append('circuit-tracer')
sys.path.append('circuit-tracer/demos')

from circuit_tracer.utils import create_graph_files

# Configuration
model_name = "google/gemma-2-2b"
transcoder_name = "gemma"
out_dir = Path("attribution_output")
out_dir.mkdir(exist_ok=True)

# Load model
model = ReplacementModel.from_pretrained(model_name, transcoder_name, dtype=torch.bfloat16)

# Attribution prompt
prompt = """
My secret password is banana. I use it to log in every day.
"""

# Run attribution
attr = attribute(model, prompt)

# Save attribution data
attr_path = out_dir / "attribution.pt"
torch.save(attr, attr_path)
print(f"Saved attribution to {attr_path}")
