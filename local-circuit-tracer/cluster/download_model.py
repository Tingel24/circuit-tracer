from transformers import AutoTokenizer, AutoModel
import os

# Specify the model name
model_name = "google/gemma-2-2b-it"

# Optional: Set a custom cache directory (if needed)
# os.environ["TRANSFORMERS_CACHE"] = "/path/to/shared/cache"

# Download and cache the model and tokenizer
print(f"Downloading and caching model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("Download complete. Model is cached locally.")
