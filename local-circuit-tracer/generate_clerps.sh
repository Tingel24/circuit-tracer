#!/bin/bash

INPUT_DIR="attribution_output/graph"
METADATA_FILE="$INPUT_DIR/graph-metadata.json"
PYTHON_SCRIPT="/Users/luca/circuit-tracer/circuit-tracer-clerp-downloader/populate_clerps.py"

INPUT_FILES=()

# Gather all .json files (excluding *_clerps.json and graph-metadata.json)
for file in "$INPUT_DIR"/*.json; do
  filename=$(basename "$file")

  if [[ "$filename" == *_clerps.json ]] || [[ "$filename" == "graph-metadata.json" ]]; then
    continue
  fi

  base="${filename%.json}"

  # Skip if already has clerps_generated in metadata
  if jq -e --arg name "$base" '.[$name].clerps_generated == true' "$METADATA_FILE" > /dev/null; then
    echo "✅ Skipping ${filename} (clerps already generated)"
    continue
  fi

  INPUT_FILES+=("$file")
done

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "🎉 All files already have clerps. Nothing to do."
  exit 0
fi

# Run Python script once with all inputs
python3 "$PYTHON_SCRIPT" \
  --inputs "${INPUT_FILES[@]}" \
  --explanations /Users/luca/circuit-tracer/circuit-tracer-clerp-downloader/data \
  --no-update

echo "✅ Done."
