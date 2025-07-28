#!/bin/bash

INPUT_DIR="attribution_output/graph"
METADATA_FILE="$INPUT_DIR/graph-metadata.json"
PYTHON_SCRIPT="/Users/luca/circuit-tracer/circuit-tracer-clerp-downloader/populate_clerps.py"

# Create a temporary file to store updated metadata
TEMP_METADATA=$(mktemp)

# Parse original metadata into jq-ready format
jq '.' "$METADATA_FILE" > "$TEMP_METADATA" || { echo "Invalid JSON in metadata"; exit 1; }

# Loop through .json files that are not _clerps.json or graph-metadata.json
for file in "$INPUT_DIR"/*.json; do
  filename=$(basename "$file")

  if [[ "$filename" == *_clerps.json ]] || [[ "$filename" == "graph-metadata.json" ]]; then
    continue
  fi

  base="${filename%.json}"
  clerps_file="$INPUT_DIR/${base}_clerps.json"

  if [[ ! -f "$clerps_file" ]]; then
    echo "Generating: ${base}_clerps.json"

    python3 "$PYTHON_SCRIPT" \
      --input "$file" \
      --output "$clerps_file" \
      --explanations /Users/luca/circuit-tracer/circuit-tracer-clerp-downloader/data \
      --no-update

    if [[ $? -ne 0 ]]; then
      echo "Failed to process $file"
      continue
    fi

    # Update slug in place in metadata
    jq --arg slug "$base" --arg new_slug "${base}_clerps" '
      .graphs |= map(if .slug == $slug then .slug = $new_slug else . end)
    ' "$TEMP_METADATA" > "$TEMP_METADATA.tmp" && mv "$TEMP_METADATA.tmp" "$TEMP_METADATA"
  fi
done

# Save updated metadata
cp "$METADATA_FILE" "${METADATA_FILE}.bak"  # Backup
mv "$TEMP_METADATA" "$METADATA_FILE"

echo "Done. Metadata updated (in place). Backup created at ${METADATA_FILE}.bak"
