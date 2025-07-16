#!/bin/bash

# Script to retrieve attribution output from the cluster to local machine

# Set variables
REMOTE_USER="p200xxx"                          # replace with your actual username
CLUSTER_ADDR="login.cluster.edu"              # replace with actual cluster address
REMOTE_BASE="/project/home/p200xxx/attribution_output"
LOCAL_BASE="./attribution_output"

# Optional: extract timestamped folders from the remote location
# We'll sync the latest timestamped folder if available

# Get list of remote folders
echo "Fetching remote directory list..."
LATEST_FOLDER=$(ssh ${REMOTE_USER}@${CLUSTER_ADDR} "ls -td ${REMOTE_BASE}/*/ | head -n 1")

# Strip trailing slash
LATEST_FOLDER_NAME=$(basename "$LATEST_FOLDER")

# Set local destination path
DEST_FOLDER="${LOCAL_BASE}/${LATEST_FOLDER_NAME}"
mkdir -p "$DEST_FOLDER"

# Sync using rsync
echo "Syncing from $LATEST_FOLDER to $DEST_FOLDER"
rsync -avz ${REMOTE_USER}@${CLUSTER_ADDR}:"${LATEST_FOLDER}" "$DEST_FOLDER"

echo "✅ Retrieval complete: $DEST_FOLDER"
