#!/bin/bash

# Script to retrieve attribution output from the cluster to local machine

# Set variables
REMOTE_USER="u103092"                          # replace with your actual username
CLUSTER_ADDR="login.lxp.lu"              # replace with actual cluster address
#REMOTE_BASE="/mnt/tier2/users/u103092/circuit-tracer/local-circuit-tracer/cluster/attribution_output"
REMOTE_BASE="/mnt/tier2/users/u103092/circuit-tracer/test"
#LOCAL_BASE="./attribution_output"
LOCAL_BASE="./test"

# Optional: extract timestamped folders from the remote location
# We'll sync the latest timestamped folder if available

# Get list of remote folders
echo "Fetching remote directory list..."
#LATEST_FOLDER=$(ssh -i ~/.ssh/id_ed25519_meluna -p 8822 ${REMOTE_USER}@${CLUSTER_ADDR} "ls -td ${REMOTE_BASE}/*/ | head -n 1")

# Strip trailing slash
#LATEST_FOLDER_NAME=$(basename "$LATEST_FOLDER")
#LATEST_FOLDER_NAME=$(basename "$LATEST_FOLDER")

# Set local destination path
DEST_FOLDER="${LOCAL_BASE}/*"
mkdir -p "$DEST_FOLDER"

# Sync using rsync
#echo "Syncing from $LATEST_FOLDER to $DEST_FOLDER"
rsync --max-size=10M -avz -e "ssh -p 8822 -i ~/.ssh/id_ed25519_meluna" ${REMOTE_USER}@${CLUSTER_ADDR}:"/mnt/tier2/users/u103092/circuit-tracer/test/" "./code_submission-2/"

