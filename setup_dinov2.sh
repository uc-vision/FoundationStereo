#!/bin/bash
# Script to ensure DINOv2 compatibility for new machine installations

set -e

echo "Setting up DINOv2 compatibility..."

# Define paths
TORCH_HUB_DIR="$HOME/.cache/torch/hub"
DINOV2_CACHE_DIR="$TORCH_HUB_DIR/facebookresearch_dinov2_main"

# Create torch hub directory if it doesn't exist
mkdir -p "$TORCH_HUB_DIR"

# Clone or update the DINOv2 repository to a specific working commit
if [ -d "$DINOV2_CACHE_DIR" ]; then
    echo "Removing existing DINOv2 cache to ensure clean installation..."
    rm -rf "$DINOV2_CACHE_DIR"
fi

echo "Cloning DINOv2 repository..."
git clone https://github.com/facebookresearch/dinov2.git "$DINOV2_CACHE_DIR"
cd "$DINOV2_CACHE_DIR"
# Pin to a known working commit that has dinotxt.py
git checkout 12592a9

echo "DINOv2 setup complete!"
echo "The repository is now pinned to a working commit with dinotxt.py support"
