#!/bin/bash
# Script to ensure DINOv2 compatibility for new machine installations

set -e

echo "Setting up DINOv2 compatibility..."

# Check if the local DINOv2 has the required dinotxt.py file
LOCAL_DINOV2_DIR="src/FoundationStereo/foundation_stereo/dinov2"
DINOTXT_FILE="$LOCAL_DINOV2_DIR/hub/dinotxt.py"

if [ ! -f "$DINOTXT_FILE" ]; then
    echo "Missing dinotxt.py in local DINOv2, downloading from GitHub..."
    
    # Define paths
    TORCH_HUB_DIR="$HOME/.cache/torch/hub"
    DINOV2_CACHE_DIR="$TORCH_HUB_DIR/facebookresearch_dinov2_main"
    
    # Create torch hub directory if it doesn't exist
    mkdir -p "$TORCH_HUB_DIR"
    
    # Clone or update the DINOv2 repository to get the missing file
    if [ -d "$DINOV2_CACHE_DIR" ]; then
        echo "Removing existing DINOv2 cache to ensure clean installation..."
        rm -rf "$DINOV2_CACHE_DIR"
    fi
    
    echo "Cloning DINOv2 repository..."
    git clone https://github.com/facebookresearch/dinov2.git "$DINOV2_CACHE_DIR"
    cd "$DINOV2_CACHE_DIR"
    # Pin to a known working commit that has dinotxt.py
    git checkout 12592a9
    
    # Copy the missing file to the local DINOv2
    echo "Copying dinotxt.py to local DINOv2..."
    cp "$DINOV2_CACHE_DIR/dinov2/hub/dinotxt.py" "$LOCAL_DINOV2_DIR/hub/"
    
    # Clean up the temporary cache
    rm -rf "$DINOV2_CACHE_DIR"
    echo "Cleaned up temporary cache"
else
    echo "dinotxt.py already exists in local DINOv2"
fi

# Ensure the local torchhub directory structure exists
echo "Setting up local torchhub directory structure..."
mkdir -p torchhub
if [ ! -L "torchhub/facebookresearch_dinov2_main" ]; then
    ln -sf "../$LOCAL_DINOV2_DIR" "torchhub/facebookresearch_dinov2_main"
    echo "Created symbolic link for local DINOv2"
else
    echo "Symbolic link already exists"
fi

# Ensure hubconf.py exists in the local DINOv2
HUBCONF_FILE="$LOCAL_DINOV2_DIR/hubconf.py"
if [ ! -f "$HUBCONF_FILE" ]; then
    echo "Creating hubconf.py for local DINOv2..."
    cat > "$HUBCONF_FILE" << 'EOF'
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from dinov2.hub.backbones import dinov2_vitb14_reg, dinov2_vitg14_reg, dinov2_vitl14_reg, dinov2_vits14_reg
from dinov2.hub.classifiers import dinov2_vitb14_lc, dinov2_vitg14_lc, dinov2_vitl14_lc, dinov2_vits14_lc
from dinov2.hub.classifiers import dinov2_vitb14_reg_lc, dinov2_vitg14_reg_lc, dinov2_vitl14_reg_lc, dinov2_vits14_reg_lc
from dinov2.hub.depthers import dinov2_vitb14_ld, dinov2_vitg14_ld, dinov2_vitl14_ld, dinov2_vits14_ld
from dinov2.hub.depthers import dinov2_vitb14_dd, dinov2_vitg14_dd, dinov2_vitl14_dd, dinov2_vits14_dd
from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l

dependencies = ["torch"]
EOF
    echo "Created hubconf.py"
else
    echo "hubconf.py already exists"
fi

echo "🎉 DINOv2 is now fully configured and ready to use!"
echo "The setup uses local files and symbolic links, avoiding hardcoded cache paths."
echo "This configuration will work across different platforms and users."
