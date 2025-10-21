#!/bin/bash

# St4RTrack Two-Image Inference Script
# This script performs 3D reconstruction and tracking inference on exactly 2 input images
# Configure the parameters below directly in the script

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Edit these parameters as needed
# =============================================================================

# Input images (REQUIRED - Change these paths to your images)
IMAGE1="./test_data/view_0_00000_00000.png"
IMAGE2="./test_data/view_5_00000_00009.png"

# Model configuration
# MODEL_NAME="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"  # Default model
WEIGHTS_PATH="./ckpt/St4RTrack_Pairmode_reweightMax5.pth"  # Uncomment and set path to use local weights instead
# HF_MODEL="yupengchengg147/St4RTrack"      # Uncomment and set to use HuggingFace model (e.g., "yupengchengg147/St4RTrack")
HF_VARIANT="pair"  # HuggingFace variant: "seq" or "pair"
HF_FORCE_DOWNLOAD=true  # Force download from HuggingFace (ignore cache)

# Inference parameters
CUDA_VISIBLE_DEVICES="0"  # GPU device ID (e.g., "0", "1", "0,1", or "" for all GPUs)
DEVICE="cuda"        # Device: "cuda" or "cpu"
IMAGE_SIZE=512       # Image size: 224 or 512
BATCH_SIZE=4       # Batch size for inference
OUTPUT_DIR="./inference_results_2images"  # Output directory

# Optional flags (set to "true" to enable)
USE_MID_ANCHOR=false  # Use middle anchor for inference
SILENT_MODE=false     # Suppress output logs

# =============================================================================

# Function to validate image file
validate_image() {
    local image_path="$1"
    local image_name="$2"
    
    if [[ ! -f "$image_path" ]]; then
        echo "Error: $image_name file does not exist: $image_path"
        exit 1
    fi
    
    # Check if file is a valid image (basic check)
    if ! file "$image_path" | grep -qi "image"; then
        echo "Warning: $image_name may not be a valid image file: $image_path"
        echo "Continuing anyway..."
    fi
}

# Function to create temporary directory for the two images
create_temp_dir() {
    local temp_dir=$(mktemp -d -t st4rtrack_2images_XXXXXX)
    echo "$temp_dir"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=========================================="
echo "St4RTrack Two-Image Inference"
echo "=========================================="

# Set CUDA device visibility
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "Using all available GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

# Validate input images
echo "Validating input images..."
validate_image "$IMAGE1" "First image"
validate_image "$IMAGE2" "Second image"

# Get absolute paths
IMAGE1_ABS=$(realpath "$IMAGE1")
IMAGE2_ABS=$(realpath "$IMAGE2")

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for the two images
TEMP_DIR=$(create_temp_dir)
echo "Created temporary directory: $TEMP_DIR"

# Copy images to temp directory with standardized names
cp "$IMAGE1_ABS" "$TEMP_DIR/image1.jpg"
cp "$IMAGE2_ABS" "$TEMP_DIR/image2.jpg"

# Generate sequence name from image filenames
IMAGE1_NAME=$(basename "$IMAGE1" | sed 's/\.[^.]*$//')
IMAGE2_NAME=$(basename "$IMAGE2" | sed 's/\.[^.]*$//')
SEQ_NAME="${IMAGE1_NAME}_${IMAGE2_NAME}"

# Set up optional flags
MID_ANCHOR_FLAG=""
if [[ "$USE_MID_ANCHOR" == "true" ]]; then
    MID_ANCHOR_FLAG="--mid_anchor"
fi

SILENT_FLAG=""
if [[ "$SILENT_MODE" == "true" ]]; then
    SILENT_FLAG="--silent"
fi

echo "Starting inference on 2 images:"
echo "  Image 1: $IMAGE1"
echo "  Image 2: $IMAGE2"
echo "  Sequence name: $SEQ_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  CUDA device: $CUDA_VISIBLE_DEVICES"
echo "  Device: $DEVICE"
echo "  Image size: $IMAGE_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Mid anchor: $USE_MID_ANCHOR"
echo "  Silent mode: $SILENT_MODE"

# Build the inference command
INFERENCE_CMD="python infer.py --input_dir \"$TEMP_DIR\" --seq_name \"$SEQ_NAME\" --output_dir \"$OUTPUT_DIR\" --device \"$DEVICE\" --image_size $IMAGE_SIZE --batch_size $BATCH_SIZE $MID_ANCHOR_FLAG $SILENT_FLAG"

# Add model specification
if [[ -n "$HF_MODEL" ]]; then
    INFERENCE_CMD="$INFERENCE_CMD --hf_model \"$HF_MODEL\" --hf_variant \"$HF_VARIANT\""
    if [[ "$HF_FORCE_DOWNLOAD" == "true" ]]; then
        INFERENCE_CMD="$INFERENCE_CMD --hf_force_download"
    fi
    echo "  Model: HuggingFace ($HF_MODEL, variant: $HF_VARIANT, force_download: $HF_FORCE_DOWNLOAD)"
elif [[ -n "$WEIGHTS_PATH" ]]; then
    INFERENCE_CMD="$INFERENCE_CMD --weights \"$WEIGHTS_PATH\""
    echo "  Model: Local weights ($WEIGHTS_PATH)"
else
    INFERENCE_CMD="$INFERENCE_CMD --model_name \"$MODEL_NAME\""
    echo "  Model: $MODEL_NAME"
fi

# Run inference
echo ""
echo "Running inference command:"
echo "$INFERENCE_CMD"
echo ""

eval $INFERENCE_CMD

# Check if inference was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Inference completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR/$SEQ_NAME"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR/$SEQ_NAME" 2>/dev/null || echo "Output directory not found"
else
    echo ""
    echo "❌ Inference failed!"
    exit 1
fi

# Clean up temporary directory
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "Temporary directory removed: $TEMP_DIR"

echo ""
echo "🎉 Two-image inference completed!"
echo "=========================================="
