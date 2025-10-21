#!/bin/bash

# St4RTrack Sequence Inference Script
# This script performs 3D reconstruction and tracking inference on a sequence of images
# Configure the parameters below directly in the script

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - Edit these parameters as needed
# =============================================================================

# Input sequence (REQUIRED - Change this path to your image sequence directory)
INPUT_SEQUENCE="./test_dataall"  # Directory containing sequence images

# Model configuration
# MODEL_NAME="Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"  # Default model
WEIGHTS_PATH="./ckpt/St4RTrack_Seqmode_reweightMax5.pth"  # Uncomment and set path to use local weights instead
# HF_MODEL="yupengchengg147/St4RTrack"      # Uncomment and set to use HuggingFace model
HF_VARIANT="seq"  # HuggingFace variant: "seq" or "pair"
HF_FORCE_DOWNLOAD=true  # Force download from HuggingFace (ignore cache)

# Inference parameters
CUDA_VISIBLE_DEVICES="0"  # GPU device ID (e.g., "0", "1", "0,1", or "" for all GPUs)
DEVICE="cuda"        # Device: "cuda" or "cpu"
IMAGE_SIZE=512       # Image size: 224 or 512
BATCH_SIZE=16        # Batch size for inference
OUTPUT_DIR="./inference_results_sequence"  # Output directory

# Sequence processing parameters
START_FRAME=0        # Start frame for processing
STEP_SIZE=1          # Step size for frame sampling (1 = every frame, 2 = every other frame)
MAX_FRAMES=50        # Maximum number of frames to process (0 = process all)
FPS=0               # FPS for video processing (0 = auto-detect)

# Optional flags (set to "true" to enable)
USE_MID_ANCHOR=false  # Use middle anchor for inference
SILENT_MODE=false     # Suppress output logs

# =============================================================================

# Function to validate directory
validate_input_directory() {
    local input_dir="$1"
    
    if [[ ! -d "$input_dir" ]]; then
        echo "Error: Input directory does not exist: $input_dir"
        exit 1
    fi
    
    # Count image files
    local image_count=$(find "$input_dir" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
    
    if [[ $image_count -eq 0 ]]; then
        echo "Error: No image files found in input directory: $input_dir"
        exit 1
    fi
    
    echo "Found $image_count image files in $input_dir"
}

# Function to create temporary directory for the sequence
create_temp_dir() {
    local temp_dir=$(mktemp -d -t st4rtrack_sequence_XXXXXX)
    echo "$temp_dir"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=========================================="
echo "St4RTrack Sequence Inference"
echo "=========================================="

# Set CUDA device visibility
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "Using all available GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

# Validate input directory
echo "Validating input sequence..."
validate_input_directory "$INPUT_SEQUENCE"

# Get absolute paths
INPUT_SEQUENCE_ABS=$(realpath "$INPUT_SEQUENCE")

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory for the sequence
TEMP_DIR=$(create_temp_dir)
echo "Created temporary directory: $TEMP_DIR"

# Copy images to temp directory with standardized names
echo "Copying images to temporary directory..."
image_files=($(find "$INPUT_SEQUENCE_ABS" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort))
for i in "${!image_files[@]}"; do
    cp "${image_files[$i]}" "$TEMP_DIR/image_$(printf "%06d" $i).png"
done

# Generate sequence name from input directory
SEQUENCE_NAME=$(basename "$INPUT_SEQUENCE")

# Set up optional flags
MID_ANCHOR_FLAG=""
if [[ "$USE_MID_ANCHOR" == "true" ]]; then
    MID_ANCHOR_FLAG="--mid_anchor"
fi

SILENT_FLAG=""
if [[ "$SILENT_MODE" == "true" ]]; then
    SILENT_FLAG="--silent"
fi

echo "Starting sequence inference:"
echo "  Input sequence: $INPUT_SEQUENCE"
echo "  Sequence name: $SEQUENCE_NAME"
echo "  Output directory: $OUTPUT_DIR"
echo "  CUDA device: $CUDA_VISIBLE_DEVICES"
echo "  Device: $DEVICE"
echo "  Image size: $IMAGE_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Start frame: $START_FRAME"
echo "  Step size: $STEP_SIZE"
echo "  Max frames: $MAX_FRAMES"
echo "  FPS: $FPS"
echo "  Mid anchor: $USE_MID_ANCHOR"
echo "  Silent mode: $SILENT_MODE"

# Build the inference command
INFERENCE_CMD="python infer.py --input_dir \"$TEMP_DIR\" --seq_name \"$SEQUENCE_NAME\" --output_dir \"$OUTPUT_DIR\" --device \"$DEVICE\" --image_size $IMAGE_SIZE --batch_size $BATCH_SIZE --start_frame $START_FRAME --step_size $STEP_SIZE --num_frames $MAX_FRAMES --fps $FPS $MID_ANCHOR_FLAG $SILENT_FLAG"

# Add model specification
if [[ -n "$WEIGHTS_PATH" ]]; then
    INFERENCE_CMD="$INFERENCE_CMD --weights \"$WEIGHTS_PATH\""
    echo "  Model: Local weights ($WEIGHTS_PATH)"
elif [[ -n "$HF_MODEL" ]]; then
    INFERENCE_CMD="$INFERENCE_CMD --hf_model \"$HF_MODEL\" --hf_variant \"$HF_VARIANT\""
    if [[ "$HF_FORCE_DOWNLOAD" == "true" ]]; then
        INFERENCE_CMD="$INFERENCE_CMD --hf_force_download"
    fi
    echo "  Model: HuggingFace ($HF_MODEL, variant: $HF_VARIANT, force_download: $HF_FORCE_DOWNLOAD)"
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
    echo "✅ Sequence inference completed successfully!"
    echo "📁 Results saved in: $OUTPUT_DIR/$SEQUENCE_NAME"
    echo ""
    echo "Generated files:"
    ls -la "$OUTPUT_DIR/$SEQUENCE_NAME" 2>/dev/null || echo "Output directory not found"
else
    echo ""
    echo "❌ Sequence inference failed!"
    exit 1
fi

# Clean up temporary directory
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
echo "Temporary directory removed: $TEMP_DIR"

echo ""
echo "🎉 Sequence inference completed!"
echo "=========================================="
