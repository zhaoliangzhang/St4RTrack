#!/bin/bash

# Robot Inference Script
# Simple script to run robot inference with hardcoded parameters
# 
# To test a specific data index, modify the python command below to add:
# --data_index 5  # (replace 5 with desired index)

# Set default checkpoint path (look for best checkpoint)
CHECKPOINT_PATH="./outputs/robot_256x144_20251018_214325/checkpoint-5.pth"

# Create output directory with timestamp
OUTPUT_DIR="./robot_inference_output"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Robot Inference Script"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: No checkpoint found!"
    echo "Looking for checkpoints in ./output_robot/"
    find ./output_robot -name "checkpoint-*.pth" 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Run robot inference
python infer_robot.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 1 \
    --num_workers 4 \
    --device cuda \
    --data_index 5

echo "=========================================="
echo "Robot inference completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
