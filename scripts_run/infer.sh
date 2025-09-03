#!/bin/bash

# This script is used to infer the model on the custom sequence, replace the checkpoint path and image sequence path.

your_ckpt_path="/path/to/your/ckpt"
your_image_sequence_path="/path/to/your/image/sequence"

ckpt_names=(
    "Pairmode_reweight"
    "Seqmode_reweight"
)

checkpoints=(
    "${your_ckpt_path}/Pair_reweight5/checkpoint-best.pth"
    "${your_ckpt_path}/Seq_reweight5/checkpoint-best.pth"
)

to_test=(
    "${your_image_sequence_path}"
)
start_frames=(0)

output_dir="./infer_results"

for ((i=0; i<${#checkpoints[@]}; i++)); do
    checkpoint="${checkpoints[$i]}"
    ckpt_name="${ckpt_names[$i]}"

    for ((j=0; j<${#to_test[@]}; j++)); do
        test_path="${to_test[$j]}"
        start_frame="${start_frames[$j]}"
        # Get basename of test path
        test_name=$(basename "$test_path")
        
        python infer.py \
            --batch_size 128 \
            --input_dir "$test_path" \
            --weights "$checkpoint" \
            --output_dir "$output_dir/${test_name}_${ckpt_name}_${start_frame}" \
            --start_frame "$start_frame"
    done
done

