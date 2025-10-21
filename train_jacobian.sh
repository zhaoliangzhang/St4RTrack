#!/bin/bash

# Training script for Robot Jacobian Field Model with resolution [256,144]
# Based on the robot training infrastructure

# Create output directory
OUTPUT_DIR="./outputs/jacobian_256x144_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Training command for Jacobian model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29667 train_jacobian.py \
    --model "AsymmetricCroCo3DStereoJacobian(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(256, 144), head_type1='jacobian', head_type='linear', output_mode='jacobian', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, action_dim=8, freeze='encoder')" \
    --train_dataset "149040 @ RobotDUSt3R(dataset_location='./robots/allegro/', dset='train', resolution=[(256, 144)], S=2, strides=[1,2,3,4,5], clip_step=32, curriculum_learning=True, training_mode='pair')" \
    --test_dataset "1000 @ RobotDUSt3R(dataset_location='./robots/allegro/', dset='test', resolution=[(256, 144)], S=16, strides=[1,2,3], clip_step=32)" \
    --test_criterion "JacobianLoss(rgb_weight=1.0, pts3d_weight=0.1, conf_weight=0.1)" \
    --output_dir "$OUTPUT_DIR" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 100 --batch_size 4 --accum_iter 1 \
    --save_freq 5 --keep_freq 5 --eval_freq 5 --grad_clip \
    --train_criterion "JacobianLoss(rgb_weight=1.0, pts3d_weight=0.1, conf_weight=0.1)" \
    --num_workers 8 \
    --seed 42 \
    --wandb --wandb_project "robot-jacobian" --wandb_entity "zhaoliangzhang"

echo "Robot Jacobian training completed. Output saved to: $OUTPUT_DIR"
