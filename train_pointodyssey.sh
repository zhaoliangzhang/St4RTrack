#!/bin/bash

# Training script for PointOdyssey dataset with resolution [256,144]
# Based on scripts_run/train_pair_reweight.sh


# Create output directory
OUTPUT_DIR="./outputs/pointodyssey_256x144_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Training command (following the format from train_pair_reweight.sh)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29666 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(256, 144), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --train_dataset "1000 @ PointOdysseyDUSt3R(dset='train', dataset_location='./point_odyssey', aug_crop=16, resolution=[(256, 144)], S=2, strides=[1,2,3,4,5], clip_step=32, z_far=80, curriculum_learning=True, training_mode='seq')" \
    --test_dataset "100 @ PointOdysseyDUSt3R(dset='test', dataset_location='./point_odyssey', clip_step=32, S=2, strides=[1,2,3], aug_crop=16, resolution=[(256, 144)])" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
    --output_dir "$OUTPUT_DIR" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 100 --batch_size 8 --accum_iter 1 \
    --save_freq 5 --keep_freq 5 --eval_freq 5 --grad_clip \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis', velo_loss=False), alpha=0.2, velo_weight=0, pose_weight=0, traj_weight=0.0, align3d_weight=0.0, depth_weight=0, cotracker=False, reweight_mode='max', reweight_scale=5.0)" \
    --num_workers 8 \
    --seed 42

echo "Training completed. Output saved to: $OUTPUT_DIR"
