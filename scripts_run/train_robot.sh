#!/bin/bash

# This script trains the robot action-image model

# Set data path
your_data_path="/path/to/your/robot/data"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29066 train_robot.py \
--model "AsymmetricCroCo3DStereoRobot(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), \
head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
freeze='encoder', action_dim=7)" \
--train_dataset "800 @ RobotDUSt3R(dataset_location='${your_data_path}/train', \
S=2, strides=[1,2,4], aug_crop=16, resolution=[(512, 288)], clip_step=1)" \
--test_dataset "50 @ RobotDUSt3R(dataset_location='${your_data_path}/test', \
S=2, strides=[1,2,4], aug_crop=16, resolution=[(512, 288)], clip_step=1)" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
--lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 100 --batch_size 1 --accum_iter 1 \
--save_freq 1 --keep_freq 5 --eval_freq 1 --fixed_eval_set --grad_clip --num_workers 8 \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, \
pose_weight=0, traj_weight=0.5, align3d_weight=5.0, depth_weight=10.0)" \
--output_dir "./train_results/Robot_Action_Image"
