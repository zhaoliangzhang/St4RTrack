#!/bin/bash

# This script is used to train the model with pair mode and reweighting, replace the dataset path.

# Set data path
your_data_path="/path/to/your/data"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29666 train.py \
--model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt',\
output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24,\
enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, \
freeze='encoder')" \
--train_dataset "12800 @ PointOdysseyDUSt3R(dset='train', dataset_location='${your_data_path}/point_odyssey', \
aug_crop=16, resolution=[(512, 288)], \
S=2, strides=[1,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120], clip_step=32, z_far=80, \
curriculum_learning=True, training_mode='pair') + \
12800 @ DynamicReplicaDUSt3R(dataset_location='${your_data_path}/dynamic_replica_data/train', \
aug_crop=16, resolution=[(512, 288)], \
S=2, strides=[1,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120], clip_step=32, \
curriculum_learning=True, training_mode='pair') + \
12800 @ KubrickDUSt3R(resolution=[(512, 288)], dataset_location='${your_data_path}/kubric_data', \
S=2,strides=[1,3,5,7,9,11], clip_step=8, \
curriculum_learning=True, training_mode='pair')" \
--test_dataset "16 @ DynamicReplicaDUSt3R(dataset_location='${your_data_path}/dynamic_replica_data/train', S=16, clip_step=32, strides=[6,7,8], aug_crop=16, resolution=[(512, 288)], dist_type='linear_1_2', aug_focal=0.9) + \
160 @ DynamicReplicaDUSt3R(dataset_location='${your_data_path}/dynamic_replica_data/test', S=16, aug_crop=16, clip_step=32, strides=[6,7,8], resolution=[(512, 288)], dist_type='linear_1_2') + \
16 @ PointOdysseyDUSt3R(dset='train', dataset_location='${your_data_path}/point_odyssey', clip_step=32, S=16, strides=[1,2,3], aug_crop=16, resolution=[(512, 288)]) + \
80 @ KubrickDUSt3R(S=16, resolution=[(512, 288)], dataset_location='${your_data_path}/kubric_data') + \
160 @ PointOdysseyDUSt3R(dset='test', dataset_location='${your_data_path}/point_odyssey', clip_step=32, S=16, strides=[1,2,3], aug_crop=16, resolution=[(512, 288)])" \
--test_criterion "Regr3D(L21, norm_mode='avg_dis')" \
--pretrained "./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" \
--lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 100 --batch_size 22 --accum_iter 1 \
--save_freq 1 --keep_freq 5 --eval_freq 1 --fixed_eval_set --grad_clip --track_eval_freq 100 \
--train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis', velo_loss=False), alpha=0.2, velo_weight=0,\
 pose_weight=0, traj_weight=0.0, align3d_weight=0.0, depth_weight=0, cotracker=False, \
 reweight_mode='max', reweight_scale=5.0)" \
--output_dir "./train_results/Pair_reweight5"


