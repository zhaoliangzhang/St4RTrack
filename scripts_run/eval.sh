#!/bin/bash

# This script is used to evaluate the model on the StaRTrack's benchmark, replace the checkpoint path, output directory and conda environment accordingly.

# This script is used to evaluate the model on the StaRTrack's benchmark, replace the checkpoint path, output directory and conda environment accordingly.

your_ckpt_path="/path/to/your/ckpt"

ckpt_names=(
    "Pairmode_reweight"
    "Seqmode_reweight"
)

checkpoints=(
    "${your_ckpt_path}/Pair_reweight5/checkpoint-best.pth"
    "${your_ckpt_path}/Seq_reweight5/checkpoint-best.pth"
)

output_dir="./eval_results"

for i in "${!checkpoints[@]}"; do
    #adt
    echo "Running ${checkpoints[$i]} on adt_mini"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29688 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis', traj_loss=True)" \
    --pretrained "${checkpoints[$i]}" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --pose_eval_freq 1 --fixed_eval_set \
    --track_eval_freq 1 --eval_only --num_frames 64 \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, depth_weight=1, traj_weight=0.05, cotracker=True)" --grad_clip \
    --output_dir "${output_dir}/${ckpt_names[$i]}_adt_mini" --data_type adt_mini &

    # #po
    echo "Running ${checkpoints[$i]} on po_mini"
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29689 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis', traj_loss=True)" \
    --pretrained "${checkpoints[$i]}" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --pose_eval_freq 1 --fixed_eval_set \
    --track_eval_freq 1 --eval_only --num_frames 64 \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, depth_weight=1, traj_weight=0.05, cotracker=True)" --grad_clip \
    --output_dir "${output_dir}/${ckpt_names[$i]}_po_mini" --data_type po_mini &

    #pstudio
    echo "Running ${checkpoints[$i]} on pstudio_mini"
    CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29691 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis', traj_loss=True)" \
    --pretrained "${checkpoints[$i]}" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --pose_eval_freq 1 --fixed_eval_set \
    --track_eval_freq 1 --eval_only --num_frames 64 \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, depth_weight=1, traj_weight=0.05, cotracker=True)" --grad_clip \
    --output_dir "${output_dir}/${ckpt_names[$i]}_pstudio_mini" --data_type pstudio_mini &

    #ds 
    echo "Running ${checkpoints[$i]} on ds_mini"
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29692 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis', traj_loss=True)" \
    --pretrained "${checkpoints[$i]}" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --pose_eval_freq 1 --fixed_eval_set \
    --track_eval_freq 1 --eval_only --num_frames 64 \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, depth_weight=1, traj_weight=0.05, cotracker=True)" --grad_clip \
    --output_dir "${output_dir}/${ckpt_names[$i]}_ds_mini" --data_type ds_mini &

    #tum
    echo "Running ${checkpoints[$i]} on tum"
    CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29690 train.py \
    --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --test_criterion "Regr3D(L21, norm_mode='avg_dis', traj_loss=True)" \
    --pretrained "${checkpoints[$i]}" \
    --lr 0.00005 --min_lr 1e-06 --warmup_epochs 1 --epochs 1 --batch_size 1 --accum_iter 2 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --pose_eval_freq 1 --fixed_eval_set \
    --track_eval_freq 1 --eval_only --num_frames 64 \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2, velo_weight=1, pose_weight=0, depth_weight=1, traj_weight=0.05, cotracker=True)" --grad_clip \
    --output_dir "${output_dir}/${ckpt_names[$i]}_tum" --data_type tum &

    # Wait for all background processes to complete before moving to next checkpoint
    wait

done
