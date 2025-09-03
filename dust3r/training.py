# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ["WANDB_MODE"] = "offline"
import argparse
import datetime
import json
import numpy as np
import sys
import time
import math
import wandb
from collections import defaultdict
from pathlib import Path
from typing import Sized
import torch.distributed as dist
import glob

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  
from dust3r.datasets import get_data_loader  
from dust3r.datasets import CustomDUSt3R
from dust3r.losses import *  
from dust3r.inference import loss_of_one_batch, visualize_results

from dust3r.track_eval import eval_ours_tapvid3d
import dust3r.utils.path_to_croco  
import croco.utils.misc as misc  
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  
from infer import get_reconstructed_scene


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(512, 288), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder', arch_mode='TempDust3r')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default=None, type=str, help="training set")
    parser.add_argument('--test_dataset', default=None, type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--warmup_epochs_cl', type=int, default=5, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    parser.add_argument("--eval_only", action='store_true', default=False)
    parser.add_argument("--first_eval", action='store_true', default=False)
    parser.add_argument("--fixed_eval_set", action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='clip gradients')

    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=True, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')
    
    # switch mode for train / eval pose / eval depth
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')
    parser.add_argument('--mixed', action='store_true', default=False, help='use mixed dataset')

    # for pose eval
    parser.add_argument('--pose_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--track_eval_freq', default=0, type=int, help='pose evaluation frequency')
    parser.add_argument('--num_frames', default=64, type=int, help='number of frames for track evaluation')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='batch size for track evaluation')
    parser.add_argument('--pose_eval_stride', default=1, type=int, help='stride for pose evaluation')
    parser.add_argument('--trackEval_data_root', default=None, type=str, help='root directory for track evaluation data')
    parser.add_argument('--save_best_pose', action='store_true', default=False, help='save best pose')
    parser.add_argument('--temporal_smoothing_weight', default=0.01, type=float, help='temporal smoothing weight for pose optimization')
    
    parser.add_argument('--flow_loss_weight', default=0.0, type=float, help='flow loss weight for pose optimization')
    parser.add_argument('--flow_loss_fn', default='smooth_l1', type=str, help='flow loss type for pose optimization')
    parser.add_argument('--use_flow_valid_mask', action='store_true', default=False, help='use flow valid mask for pose optimization')
    parser.add_argument('--use_self_mask', action='store_true', default=False, help='use self mask for pose optimization')
    parser.add_argument('--flow_loss_start_epoch', default=0.2, type=float, help='start epoch for flow loss')
    parser.add_argument('--flow_loss_thre', default=20, type=float, help='threshold for flow loss')
    parser.add_argument('--pxl_thresh', default=50.0, type=float, help='threshold for flow loss')
    parser.add_argument('--depth_regularize_weight', default=0.0, type=float, help='depth regularization weight for pose optimization')
    parser.add_argument('--translation_weight', default=1, type=float, help='translation weight for pose optimization')
    parser.add_argument('--silent', action='store_true', default=False, help='silent mode for pose evaluation')
    parser.add_argument('--full_seq', action='store_true', default=False, help='use full sequence for pose evaluation')
    parser.add_argument('--seq_list', nargs='+', default=None, help='list of sequences for pose evaluation')

    parser.add_argument('--pose_dataset', type=str, default='sintel', 
                    choices=['davis', 'kitti', 'kitti_new', 'shibuya', 'bonn', 'bonn_new', 'scannet', 'tum', 'tum_new', 'nyu'], 
                    help='choose dataset for pose evaluation')
    parser.add_argument('--reverse_seq', action='store_true', default=False, help='reverse the sequence for pose evaluation')

    # for monocular depth eval
    parser.add_argument('--no_crop', action='store_true', default=False, help='do not crop the image for monocular depth evaluation')
    parser.add_argument('--data_type', type=str, default=None, help='data type for monocular depth evaluation')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    parser.add_argument('--tta_eval', default=None, type=str, help="path to the tta eval seq per seq")
    parser.add_argument('--no_model_save', action='store_true', default=False, help='do not save the model')
    
    # HuggingFace model loading
    parser.add_argument('--hf_model', type=str, default=None, help='HuggingFace model repo (yupengchengg147/St4RTrack)')
    parser.add_argument('--hf_variant', type=str, default='seq', choices=['seq', 'pair'], help='HuggingFace model variant')
    parser.add_argument('--hf_force_download', action='store_true', default=False, help='Force download from HuggingFace')

    return parser

def load_model(args, device):
    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    return model, model_without_ddp

def clean_hf_cache():
    """Clean HuggingFace cache to force fresh download"""
    import shutil
    from huggingface_hub import hf_hub_download
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("Cleaned HuggingFace cache")
        except Exception as e:
            print(f"Warning: Could not clean cache: {e}")

def load_hf_model(hf_model_repo, model_variant, device, force_download=True):
    """Load the HuggingFace model with fresh download"""
    import tempfile
    import shutil
    print(f"Loading {model_variant} model from HuggingFace repo: {hf_model_repo}")
    
    if force_download:
        clean_hf_cache()
        print("Forcing fresh download from HuggingFace...")
    
    try:
        if model_variant == "seq":
            model = AsymmetricCroCo3DStereo.from_pretrained(
                hf_model_repo, 
                force_download=force_download
            )
        else:  # pair
            # Use HuggingFace Hub API to download from subfolder
            from huggingface_hub import hf_hub_download
            
            # Create a fresh temporary directory for the model files
            temp_dir = os.path.join(tempfile.gettempdir(), "st4rtrack_pair_fresh")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download the config and model files from the Pair subfolder
            config_path = hf_hub_download(
                repo_id=hf_model_repo,
                filename="Pair/config.json",
                cache_dir=temp_dir,
                force_download=force_download
            )
            model_path = hf_hub_download(
                repo_id=hf_model_repo, 
                filename="Pair/model.safetensors",
                cache_dir=temp_dir,
                force_download=force_download
            )
            # Load the model from the downloaded path
            model_dir = os.path.dirname(config_path)
            model = AsymmetricCroCo3DStereo.from_pretrained(model_dir)
            
    except Exception as e:
        print(f"Failed to load {model_variant} model: {e}")
        print("Falling back to seq model...")
        model = AsymmetricCroCo3DStereo.from_pretrained(
            hf_model_repo, 
            force_download=force_download
        )
    
    # Move model to device
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    return model

def load_model_with_hf_support(args, device):
    """Enhanced load_model function with HuggingFace support"""
    if args.hf_model is not None:
        # Load from HuggingFace
        print(f'Loading HuggingFace model: {args.hf_model} (variant: {args.hf_variant})')
        model = load_hf_model(args.hf_model, args.hf_variant, device, args.hf_force_download)
        model_without_ddp = model
        
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
            model_without_ddp = model.module
            
        return model, model_without_ddp
    else:
        # Use existing load_model function
        return load_model(args, device)

def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    # if main process, init wandb
    if args.wandb and misc.is_main_process():
        wandb.init(name=args.output_dir.split('/')[-1], 
                   project='st4rtrack', 
                   config=args, 
                   sync_tensorboard=True,
                   dir=args.output_dir)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("output_dir: " + args.output_dir)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        if os.path.isfile(last_ckpt_fname) and (not args.eval_only): args.resume = last_ckpt_fname

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    model, model_without_ddp = load_model_with_hf_support(args, device)

    # training dataset and loader
    print('Building train dataset {}'.format(args.train_dataset if args.train_dataset is not None else 'None'))
    #  dataset and loader
    if args.train_dataset is None:
        data_loader_train = None
    else:
        data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    # testing dataset and loader
    print('Building test dataset {}'.format(args.test_dataset if args.test_dataset is not None else 'None'))
    data_loader_test = {}
    if args.test_dataset:
        for dataset in args.test_dataset.split('+'):
            testset = build_dataset(dataset, 1, args.num_workers, test=True) # batch size = 1 for eval
            name_testset = dataset.split('(')[0]
            if getattr(testset.dataset.dataset, 'strides', None) is not None:
                name_testset += f'_stride{testset.dataset.dataset.strides}'
            data_loader_test[name_testset] = testset

    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.train_criterion).to(device)

    eff_batch_size = args.batch_size * args.accum_iter * world_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            gathered_test_stats = {}
            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            for test_name, testset in data_loader_test.items():

                if test_name not in test_stats:
                    continue

                if getattr(testset.dataset.dataset, 'strides', None) is not None:
                    original_test_name = test_name.split('_stride')[0]
                    if original_test_name not in gathered_test_stats.keys():
                        gathered_test_stats[original_test_name] = []
                    gathered_test_stats[original_test_name].append(test_stats[test_name])

                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            if len(gathered_test_stats) > 0:
                for original_test_name, stride_stats in gathered_test_stats.items():
                    if len(stride_stats) > 1:
                        stride_stats = {k: np.mean([x[k] for x in stride_stats]) for k in stride_stats[0]}
                        log_stats.update({original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()})
                        if args.wandb:
                            log_dict = {original_test_name + '_stride_mean_' + k: v for k, v in stride_stats.items()}
                            log_dict.update({'epoch': epoch})
                            wandb.log(log_dict)

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far, best_pose_ate_sofar=None):
        if not args.no_model_save:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far, best_pose_ate_sofar=best_pose_ate_sofar)

    best_so_far, best_pose_ate_sofar = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if best_pose_ate_sofar is None:
        best_pose_ate_sofar = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far, best_pose_ate_sofar)

        # Test on multiple datasets
        new_best = False
        new_pose_best = False

        if (epoch > args.start_epoch and args.eval_freq > 0 and epoch % args.eval_freq == 0) or args.eval_only or args.first_eval:
            test_stats = {}
            with torch.no_grad():
                if args.track_eval_freq>0 and (epoch % args.track_eval_freq==0 or args.eval_only) and misc.is_main_process():
                    # Perform trajectory evaluation in both recon and track modes
                    traj_eval_epoch(args, model, device, epoch, mode='both')

                # set a barrier
                for test_name, testset in data_loader_test.items():
                    stats = test_one_epoch(model, test_criterion, testset,
                                        device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                    test_stats[test_name] = stats

                    # Save best of all
                    if stats['loss_med'] < best_so_far:
                        best_so_far = stats['loss_med']
                        new_best = True

                # Add TTA evaluation if specified
                if args.tta_eval is not None and misc.is_main_process():
                    tta_eval_epoch(args, model_without_ddp, device, epoch)
        
        # set a barrier
        torch.distributed.barrier()
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)
        if args.eval_only and args.epochs <= 1:
            exit(0)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, best_pose_ate_sofar)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far, best_pose_ate_sofar)
            if new_pose_best and args.save_best_pose:
                save_model(epoch - 1, 'best_pose', best_so_far, best_pose_ate_sofar)
        
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    if test:
        batch_size = 1
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    
    assert torch.backends.cuda.matmul.allow_tf32 == True
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch, max_epoch=args.epochs, warmup_epoch=args.warmup_epochs_cl)
        
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        if 'traj_mask' in batch[0] and ( (batch[0]['traj_mask']).sum() == 0 or (batch[1]['valid_mask']).sum() == 0 ):
            print('Rank = {}, Empty batch, skipping gradient update'.format(dist.get_rank()))
            local_is_empty = True
        else:
            try:
                batch_result = loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=bool(args.amp))
                loss, loss_details = batch_result['loss']
                local_is_empty = False
            except Exception as e:
                print(f'Error in loss computation: {e}')
                local_is_empty = True

        # aggregate local_is_empty from all ranks
        local_is_empty_tensor = torch.tensor(local_is_empty, device=device)
        dist.all_reduce(local_is_empty_tensor, op=dist.ReduceOp.SUM)

        # if any rank has error, skip the whole batch
        if local_is_empty_tensor.item() > 0:
            print(f"Skipping batch due to empty data on one or more ranks")
            # delete the loss and batch_result to free up memory
            if not local_is_empty:
                del loss
                del batch_result
            continue

        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad = 0.8 if args.grad_clip else None,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch
        del batch_result

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch) if not args.fixed_eval_set else data_loader.dataset.set_epoch(0)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch) if not args.fixed_eval_set else data_loader.sampler.set_epoch(0)

    for idx, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        batch_result = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=False,
                                       use_amp=bool(args.amp))
        loss_tuple = batch_result['loss']
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

        if args.num_save_visual>0 and (idx % max((len(data_loader) // args.num_save_visual), 1) == 0) and misc.is_main_process() : # save visualizations
            save_dir = f'{args.output_dir}/{epoch}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            view1, view2, pred1, pred2 = batch_result['view1'], batch_result['view2'], batch_result['pred1'], batch_result['pred2']
            gt_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='gt')
            pred_visual = visualize_results(view1, view2, pred1, pred2, save_dir=save_dir, visualize_type='pred')
            
            if args.wandb:
                wandb.log({
                    'epoch': epoch,
                    'test_visual_gt': wandb.Object3D(open(gt_visual)),
                    'test_visual_pred': wandb.Object3D(open(pred_visual))
                })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results

@torch.no_grad()
def traj_eval_epoch(args, model, device, epoch, mode='both'):
    """
    Perform trajectory evaluation in different modes.
    
    Args:
        args: Command line arguments
        model: The model to evaluate
        device: Device to run evaluation on
        epoch: Current epoch number
        mode: Evaluation mode - 'recon_eval', 'track_eval', or 'both'
    """
    if mode in ['recon_eval', 'both']:
        # Reconstruction evaluation
        recon_eval_data_types = ['po_mini', 'tum']
        if args.data_type is not None:
            if args.data_type in recon_eval_data_types:
                recon_eval_data_types = [args.data_type]
            else:
                recon_eval_data_types = []
        
        for data_type in recon_eval_data_types:
            eval_kwargs = {}
            if args.trackEval_data_root is not None:
                eval_kwargs['data_root'] = args.trackEval_data_root
            eval_result = eval_ours_tapvid3d(args, model, device, data_type=data_type, eval_recon=True, visualize='test' in args.output_dir, visualize_all='testall' in args.output_dir, **eval_kwargs)
            
            # Check if evaluation returned empty result (no data files found)
            if not eval_result:
                print(f"Warning: No evaluation data found for {data_type} in recon_eval mode, skipping...")
                continue
                
            final_global, _, final_sim3, final_sim3_closed, final_sim3_closed_dyn, epe_global, epe_pertraj, epe_sim3, epe_sim3_closed = eval_result

            if misc.is_main_process():
                if args.wandb:
                    wandb.log({
                        'epoch': epoch,
                        f'final_global_{data_type}_recon': final_global,
                        f'final_sim3_{data_type}_recon': final_sim3,
                        f'final_sim3_closed_{data_type}_recon': final_sim3_closed,
                        f'epe_global_{data_type}_recon': epe_global,
                        f'epe_pertraj_{data_type}_recon': epe_pertraj,
                        f'epe_sim3_{data_type}_recon': epe_sim3,
                        f'epe_sim3_closed_{data_type}_recon': epe_sim3_closed,
                    })
                # save to log file
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps({'data_type': data_type, 'epoch': epoch, 'final_global': final_global, 'final_sim3': final_sim3, 'final_sim3_closed': final_sim3_closed, 'epe_global': epe_global, 'epe_pertraj': epe_pertraj, 'epe_sim3': epe_sim3, 'epe_sim3_closed': epe_sim3_closed}) + "\n")
    
    if mode in ['track_eval', 'both']:
        # Track evaluation
        track_eval_data_types = ['adt_mini', 'pstudio_mini', 'po_mini', 'ds_mini']
        if args.data_type is not None:
            if args.data_type in track_eval_data_types:
                track_eval_data_types = [args.data_type]
            else:
                track_eval_data_types = []
        
        for data_type in track_eval_data_types:
            eval_kwargs = {}
            if args.trackEval_data_root is not None:
                eval_kwargs['data_root'] = args.trackEval_data_root
            eval_result = eval_ours_tapvid3d(args, model, device, data_type=data_type, visualize='test' in args.output_dir, visualize_all='testall' in args.output_dir, **eval_kwargs)
            
            # Check if evaluation returned empty result (no data files found)
            if not eval_result:
                print(f"Warning: No evaluation data found for {data_type} in track_eval mode, skipping...")
                continue
                
            final_global, final_pertraj, final_sim3, final_sim3_closed, final_sim3_closed_dyn, epe_global, epe_pertraj, epe_sim3, epe_sim3_closed = eval_result
            
            if misc.is_main_process():
                if args.wandb:
                    wandb.log({
                        'epoch': epoch,
                        f'final_global_{data_type}': final_global,
                        f'final_pertraj_{data_type}': final_pertraj,
                        f'final_sim3_{data_type}': final_sim3,
                        f'final_sim3_closed_{data_type}': final_sim3_closed,
                        f'final_sim3_closed_dyn_{data_type}': final_sim3_closed_dyn,
                        f'epe_global_{data_type}': epe_global,
                        f'epe_pertraj_{data_type}': epe_pertraj,
                        f'epe_sim3_{data_type}': epe_sim3,
                        f'epe_sim3_closed_{data_type}': epe_sim3_closed,
                    })
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps({'data_type': data_type, 'epoch': epoch, 'final_global': final_global, 'final_pertraj': final_pertraj, 'final_sim3': final_sim3, 'final_sim3_closed': final_sim3_closed, 'final_sim3_closed_dyn': final_sim3_closed_dyn, 'epe_global': epe_global, 'epe_pertraj': epe_pertraj, 'epe_sim3': epe_sim3, 'epe_sim3_closed': epe_sim3_closed}) + "\n")

@torch.no_grad()
def tta_eval_epoch(args, model_without_ddp, device, epoch):
    """
    Perform Test-Time Adaptation (TTA) evaluation on test sequences.
    
    Args:
        args: Command line arguments
        model_without_ddp: The model without DDP wrapper
        device: Device to run evaluation on
        epoch: Current epoch number
    """
        
    print(f"Running TTA evaluation for epoch {epoch}")
    # Create epoch-specific output directory based on args.output_dir
    epoch_output_dir = os.path.join(args.output_dir, f"tta_eval_epoch_{epoch}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    
    # Get list of test sequences
    test_sequences = []
    
    if os.path.isdir(args.tta_eval):
        # Check if there are subdirectories in args.tta_eval
        subdirs = [d for d in glob.glob(os.path.join(args.tta_eval, "*")) if os.path.isdir(d)]
        
        if subdirs:
            # If there are subdirectories, process each as a sequence
            for seq_dir in sorted(subdirs):
                seq_name = os.path.basename(seq_dir)
                files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))) or \
                        sorted(glob.glob(os.path.join(seq_dir, "*.png")))
                if files:
                    test_sequences.append({
                        "name": seq_name,
                        "files": files
                    })
        else:
            # If no subdirectories, treat the directory itself as a sequence
            # Use the last two parts of the path as the sequence name
            path_parts = args.tta_eval.rstrip('/').split('/')
            seq_name = '_'.join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
            
            files = sorted(glob.glob(os.path.join(args.tta_eval, "*.jpg"))) or \
                    sorted(glob.glob(os.path.join(args.tta_eval, "*.png")))
            if files:
                test_sequences.append({
                    "name": seq_name,
                    "files": files
                })
    elif os.path.isfile(args.tta_eval) and args.tta_eval.endswith('.txt'):
        # If a text file with sequence paths is provided
        with open(args.tta_eval, 'r') as f:
            for line in f:
                seq_path = line.strip()
                if os.path.isdir(seq_path):
                    # Use the last two parts of the path as the sequence name
                    path_parts = seq_path.rstrip('/').split('/')
                    seq_name = '_'.join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
                    
                    files = sorted(glob.glob(os.path.join(seq_path, "*.jpg"))) or \
                            sorted(glob.glob(os.path.join(seq_path, "*.png")))
                    if files:
                        test_sequences.append({
                            "name": seq_name,
                            "files": files
                        })
    
    # Process each test sequence
    for seq in test_sequences:
        if len(seq["files"]) > 0:
            print(f"Processing sequence {seq['name']} with {len(seq['files'])} frames")
            get_reconstructed_scene(
                filelist=seq["files"],
                model=model_without_ddp,
                device=device,
                batch_size=args.eval_batch_size,
                image_size=512,  # Default size
                output_dir=epoch_output_dir,
                seq_name=seq["name"],
                silent=True,
                start_frame=0,
                step_size=1,
                fps=0,
                num_frames=len(seq["files"])
            )
    
    print(f"TTA evaluation completed for epoch {epoch}. Results saved to {epoch_output_dir}")
