# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Training script for Robot Jacobian Field Model
# --------------------------------------------------------
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dust3r.model_jacobian import AsymmetricCroCo3DStereoJacobian
from dust3r.datasets.robot import RobotDataset
from dust3r.losses.robot_losses import MultiLoss
import os
import numpy as np
from tqdm import tqdm
import wandb
import time
import json
import datetime
from pathlib import Path

from dust3r.datasets import build_dataset
from dust3r.utils import utils
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from collections import defaultdict
import math
import sys
import torch.backends.cudnn as cudnn


class JacobianLoss(MultiLoss):
    """
    Loss function for Jacobian field model.
    
    The model predicts:
    1. Jacobian field: [B*S, H, W, 3, joint_dim] - maps joint actions to RGB changes
    2. 3D points: [B*S, H, W, 3] - 3D point cloud from image head
    
    The loss computes:
    1. RGB delta loss: ||predicted_rgb_delta - gt_rgb_delta||
    2. 3D consistency loss: standard 3D point loss
    """
    
    def __init__(self, rgb_weight=1.0, pts3d_weight=0.1, conf_weight=0.1):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.pts3d_weight = pts3d_weight
        self.conf_weight = conf_weight

    def compute_jacobian_loss(self, pred1, pred2, gt1, gt2):
        """Compute Jacobian field loss with RGB delta prediction"""
        # Get predictions
        jacobian_field = pred1['jacobian_field']  # [B*S, H, W, 3, joint_dim]
        
        # Get ground truth images
        gt_img1 = gt1['img']  # [B, S, 3, H, W] or [B, 3, H, W]
        gt_img2 = gt2['img']  # [B, S, 3, H, W] or [B, 3, H, W]
        
        # Get actions (joint position differences)
        joint_pos_view1 = gt1['joint_pos']  # [B, S, joint_dim] or [B, joint_dim]
        joint_pos_view2 = gt2['joint_pos']  # [B, S, joint_dim] or [B, joint_dim]
        actions = joint_pos_view2 - joint_pos_view1  # [B, S, joint_dim] or [B, joint_dim]
        
        # Handle batch dimension reshaping
        B = jacobian_field.shape[0]
        if gt_img1.ndim == 5:  # [B, S, C, H, W] format
            B_orig, S_orig = gt_img1.shape[:2]
            gt_img1 = gt_img1.view(B_orig * S_orig, *gt_img1.shape[2:])  # [B*S, C, H, W]
            gt_img2 = gt_img2.view(B_orig * S_orig, *gt_img2.shape[2:])  # [B*S, C, H, W]
            actions = actions.view(B_orig * S_orig, -1)  # [B*S, joint_dim]
        else:  # [B, C, H, W] format
            actions = actions.view(B, -1)  # [B, joint_dim]
        
        # Move tensors to same device
        gt_img1 = gt_img1.to(jacobian_field.device)
        gt_img2 = gt_img2.to(jacobian_field.device)
        actions = actions.to(jacobian_field.device)
        
        # Compute ground truth RGB delta
        gt_rgb_delta = gt_img2 - gt_img1  # [B*S, 3, H, W]
        gt_rgb_delta = gt_rgb_delta.permute(0, 2, 3, 1)  # [B*S, H, W, 3]
        
        # Apply Jacobian field to actions to get predicted RGB delta
        # jacobian_field: [B*S, H, W, 3, joint_dim]
        # actions: [B*S, joint_dim]
        # We need to compute: sum over joint_dim of jacobian_field * actions
        actions_expanded = actions.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B*S, 1, 1, 1, joint_dim]
        predicted_rgb_delta = torch.sum(jacobian_field * actions_expanded, dim=-1)  # [B*S, H, W, 3]
        
        # Compute RGB delta loss
        rgb_loss = F.mse_loss(predicted_rgb_delta, gt_rgb_delta)
        
        # Compute 3D consistency loss (if available)
        pts3d_loss = 0.0
        conf_loss = 0.0
        
        if 'pts3d_in_other_view' in pred2:
            pts3d_img = pred2['pts3d_in_other_view']  # [B*S, H, W, 3]
            # For now, we don't have ground truth 3D points, so we skip this loss
            # In a real scenario, you would compute 3D consistency loss here
            
        if 'conf' in pred1:
            # Confidence loss (optional)
            conf_action = pred1['conf']  # [B*S, H, W]
            # You can add confidence-based weighting here
            conf_loss = torch.mean(conf_action)  # Simple confidence regularization
        
        # Combine losses
        total_loss = self.rgb_weight * rgb_loss + self.pts3d_weight * pts3d_loss + self.conf_weight * conf_loss
        
        return {
            'total': total_loss,
            'rgb_delta': rgb_loss,
            'pts3d': pts3d_loss,
            'conf': conf_loss,
            'predicted_rgb_delta': predicted_rgb_delta,
            'gt_rgb_delta': gt_rgb_delta,
            'jacobian_field': jacobian_field,
            'original_img': gt_img1,  # For wandb logging
            'gt_img': gt_img2,  # For wandb logging
            'deformed_img': gt_img1 + predicted_rgb_delta.permute(0, 3, 1, 2)  # For wandb logging
        }

    def __call__(self, pred1, pred2, gt1, gt2):
        """Main loss computation"""
        return self.compute_jacobian_loss(pred1, pred2, gt1, gt2)


def create_model(args):
    """Create the Jacobian model"""
    model = AsymmetricCroCo3DStereoJacobian(
        output_mode='jacobian',
        head_type1='jacobian',
        head_type='linear',
        depth_mode=('exp', -float('inf'), float('inf')),
        conf_mode=('exp', 1, float('inf')),
        freeze=args.freeze,
        landscape_only=True,
        patch_embed_cls='PatchEmbedDust3R',
        arch_mode='VanillaDust3r',
        rope_mode='full_3d',
        action_dim=args.action_dim,
        img_size=args.img_size,
        patch_size=args.patch_size,
        enc_embed_dim=args.enc_embed_dim,
        enc_depth=args.enc_depth,
        enc_num_heads=args.enc_num_heads,
        dec_embed_dim=args.dec_embed_dim,
        dec_depth=args.dec_depth,
        dec_num_heads=args.dec_num_heads,
    )
    
    return model


def log_images_to_wandb(loss_dict, epoch, global_iteration, prefix='train'):
    """Log images to wandb for visualization"""
    # Extract images from loss_dict
    original_img = loss_dict.get('original_img')  # [B*S, 3, H, W]
    gt_img = loss_dict.get('gt_img')  # [B*S, 3, H, W] 
    deformed_img = loss_dict.get('deformed_img')  # [B*S, 3, H, W]
    
    if original_img is None or gt_img is None or deformed_img is None:
        return
    
    # Convert to numpy
    original_img_np = original_img.detach().cpu().numpy()
    gt_img_np = gt_img.detach().cpu().numpy()
    deformed_img_np = deformed_img.detach().cpu().numpy()
    
    # All images have shape: [B*S, 3, H, W]
    batch_size = original_img_np.shape[0]
    
    # Log up to 4 samples from the batch
    num_samples_to_log = min(4, batch_size)
    
    for i in range(num_samples_to_log):
        # Process original image
        original_img_viz = original_img_np[i]  # [3, H, W]
        original_img_viz = np.transpose(original_img_viz, (1, 2, 0))  # [H, W, 3]
        
        # Normalize to [0, 1] if needed
        if original_img_viz.min() < 0:  # If in [-1, 1] range
            original_img_viz = (original_img_viz + 1.0) / 2.0  # Convert to [0, 1]
        original_img_viz = np.clip(original_img_viz, 0, 1)
        
        # Process ground truth image
        gt_img_viz = gt_img_np[i]  # [3, H, W]
        gt_img_viz = np.transpose(gt_img_viz, (1, 2, 0))  # [H, W, 3]
        
        # Normalize to [0, 1] if needed
        if gt_img_viz.min() < 0:  # If in [-1, 1] range
            gt_img_viz = (gt_img_viz + 1.0) / 2.0  # Convert to [0, 1]
        gt_img_viz = np.clip(gt_img_viz, 0, 1)
        
        # Process deformed image
        deformed_img_viz = deformed_img_np[i]  # [3, H, W]
        deformed_img_viz = np.transpose(deformed_img_viz, (1, 2, 0))  # [H, W, 3]
        
        # Normalize to [0, 1] if needed
        if deformed_img_viz.min() < 0:  # If in [-1, 1] range
            deformed_img_viz = (deformed_img_viz + 1.0) / 2.0  # Convert to [0, 1]
        deformed_img_viz = np.clip(deformed_img_viz, 0, 1)
        
        # Create combined image: [Original | GT | Deformed]
        combined_img = np.concatenate([original_img_viz, gt_img_viz, deformed_img_viz], axis=1)
        
        # Log to wandb
        wandb.log({
            f'{prefix}_images/sample_{i}_combined': wandb.Image(combined_img, caption=f'Original | GT | Deformed (Epoch {epoch}, Iter {global_iteration})'),
            f'{prefix}_images/sample_{i}_original': wandb.Image(original_img_viz, caption=f'Original (Epoch {epoch}, Iter {global_iteration})'),
            f'{prefix}_images/sample_{i}_gt': wandb.Image(gt_img_viz, caption=f'Ground Truth (Epoch {epoch}, Iter {global_iteration})'),
            f'{prefix}_images/sample_{i}_deformed': wandb.Image(deformed_img_viz, caption=f'Deformed (Epoch {epoch}, Iter {global_iteration})'),
        })


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm: float = 0, args=None, lr_schedule_values=None):
    """Train for one epoch - compatible with DUSt3R infrastructure"""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq if args else 20

    if lr_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule_values[i]

    global_iteration = 0
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if lr_schedule_values is not None and data_iter_step % len(data_loader) == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule_values[i]

        view1, view2 = batch
        
        # Move data to device
        for key in view1:
            if isinstance(view1[key], torch.Tensor):
                view1[key] = view1[key].to(device, non_blocking=True)
        for key in view2:
            if isinstance(view2[key], torch.Tensor):
                view2[key] = view2[key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Forward pass
            pred1, pred2 = model(view1, view2)
            
            # Compute loss
            loss_dict = criterion(pred1, pred2, view1, view2)
            loss = loss_dict['total']

        loss_value = float(loss)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= args.accum_iter  # accumulation steps
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=True)
        
        # Extract individual losses for logging
        loss_details = {k: v for k, v in loss_dict.items() if k != 'total' and isinstance(v, (int, float, torch.Tensor))}
        if isinstance(loss_dict['rgb_delta'], torch.Tensor):
            loss_details['rgb_delta'] = float(loss_dict['rgb_delta'])
        if isinstance(loss_dict['pts3d'], torch.Tensor):
            loss_details['pts3d'] = float(loss_dict['pts3d'])
        if isinstance(loss_dict['conf'], torch.Tensor):
            loss_details['conf'] = float(loss_dict['conf'])

        # Update metric logger
        metric_logger.update(loss=loss_value, **loss_details)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=loss_scaler)

        # Log images to wandb every 200 iterations
        if (global_iteration % 200 == 0) and args.wandb and args.wandb and 'original_img' in loss_dict:
            log_images_to_wandb(loss_dict, epoch, global_iteration, 'train')
        
        global_iteration += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):
    """Test for one epoch - compatible with DUSt3R infrastructure"""
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
        view1, view2 = batch
        
        # Move data to device
        for key in view1:
            if isinstance(view1[key], torch.Tensor):
                view1[key] = view1[key].to(device, non_blocking=True)
        for key in view2:
            if isinstance(view2[key], torch.Tensor):
                view2[key] = view2[key].to(device, non_blocking=True)
        
        # Forward pass
        with torch.cuda.amp.autocast():
            pred1, pred2 = model(view1, view2)
            
            # Compute loss
            loss_dict = criterion(pred1, pred2, view1, view2)
            loss = loss_dict['total']

        # Extract individual losses for logging
        loss_details = {k: v for k, v in loss_dict.items() if k != 'total' and isinstance(v, (int, float, torch.Tensor))}
        if isinstance(loss_dict['rgb_delta'], torch.Tensor):
            loss_details['rgb_delta'] = float(loss_dict['rgb_delta'])
        if isinstance(loss_dict['pts3d'], torch.Tensor):
            loss_details['pts3d'] = float(loss_dict['pts3d'])
        if isinstance(loss_dict['conf'], torch.Tensor):
            loss_details['conf'] = float(loss_dict['conf'])

        metric_logger.update(loss=float(loss), **loss_details)
        
        # Log images to wandb for first batch of validation
        if idx == 0 and args.wandb and 'original_img' in loss_dict:
            log_images_to_wandb(loss_dict, epoch, idx, 'val')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R Robot Jacobian training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereoJacobian(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(256, 144), head_type1='jacobian', head_type='linear', output_mode='jacobian', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder', \
                        arch_mode='TempDust3r', action_dim=8)",
                        type=str, help="string containing the robot jacobian model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="JacobianLoss(rgb_weight=1.0, pts3d_weight=0.1, conf_weight=0.1)",
                        type=str, help="train criterion for robot jacobian")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', default="RobotDUSt3R(dataset_location='./robots/allegro/', dset='train', resolution=[(256, 144)], S=2, strides=[1,2,3,4,5], clip_step=32, curriculum_learning=True, training_mode='pair')", type=str, help="training set")
    parser.add_argument('--test_dataset', default="RobotDUSt3R(dataset_location='./robots/allegro/', dset='test', resolution=[(256, 144)], S=16, strides=[1,2,3], clip_step=32)", type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=100, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_only', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--eval_freq', type=int, default=5, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=10, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='frequence (number of iterations) to print infos while training')
    parser.add_argument('--wandb', action='store_true', default=True, help='use wandb for logging')
    parser.add_argument('--num_save_visual', default=1, type=int, help='number of visualizations to save')
    
    # robot-specific
    parser.add_argument('--rgb_weight', default=1.0, type=float, help='weight for RGB delta loss')
    parser.add_argument('--pts3d_weight', default=0.1, type=float, help='weight for 3D points loss')
    parser.add_argument('--conf_weight', default=0.1, type=float, help='weight for confidence loss')
    parser.add_argument('--use_identity_camera', action='store_true', default=True, help='use identity camera pose')
    
    # mode
    parser.add_argument('--mode', default='train', type=str, help='train / eval_pose / eval_depth')

    # output dir
    parser.add_argument('--output_dir', default='./output_jacobian/', type=str, help="path where to save the output")
    parser.add_argument('--no_model_save', action='store_true', default=False, help='do not save the model')

    return parser


def load_model(args, device):
    """Load Jacobian model"""
    model = eval(args.model)
    model = model.to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        print("Resume checkpoint %s" % args.pretrained)
        checkpoint_model = checkpoint['model']
        msg = model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    return model, model_without_ddp


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    
    # if main process, init wandb
    if args.wandb and misc.is_main_process():
        wandb.init(name=args.output_dir.split('/')[-1], 
                   project='robot-jacobian', 
                   entity='zhaoliangzhang',
                   config=args, 
                   sync_tensorboard=True,
                   dir=args.output_dir)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("output_dir: " + args.output_dir)

    # auto resume if not specified
    if args.resume is None:
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
        if os.path.isfile(last_ckpt_fname) and (not args.eval_only): 
            args.resume = last_ckpt_fname

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    model, model_without_ddp = load_model(args, device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # training dataset and loader
    print('Building train dataset {}'.format(args.train_dataset if args.train_dataset is not None else 'None'))
    if args.train_dataset is None:
        data_loader_train = None
    else:
        data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    # testing dataset and loader
    print('Building test dataset {}'.format(args.test_dataset if args.test_dataset is not None else 'None'))
    data_loader_test = {}
    if args.test_dataset:
        for dataset in args.test_dataset.split('+'):
            name = dataset.split('(')[0]
            data_loader_test[name] = build_dataset(dataset, args.batch_size, args.num_workers, test=True)

    # training criterion
    print('Building train criterion {}'.format(args.train_criterion))
    train_criterion = eval(args.train_criterion)
    
    # testing criterion
    test_criterion = None
    if args.test_criterion:
        print('Building test criterion {}'.format(args.test_criterion))
        test_criterion = eval(args.test_criterion)

    # build optimizer with layer-wise lr decay (lrd)
    lr_schedule_values = None
    lr_schedule_values = utils.update_lr_sched(args, data_loader_train, model_without_ddp, optimizer=None)
    optimizer = utils.build_optimizer(args, model_without_ddp, lr_schedule_values)

    loss_scaler = NativeScaler()  # for mixed precision training

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # resume from a checkpoint
    if args.resume:
        utils.resume_model(args, model_without_ddp, optimizer, loss_scaler)

    # build scheduler
    lr_scheduler, _ = utils.build_scheduler(args, optimizer)

    # training loop
    if data_loader_train:
        print(f"Actually training with lr={optimizer.param_groups[0]['lr']:.2e}")
        misc.train_one_epoch_jacobian = train_one_epoch  # monkey patch
        utils.train_one_epoch = train_one_epoch  # monkey patch
        
        for epoch in range(args.start_epoch, args.epochs):
            if hasattr(data_loader_train, 'sampler'):
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, train_criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, args=args,
                lr_schedule_values=lr_schedule_values
            )

            if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)

            # evaluation
            if data_loader_test and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
                misc.test_one_epoch_jacobian = test_one_epoch  # monkey patch
                utils.test_one_epoch = test_one_epoch  # monkey patch
                
                test_stats = {}
                for name, data_loader in data_loader_test.items():
                    test_stats[name] = test_one_epoch(data_loader, model, test_criterion, device, epoch, args, prefix=name)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

                if args.output_dir and misc.is_main_process():
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.wandb and misc.is_main_process():
        wandb.finish()


