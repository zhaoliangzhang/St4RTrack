# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils import cotracker_vis
from dust3r.utils.geometry import inv, geotrf, normalize_pointcloud, normalize_pointcloud_seq
from dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from dust3r.camera_solver import CameraLoss
import torch.nn.functional as F
from dust3r.arap_utils import distance_preservation_loss_3d
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from dust3r.normal_loss import get_normal_loss

def scale_tracks_based_on_factor(tracks, scale_factor, H, W):
    """
    Scale trajectories by a given scale factor.

    Args:
        tracks: (T, H, W, 2) Trajectories containing x,y coordinates for each frame
        scale_factor: Scale factor to apply
        H: Image height
        W: Image width

    Returns:
        scaled_tracks: (N, 2) Scaled trajectories
    """

    T, H, W, _ = tracks.shape

    # Calculate image center
    center = torch.tensor([W / 2, H / 2], device=tracks.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 2)

    # Scale trajectories by scale factor
    scaled_tracks = (tracks - center) * scale_factor + center  # (T, H, W, 2)

    return scaled_tracks

def scale_tracks_to_align(pred_tracks, gt_tracks, H, W):
    """
    Scale predicted trajectories to align with ground truth trajectories
    by computing distances from each point to image center and optimizing a scale factor.

    Args:
        pred_tracks: (N, 2) Predicted trajectories containing x,y coordinates for each frame
        gt_tracks: (N, 2) Ground truth trajectories containing x,y coordinates for each frame 
        H: Image height
        W: Image width

    Returns:
        scaled_pred_tracks: (T, N, 2) Scaled predicted trajectories
    """
    center = torch.tensor([W / 2, H / 2], device=pred_tracks.device)

    pred_distances = torch.norm(pred_tracks - center, dim=-1)  # (N)
    gt_distances = torch.norm(gt_tracks - center, dim=-1)  # (N)
    scale_factor = torch.mean(gt_distances / (pred_distances + 1e-6))
    scaled_pred_tracks = (pred_tracks - center) * scale_factor.unsqueeze(-1) + center  # (N, 2)

    return scaled_pred_tracks, scale_factor

def dynamic_static_reweight(conf1_ori, gt1, dataset_mask, reweight_mode, reweight_scale, quantile=0.75):
    """
    Apply dynamic-static reweighting to confidence values based on trajectory movement.
    
    Args:
        conf1_ori: (B_subset, H, W) Original confidence values for the subset
        gt1: Ground truth data dictionary containing 'traj_ptc', 'pts3d', 'valid_mask', 'traj_mask'
        dataset_mask: (B,) Boolean mask indicating which samples to process
        reweight_mode: 'mean' or 'max' - how to compute static region confidence
        reweight_scale: Scale factor to apply to static confidence
        quantile: Quantile threshold for dynamic region detection (0.75 for PD, 0.5 for KB)
    
    Returns:
        conf1_result: (B_subset, H, W) Reweighted confidence values
        log_conf1_result: (B_subset, H, W) Log of reweighted confidence values
    """
    # Get subset data
    dyn_bool_valid = gt1['valid_mask'][dataset_mask] & gt1['traj_mask'][dataset_mask]  # [B_subset, H, W]
    B_subset = dyn_bool_valid.shape[0]
    h1_distance = torch.sum((gt1['traj_ptc'][dataset_mask] - gt1['pts3d'][dataset_mask])**2, dim=-1)  # [B_subset, H, W]
    
    # Calculate average movement distance per frame
    distance_sums = torch.sum(h1_distance * dyn_bool_valid.float(), dim=(1,2))  # [B_subset]
    valid_counts = dyn_bool_valid.sum(dim=(1, 2))
    has_valid = valid_counts > 0
    valid_counts = valid_counts.clamp_min(1).float()  # [B_subset]
    frame_distance_means = distance_sums / valid_counts  # [B_subset]

    # Calculate threshold
    if quantile == 0.75:  # PD mode
        q_val = torch.nanquantile(h1_distance.masked_fill(~dyn_bool_valid, float('nan')).view(B_subset, -1), quantile, dim=1)  # [B_subset]
        q_val[~has_valid] = 0.24  # rarely happens
        dy_th = torch.max(frame_distance_means, q_val)  # [B_subset]
    else:  # KB mode (quantile = 0.5)
        dy_th = torch.nanquantile(h1_distance.masked_fill(~dyn_bool_valid, float('nan')).view(B_subset, -1), quantile, dim=1)  # [B_subset]
        dy_th[~has_valid] = 0.24  # rarely happens

    # Separate dynamic parts into dynSTATIC and dynDYNAMIC
    dy_distance = dy_th.view(-1, 1, 1)  # [B_subset, 1, 1]
    dynDYNAMIC_bool = (h1_distance >= dy_distance) & dyn_bool_valid  # [B_subset, H, W]
    dynSTATIC_bool = ~dynDYNAMIC_bool & dyn_bool_valid  # [B_subset, H, W]

    # Calculate reweight values
    if reweight_mode == 'mean':
        masked_conf_dynSTATIC = conf1_ori * dynSTATIC_bool.float()  # [B_subset, H, W]
        dynSTATIC_counts = dynSTATIC_bool.sum(dim=(1, 2)).float()  # [B_subset]
        dynSTATIC_sums = masked_conf_dynSTATIC.sum(dim=(1, 2))  # [B_subset]
        dynSTATIC_counts = dynSTATIC_counts.clamp_min(1)  # [B_subset]
        conf1_static_mean = dynSTATIC_sums / dynSTATIC_counts  # [B_subset]
        conf1_static_mean = conf1_static_mean.clamp_min(1.0).detach()  # [B_subset] detach here
        reweight = conf1_static_mean * reweight_scale  # [B_subset]
    
    elif reweight_mode == 'max':
        conf_for_max = conf1_ori.masked_fill(~dynSTATIC_bool, float('-inf'))  # [B_subset, H, W]
        conf1_static_max = conf_for_max.amax(dim=(1, 2)).clamp_min(1.0)  # [B_subset]
        
        # Special handling for KB mode when no static regions exist
        if quantile == 0.5:  # KB mode
            wo_static = dyn_bool_valid.sum(dim=(1, 2)) == 0  # [B_subset]
            conf1_static_max[wo_static] = 5.0
        
        reweight = conf1_static_max.detach() * reweight_scale  # [B_subset] detach here
    
    # Apply reweight
    weight_to_fill = reweight.view(-1, 1, 1).expand_as(conf1_ori)  # [B_subset, H, W] reweight already detached
    conf1_result = torch.where(dynDYNAMIC_bool, weight_to_fill, conf1_ori)
    
    # In dynamic reweight mode, log_conf1 doesn't need to be 0 because there are valid gradients (dynSTATIC part)
    log_conf1_result = torch.log(conf1_result)
    
    return conf1_result, log_conf1_result


def track_loss_fn(pred_tracks_2d, gt_tracks_2d, H, W, align_scale=True):
    """
    Calculate trajectory loss, including optimizing a scale factor by aligning predicted trajectories with ground truth trajectories.
    
    Args:
        pred_tracks_2d: (T, N, 2) Predicted 2D trajectories
        gt_tracks_2d: (T, N, 2) Ground truth 2D trajectories 
        H: Image height
        W: Image width
    
    Returns:
        loss: Optimized L2 loss
    """
    if align_scale:
        scaled_pred_tracks, scale_factor = scale_tracks_to_align(pred_tracks_2d, gt_tracks_2d, H, W)
    else:
        scaled_pred_tracks = pred_tracks_2d
        scale_factor = 1.0
    
    loss = torch.sqrt(((scaled_pred_tracks - gt_tracks_2d) ** 2).mean())

    return loss, scale_factor

def compute_scale_invariant_depth_loss(pred_depth, gt_depth, epsilon=1e-8):
    """
    Compute a scale-invariant depth loss without taking logs.
    We:
      1) Solve for alpha to best align pred_depth to gt_depth in an MSE sense.
      2) Scale pred_depth by alpha.
      3) Compute the MSE between scaled_pred_depth and gt_depth.
    """

    numerator = torch.sum(pred_depth * gt_depth)
    denominator = torch.sum(pred_depth * pred_depth) + epsilon
    alpha = numerator / denominator

    # Scale the predicted depth
    scaled_pred_depth = alpha * pred_depth

    depth_loss = F.mse_loss(scaled_pred_depth, gt_depth)
    return depth_loss

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        return losses_and_masks
    else:
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


class LLoss (BaseCriterion):
    """ L-norm loss
    """

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance

class SmoothL1Loss(LLoss):
    """ Smooth L1 loss between points """

    def distance(self, a, b):
        # Use reduction='none' to get element-wise loss, then sum over the last dimension
        return torch.nn.functional.smooth_l1_loss(a, b, reduction='none').sum(dim=-1)


class L1Loss(LLoss):
    """ L1 (Manhattan) distance between points """
    
    def distance(self, a, b):
        return torch.abs(a - b).sum(dim=-1)  # Sum across the last dimension (coordinates)


class LogL1Loss(LLoss):
    """ Log L1 distance between points: first apply log transform, then L1 distance
    Helps with scale variations in the input space """
    
    def distance(self, a, b):
        # First apply log transform to both inputs (with epsilon for numerical stability)
        log_a = torch.log(torch.abs(a) + 1e-6)
        log_b = torch.log(torch.abs(b) + 1e-6)
        
        # Then compute L1 distance between log-transformed points
        return torch.abs(log_a - log_b).sum(dim=-1)


class MixedL1LogL1Loss(LLoss):
    """ Mixed L1 and LogL1 loss: L1 for x,y coordinates and LogL1 for z coordinate """
    
    def distance(self, a, b):
        # Split the coordinates
        a_xy, a_z = a[..., :2], a[..., 2:3]
        b_xy, b_z = b[..., :2], b[..., 2:3]
        
        # L1 loss for x,y coordinates
        xy_loss = torch.abs(a_xy - b_xy).sum(dim=-1)
        
        # LogL1 loss for z coordinate
        z_loss = torch.abs(torch.log(torch.abs(a_z) + 1e-6) - torch.log(torch.abs(b_z) + 1e-6)).sum(dim=-1)
        
        # Combine the losses
        return xy_loss + z_loss


L21 = L21Loss()
L11 = L1Loss()
LogL1 = LogL1Loss()
SmoothL1 = SmoothL1Loss()
MixedL1LogL1 = MixedL1LogL1Loss()

class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode='none'):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1, point clouds at world coord_frame
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1), I is identical matrix
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False, traj_loss=False, velo_loss=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.traj_loss = traj_loss
        self.velo_loss = velo_loss
        if self.traj_loss:
            self.pose_loss = CameraLoss()

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1

        B, H, W, C = gt1['pts3d'].shape
        in_camera1 = inv(gt1['camera_pose']) #T_w_to_cam1
        traj_mask1 = gt1['traj_mask']
        
        # transform pts3d to camera1 coordinate
        gt_pts1 = geotrf(in_camera1, gt1['traj_ptc'])
        gt_pts2 = geotrf(in_camera1, gt2['pts3d']) 

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d_in_other_view'] 

        if self.norm_mode:
            # Use distance from pr_pts2 to camera1 origin to normalize both pr_pts1 and pr_pts2
            pr_pts1, pr_pts2 = normalize_pointcloud_seq(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:    # relative scale prediction
            # Use distance from gt_pts2 to camera1 origin to normalize both gt_pts1 and gt_pts2
            gt_pts1, gt_pts2 = normalize_pointcloud_seq(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)


        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):

        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
               
        traj_mask = gt1['traj_mask']
        l1 = self.criterion(pred_pts1[traj_mask], gt_pts1[traj_mask])

        if self.velo_loss:
            pred_velo = (pred_pts1[1:] - pred_pts1[:-1])[traj_mask[1:]]
            gt_velo = (gt_pts1[1:] - gt_pts1[:-1])[traj_mask[1:]]
            l1_velo = self.criterion(pred_velo, gt_velo)
        else:
            l1_velo = torch.tensor(0.0, device=gt_pts1.device)

        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])

        # Record various losses to details
        self_name = type(self).__name__
        details = {
            self_name + '_pts3d_1': float(l1.mean()),
            self_name + '_pts3d_1_velo': float(l1_velo.mean()),
            self_name + '_pts3d_2': float(l2.mean())
        }

        return Sum((l1, mask1), (l1_velo, mask1), (l2, mask2)), (details | monitoring)



class ConfLoss(MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1, velo_weight=1, pose_weight=0, traj_weight=0, align3d_weight=0,
                 depth_weight=0, arap_weight=0., pred_intrinsics=False,
                 cotracker=False, normal_weight=0, intr_inv_loss=False, 
                pair_mode=False, reweight_mode=None, reweight_scale=-1):
        
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')
        self.velo_weight = velo_weight
        self.pose_weight = pose_weight
        self.traj_weight = traj_weight
        self.align3d_weight = align3d_weight
        self.depth_weight = depth_weight
        self.arap_weight = arap_weight
        self.normal_weight = normal_weight
        self.intr_inv_loss = intr_inv_loss

        if self.pose_weight != 0 or traj_weight != 0 or depth_weight != 0:
            self.pose_loss = CameraLoss(pred_intrinsics=pred_intrinsics)
        self.cotracker = cotracker
        if self.cotracker:
            self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()
            self.cotracker.eval()
            self.grid_size = 32
        
        self.pair_mode = pair_mode
        self.reweight_mode = reweight_mode
        self.reweight_scale = reweight_scale

        if self.pair_mode:
            only_regression_loss = all([
                self.velo_weight == 0,
                self.pose_weight == 0, 
                self.traj_weight == 0,
                self.align3d_weight == 0,
                self.depth_weight == 0,
                self.normal_weight == 0,
                self.arap_weight == 0,
                not self.intr_inv_loss,
                not self.cotracker
            ])
            assert only_regression_loss, 'pair mode only supports regression loss'

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        """
        Compute loss with optional terms.
        """

        Bz, H, W = pred1['conf'].shape[:3]
        # Initialize to 0 by default
        conf_loss1 = 0
        conf_loss2 = 0
        conf_loss1_velo = 0
        

        conf_loss1_pose = 0
        normal_loss = 0

        supervised_training = gt1["supervised_label"].sum() + gt2["supervised_label"].sum() > 0

        # supervised training when gt 3d points are available
        if supervised_training:
            ((loss1, msk1), (loss1_velo, msk1_velo), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)

            if loss1.numel() == 0:
                print('NO VALID POINTS in img1', force=True)
            if loss2.numel() == 0:
                print('NO VALID POINTS in img2', force=True)

            traj_mask1 = gt1['traj_mask'] #[B,H,W]
            conf1_ori = pred1['conf']

            # Reweigh dynamic points loss weight according to static points confidence
            if self.reweight_mode is not None:
                assert self.reweight_mode in ['max', 'mean'], 'invalid reweight_mode'
                assert self.reweight_scale > 0, 'reweight_scale must be positive'

                if self.pair_mode:
                    # pair mode dataset for training
                    conf1_mixed = torch.ones_like(conf1_ori, device=conf1_ori.device, dtype=conf1_ori.dtype) # [B, H, W]
                    log_conf1_mixed = torch.zeros_like(conf1_ori, device=conf1_ori.device, dtype=conf1_ori.dtype) # [B, H, W]
                    
                    kb_mask = torch.tensor([d=='kubrick' for d in gt1['dataset']], device=conf1_ori.device, dtype=torch.bool)  # [B]
                    PD_batch_mask = torch.tensor([d in ['pointodyssey', 'dynamic_replica'] for d in gt1['dataset']], 
                                                    device=conf1_ori.device, dtype=torch.bool)  # [B]
                    
                    # pointodyssey, dynamic_replica: use max of {mean and 0.75 quantile} to split dynamic and static
                    if PD_batch_mask.any():
                        conf1_dyn_result, log_conf1_dyn_result = dynamic_static_reweight(
                            conf1_ori[PD_batch_mask], gt1, PD_batch_mask, 
                            self.reweight_mode, self.reweight_scale, quantile=0.75
                        )
                        
                        conf1_mixed[PD_batch_mask] = conf1_dyn_result
                        log_conf1_mixed[PD_batch_mask] = log_conf1_dyn_result



                    # for kubrick: use median to split dynamic and static
                    if kb_mask.any():
                        conf1_kb_result, log_conf1_kb_result = dynamic_static_reweight(
                            conf1_ori[kb_mask], gt1, kb_mask, 
                            self.reweight_mode, self.reweight_scale, quantile=0.5
                        )
                        
                        conf1_mixed[kb_mask] = conf1_kb_result
                        log_conf1_mixed[kb_mask] = log_conf1_kb_result

        
                    # Final result: only take traj_mask1 part
                    conf1 = conf1_mixed[traj_mask1]
                    log_conf1 = log_conf1_mixed[traj_mask1]

                else:
                    # seq mode dataset for training
                    conf1, log_conf1 = self.get_conf_log(pred1['conf'][traj_mask1])

                    if traj_mask1[0].sum() == 0:
                        conf1 = torch.zeros_like(pred1['conf'][traj_mask1], device=conf1_ori.device, dtype=conf1_ori.dtype)
                        log_conf1 = torch.zeros_like(pred1['conf'][traj_mask1], device=conf1_ori.device, dtype=conf1_ori.dtype)
                    else:
                        with torch.no_grad():
                            dynamic_mask_h1_distance = (gt1['traj_ptc'][-1] - gt1['traj_ptc'][0]).abs().sum(-1) #[H,W]
                            dynamic_mask_h1_distance /=  (gt1['traj_ptc'][0].norm(dim=-1)+1e-8)
                            mean_val = dynamic_mask_h1_distance[traj_mask1[0]].mean() #[sum(traj_mask1[0]),]

                            dyn_bool = dynamic_mask_h1_distance > mean_val
                            dyn_bool = dyn_bool.unsqueeze(0).expand(Bz, -1, -1) 
                            dynamic_mask_h1 = dyn_bool[traj_mask1]

                        if self.reweight_mode == 'max':
                            conf1_static_max = conf1[~dynamic_mask_h1].max().detach()
                            masked_val = torch.full_like(conf1, self.reweight_scale*conf1_static_max, device=conf1.device, requires_grad=False) 
                        elif self.reweight_mode == 'mean':
                            conf1_static_mean = conf1[~dynamic_mask_h1].mean().detach()
                            masked_val = torch.full_like(conf1, self.reweight_scale*conf1_static_mean, device=conf1.device, requires_grad=False) 
                        else:
                            raise ValueError(f"Invalid reweight_mode: {self.reweight_mode}")

                        # masked_val is constant without any gradient
                        conf1 = torch.where(dynamic_mask_h1, masked_val, pred1['conf'][traj_mask1])
                        log_conf1 = torch.log(conf1.clamp(min=1))

            else:
                # no reweighting
                conf1, log_conf1 = self.get_conf_log(pred1['conf'][traj_mask1])
            
            # --- conf2 original weight by confidence output ---
            conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])

            # Reshape loss to match conf shapes
            loss1 = loss1.view(conf1.shape)
            loss2 = loss2.view(conf2.shape)

            conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
            conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0

            # conf_loss2
            conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
            conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

            # If velocity weight is non-zero
            
            if self.velo_weight != 0 and not self.pair_mode: # pair mode has no velocity loss
                # Reuse conf1 for velocity subset and skip the first frame
                n_velo = int(gt1['traj_mask'][0].sum())
                conf1_velo = conf1[n_velo:].flatten()
                log_conf1_velo = log_conf1[n_velo:].flatten()

                loss1_velo = loss1_velo.view(conf1_velo.shape)

                conf_loss1_velo = loss1_velo * conf1_velo - self.alpha * log_conf1_velo
                conf_loss1_velo = conf_loss1_velo.mean() if conf_loss1_velo.numel() > 0 else 0

        

            # Pose loss 
            # by default no pose loss in tta mode.
            if self.pose_weight != 0:
                pose_loss = self.pose_loss(gt2, pred1, pred2)
                conf_loss1_pose = pose_loss['total_loss']

            # Normal loss 
            # by default no normal loss in tta mode.
            if self.normal_weight > 0:
                gt_pts1, _, pred_pts1, _, _, _, _ = self.pixel_loss.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
                rand_offset = torch.randint(0, 16, (1,), device=pred1['pts3d'].device)[0]
                normal_loss = get_normal_loss(pred_pts1[:, rand_offset::16, :], gt_pts1[:, rand_offset::16, :], device=pred1['pts3d'].device)

        details = {}

        # 1. 2D trajectory loss
        track_loss = 0
        pred_poses = None
        if self.traj_weight != 0:
            # pref 2d tracks from pred1_pts3d and pred2_pred_poses
            pred_tracks_2d, pred_poses = self.pose_loss.get_tracks_2d(gt1, gt2, pred1, pred2) # pred_poses is T_cam2_to_cam1_pred
            
            if 'pts3d' in gt1:
                # get 2d tracks from gt1_pts3d and gt2_pred_poses
                gt_tracks_2d = self.pose_loss.get_tracks_2d_gt(gt1, gt2)
                traj_mask = gt1['traj_mask']

                # set up another filter, for those gt_tracks_2d exceed 1.5*W or 1.5*H
                H, W = gt1['pts3d'].shape[1:3]
                traj_mask = traj_mask & (gt_tracks_2d[..., 0] < 1.5 * W) & (gt_tracks_2d[..., 1] < 1.5 * H) & \
                            (gt_tracks_2d[..., 0] >= -0.5 * W) & (gt_tracks_2d[..., 1] >= -0.5 * H)

                # RMSE for 2D trajectory
                valid_pred = pred_tracks_2d[traj_mask]
                valid_gt   = gt_tracks_2d[traj_mask]
                if valid_pred.numel() > 0:
                    track_loss, scale_factor = track_loss_fn(valid_pred, valid_gt.reshape(-1,2), H, W, align_scale=True)  #eq7 in paper

        # 2. 3d align loss
        align3d_loss = 0
        if self.cotracker:
            assert self.traj_weight != 0 or self.align3d_weight != 0, 'cotracker is only used for traj_weight or align3d_weight'

            video = gt2['img_org'].unsqueeze(0) * 255
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker(video, grid_size = self.grid_size + torch.randint(-8, 8, (1,)).item()) # 1, B, N, 2; 1, B, N
            pred_tracks = pred_tracks[0] # B, N, 2

            # get the valid points in the first frame
            rounded_coords = torch.round(pred_tracks[0]).long()  # (N, 2)
            x_coords = rounded_coords[:, 0]
            y_coords = rounded_coords[:, 1]
            valid = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
            x_coords = x_coords[valid]
            y_coords = y_coords[valid]
            cotracker_mask = torch.zeros((H, W), device=pred_tracks.device, dtype=torch.bool)
            cotracker_mask[y_coords, x_coords] = 1
            cotracker_mask = cotracker_mask.unsqueeze(0).repeat(Bz, 1, 1) # (B, H, W)
            
            if self.traj_weight != 0:
                assert pred_tracks_2d is not None
                valid_pred = pred_tracks_2d[cotracker_mask] # (B， N, 2)

                if valid_pred.numel() > 0:
                    # valid_pred: model predicted 2d tracks
                    # pred_tracks: cotracker predicted 2d tracks
                    cotracker_loss, scale_factor = track_loss_fn(valid_pred, pred_tracks.reshape(-1,2), H, W, align_scale=True) #eq7 in paper
                    track_loss = track_loss + cotracker_loss

            if self.align3d_weight != 0:
                rounded_tracks_all = torch.round(pred_tracks).long() # get the rounded tracks for all frames, to get how query points move overtime in camera2; B, N, 2
                T, N = rounded_tracks_all.shape[:2]

                # get the corresponding points in head2: valid_mask: B, N
                valid_mask = (rounded_tracks_all[:,:,0] >= 0) & (rounded_tracks_all[:,:,0] < W) & (rounded_tracks_all[:,:,1] >= 0) & (rounded_tracks_all[:,:,1] < H) & pred_visibility[0] # B, N
                
                # setup index
                clamped_rounded_tracks_all = torch.stack([rounded_tracks_all[...,0].clamp(0, W-1), rounded_tracks_all[...,1].clamp(0, H-1)], dim=-1)    # B, N, 2
                time_idx = torch.arange(T, device=clamped_rounded_tracks_all.device).view(T, 1).expand(T, N)
                h_idx = clamped_rounded_tracks_all[..., 1]
                w_idx = clamped_rounded_tracks_all[..., 0]

                pts3d_head2_overtime = pred2['pts3d_in_other_view'][time_idx, h_idx, w_idx]     # B, N, 3

                # get the corresponding points in head1
                pts3d_head1_overtime = pred1['pts3d'][:,cotracker_mask[0]]  # for those query points in the first frame, how they move overtime according to head1; B, N, 3
                # compute the distance between the corresponding points in head1 and head2
                distance_head1_head2 = (pts3d_head1_overtime[valid_mask] - pts3d_head2_overtime[valid_mask]).norm(dim=-1)
                align3d_loss += distance_head1_head2.mean()  # eq10 in paper

        # Depth loss
        depth_loss = 0
        if self.depth_weight != 0:
            depth_mask = gt2['valid_mask']
            pred_depth_full = self.pose_loss.get_depth_head2(gt2, pred1, pred2, pred_poses)
            gt_depth_full   = self.pose_loss.get_depth_head2_gt(gt2)
            for i in range(pred_depth_full.size(0)):
                mask_i = depth_mask[i]
                pred_depth = pred_depth_full[i][mask_i]
                gt_depth   = gt_depth_full[i][mask_i]
                if pred_depth.numel() > 0:
                    depth_loss += compute_scale_invariant_depth_loss(pred_depth, gt_depth) # eq9 in paper

        arap_loss = 0
        if self.arap_weight > 0:
            rand_offset = torch.randint(0, 4, (1,), device=pred1['pts3d'].device)[0]
            pred_pts3d = pred1['pts3d'][:, rand_offset::4, rand_offset::4, :]
            T, H, W, C = pred_pts3d.shape
            pred_pts3d = pred_pts3d.reshape(1, T, H, W, C).permute(0, 1, 4, 2, 3)  # Reshape to B,T,C,H,W with T=1
            arap_loss = distance_preservation_loss_3d(pred_pts3d, k_neighbors=8)
        
        # Combine all parts
        total_loss = (conf_loss1 + conf_loss2 + self.velo_weight * conf_loss1_velo) \
                     + self.pose_weight * conf_loss1_pose \
                     + self.traj_weight * track_loss \
                     + self.depth_weight * depth_loss \
                     + self.arap_weight * arap_loss \
                     + self.normal_weight * normal_loss \
                     + self.align3d_weight * align3d_loss

        return total_loss, dict(
            conf_loss_1=float(conf_loss1),
            conf_loss_2=float(conf_loss2),
            conf_loss_1_velo=float(conf_loss1_velo),
            loss_traj_2d=float(track_loss),
            loss_depth=float(depth_loss),
            loss_arap=float(arap_loss),
            loss_normal=float(normal_loss),
            loss_align3d=float(align3d_loss),
            **details
        )


class Regr3D_ShiftInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute unnormalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # compute median depth
        gt_z1, gt_z2 = gt_pts1[..., 2], gt_pts2[..., 2]
        pred_z1, pred_z2 = pred_pts1[..., 2], pred_pts2[..., 2]
        gt_shift_z = get_joint_pointcloud_depth(gt_z1, gt_z2, mask1, mask2)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_z1, pred_z2, mask1, mask2)[:, None, None]

        # subtract the median depth
        gt_z1 -= gt_shift_z
        gt_z2 -= gt_shift_z
        pred_z1 -= pred_shift_z
        pred_z2 -= pred_shift_z

        # monitoring = dict(monitoring, gt_shift_z=gt_shift_z.mean().detach(), pred_shift_z=pred_shift_z.mean().detach())
        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring


class Regr3D_ScaleInv (Regr3D):
    """ Same than Regr3D but invariant to depth shift.
        if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gt1, gt2, pred1, pred2):
        # compute depth-normalized points
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = super().get_all_pts3d(gt1, gt2, pred1, pred2)

        # measure scene scale
        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts1, gt_pts2, mask1, mask2)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts1, pred_pts2, mask1, mask2)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pred_pts1 *= gt_scale / pred_scale
            pred_pts2 *= gt_scale / pred_scale
            # monitoring = dict(monitoring, pred_scale=(pred_scale/gt_scale).mean())
        else:
            gt_pts1 /= gt_scale
            gt_pts2 /= gt_scale
            pred_pts1 /= pred_scale
            pred_pts2 /= pred_scale
            # monitoring = dict(monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach())

        return gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring

