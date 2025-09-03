# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb
from dust3r.camera_solver import CameraLoss

import time
import torch
import copy
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor) and value1.ndim == value2.ndim:
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def visualize_results(view1, view2, pred1, pred2, save_dir='./tmp', save_name=None, visualize_type='gt', frame_idx = 0):
    # visualize_type: 'gt' or 'pred'
    viz = SceneViz()
    views = [copy.deepcopy(view1), copy.deepcopy(view2)]
    poses = [views[view_idx]['camera_pose'][frame_idx] for view_idx in [0, 1]]
    cam_size = max(auto_cam_size(poses), 0.5)
    cam_size *= 0.1
    if visualize_type == 'pred':
        views[0]['pts3d'] = geotrf(poses[0], pred1['pts3d']) # convert from X_camera1 to X_world
        views[1]['pts3d'] = geotrf(poses[0], pred2['pts3d_in_other_view'])
    for view_idx in [0, 1]:
        pts3d = views[view_idx]['pts3d'][frame_idx]

        valid_mask = views[view_idx]['valid_mask'][frame_idx]
        colors = rgb(views[view_idx]['img'][frame_idx])
        if view_idx == 0:
            # set 2d rainbow color for view1
            rainbow_colors = np.zeros((colors.shape[0], colors.shape[1], 3), dtype=np.uint8)
            for i in range(colors.shape[0]):
                for j in range(colors.shape[1]):
                    rainbow_colors[i, j] = [int(255 * (j / colors.shape[1])), 255, int(255 * (i / colors.shape[0]))]
            colors = rainbow_colors

        viz.add_pointcloud(pts3d, colors, valid_mask) if visualize_type == 'gt' else viz.add_pointcloud(pts3d, colors, valid_mask)
        viz.add_camera(pose_c2w=views[view_idx]['camera_pose'][frame_idx],
                    focal=views[view_idx]['camera_intrinsics'][0, 0],
                    color=(255, 0, 0) if view_idx == 0 else (0, 0, 255),
                    image=colors,
                    cam_size=cam_size)

        if view_idx == 0:
            v1_traj_mask = views[view_idx]['traj_mask'][frame_idx]
            if visualize_type == 'gt':
                v1_traj = views[view_idx]['traj_ptc'][frame_idx]
            else:
                v1_traj = pts3d[v1_traj_mask].clone()
            traj_col = (255, 0, 0) 
            viz.add_pointcloud(v1_traj, traj_col)

    if save_name is None:
        save_name = f'{views[0]["dataset"][0]}_{views[0]["label"][frame_idx][0] if isinstance(views[0]["label"][frame_idx], list) else views[0]["label"][frame_idx]}_{frame_idx}_{views[0]["instance"][frame_idx][0]}_{views[1]["instance"][frame_idx][0]}_{visualize_type}'
    save_path = save_dir+'/'+save_name+'.glb'
    print(f'Saving visualization to {save_path}')
    del views
    del poses
    return viz.save_glb(save_path)



def get_batches(batch_data, anchor_view=0):
    """
    Creates pairs of views for batch processing.
    
    The first item in batch_data is the 'base'.
    We pair it with every item (including itself).
    Instead of deepcopying large tensors, we do shallow copies
    and only override 'traj_ptc'/'traj_mask' if present.
    """
    batches = []
    # Only do shallow copy once, preserve dict structure without copying underlying tensors
    view1_base = dict(batch_data[anchor_view])

    for idx in range(len(batch_data)):
        # Make another shallow copy for field overrides
        view1 = dict(view1_base)
        view2 = batch_data[idx]

        # Override fields as needed
        if 'traj_ptc' in view2 and 'traj_mask' in view2:
            view1['traj_ptc'] = view2['traj_ptc']
            view1['traj_mask'] = view2['traj_mask']
        
        batches.append([view1, view2])

    return batches

def loss_of_one_batch(batch_data, model, criterion, device,
                      symmetrize_batch=False, use_amp=False, ret=None, print_time=False, 
                      add_batch_dim=False):
    """Compute the loss (and optionally other results) for one batch with timing.
    
    Args:
        batch_data: Tuple of (view1_combined, view2_combined)
        model: Model to run inference with
        criterion: Loss criterion (can be None)
        device: Device to run on
        symmetrize_batch: Whether to symmetrize the batch
        use_amp: Whether to use automatic mixed precision
        ret: If not None, only return this key from the result dict
        print_time: Whether to print timing information
        add_batch_dim: Whether to add a batch dimension to 3D tensors
    """
    timings = {}

    # 1) Split out multi-view sub-batches from batch_data
    start_time = time.time()
    view1_combined, view2_combined = batch_data
    
    for key in view1_combined:
        if isinstance(view1_combined[key], torch.Tensor):
            view1_combined[key] = view1_combined[key].squeeze().to(device, non_blocking=True)
            # Add batch dimension if needed
            if add_batch_dim and view1_combined[key].ndim == 3:
                view1_combined[key] = view1_combined[key].unsqueeze(0)
    
    for key in view2_combined:
        if isinstance(view2_combined[key], torch.Tensor):
            view2_combined[key] = view2_combined[key].squeeze().to(device, non_blocking=True)
            # Add batch dimension if needed
            if add_batch_dim and view2_combined[key].ndim == 3:
                view2_combined[key] = view2_combined[key].unsqueeze(0)

    # 5) Symmetrize (if needed)
    start_time = time.time()
    if symmetrize_batch:
        view1_combined, view2_combined = make_batch_symmetric((view1_combined, view2_combined))
    timings['symmetrize_batch'] = time.time() - start_time

    # 6) Move to device
    start_time = time.time()
    timings['move_to_device'] = time.time() - start_time

    # 7) Forward + loss
    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1_combined, view2_combined)
    timings['model_forward'] = time.time() - start_time

    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=False):
        loss = criterion(view1_combined, view2_combined, pred1, pred2) if criterion else None
    timings['compute_loss'] = time.time() - start_time

    # 8) Final
    start_time = time.time()
    result = dict(
        view1=view1_combined,
        view2=view2_combined,
        pred1=pred1,
        pred2=pred2,
        loss=loss,
        timings=timings
    )
    timings['finalize_results'] = time.time() - start_time

    if print_time:
        # Print the timings
        for k, v in timings.items():
            print(f"Timing: {k} = {v:.3f} s")

    return result[ret] if ret else result

@torch.no_grad()
def inference(pairs, model, device, batch_size=16, verbose=False, anchor_view=0):
    all_results = []
    batchdata = get_batches(pairs, anchor_view=anchor_view)
    # Process in chunks of batch_size
    for i in range(0, len(batchdata), batch_size):
        chunk = batchdata[i:i+batch_size]
        if verbose:
            print(f"Processing chunk {i//batch_size + 1}, size {len(chunk)}")
        
        # Get batches for this chunk
        view1s = [batch[0] for batch in chunk]
        view2s = [batch[1] for batch in chunk]

        # Determine which keys are tensors vs. metadata
        tensor_keys = [
            k for k in view1s[0].keys()
            if isinstance(view1s[0][k], torch.Tensor) and k not in ['depthmap']
        ]
        metadata_keys = ['dataset', 'label', 'instance']

        # combine_views function
        def combine_views(views):
            # Direct tensor concatenation for tensor keys
            combined = {
                k: torch.cat([v[k] for v in views], dim=0).unsqueeze(0)
                for k in tensor_keys
            }
            
            # Simple list extension for metadata
            combined.update({
                mk: [v[mk] for v in views]
                for mk in metadata_keys 
                if mk in views[0]
            })
            
            return combined

        # Combine views
        view1_combined = combine_views(view1s)
        view2_combined = combine_views(view2s)

        # Process the chunk - use the merged function with add_batch_dim=True
        res = loss_of_one_batch(
            (view1_combined, view2_combined), 
            model, 
            None, 
            device, 
            print_time=verbose,
            add_batch_dim=True
        )
        all_results.append(res)

        if verbose:
            print(f"Completed chunk {i//batch_size + 1}")

    return all_results


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)
        

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    return scaling
