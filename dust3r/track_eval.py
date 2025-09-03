import os
import math
import glob
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import time
import cv2
import sys
import json
sys.path.append(".")
sys.path.append("dust3r")
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.datasets.tapvid3d import load_npz_data, visualize_results, load_npz_data_recon, visualize_results_recon
from dust3r.track_eval_util import compute_average_pts_within_thresh, get_visibility_from_depth
import croco.utils.misc as misc  # "croco" submodule with "utils.misc"
from natsort import natsorted

def save_prediction_as_npy(pred_tracks, save_dir, video_name, eval_recon=False):
    """
    Save prediction results as NPY files for later re-evaluation.
    
    Args:
        pred_tracks: Tensor or numpy array of predicted 3D tracks
        output_dir: Directory to save the NPY files
        video_name: Name of the video/sequence
        eval_recon: Whether this is a reconstruction evaluation
    
    Returns:
        Path to the saved NPY file
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if isinstance(pred_tracks, torch.Tensor):
        pred_tracks = pred_tracks.detach().cpu().numpy()
    
    # Create filename with timestamp to avoid overwriting
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "recon" if eval_recon else "track"
    filename = f"{video_name}_{mode}_{timestamp}.npy"
    save_path = os.path.join(save_dir, filename)
    
    # Save the prediction
    np.save(save_path, pred_tracks)
    print(f"Saved prediction to {save_path}")
    
    return save_path

def eval_ours_tapvid3d(
        args,
        model,
        device,
        data_type='pstudio',
        eval_recon=False,
        visualize=False,
        visualize_all=False,  # New flag to control visualization frequency
        dyn_static=True,
        save_predictions=True,  # New flag to save predictions
        data_root='./data/worldtrack_release'  # Configurable data root path
    ):
    output_dir = args.output_dir
    num_frames = args.num_frames
    return eval_tapvid3d(
        output_dir,
        num_frames,
        load_offline_data=None,
        args=args,
        model=model,
        device=device,
        data_type=data_type,
        eval_recon=eval_recon,
        visualize=visualize,
        visualize_all=visualize_all,
        dyn_static=dyn_static,
        save_predictions=save_predictions,
        data_root=data_root
    )
    
def get_inference_output(args, model, device, filelist, eval_recon=False):
    """
    Get the inference output for a list of images
    """
    model.eval()
    with torch.no_grad():
        imgs = load_images(
            filelist,
            size=512,
            num_frames=args.num_frames,
            step_size=args.pose_eval_stride,
            verbose=False,
            square_ok=True,
            crop=False,
        )
        output = inference(imgs, model, device, batch_size=args.eval_batch_size)
    if eval_recon:
        gathered_output = []
        for i in range(len(output)):
            gathered_output.append(output[i]['pred2']['pts3d_in_other_view'])
        return torch.cat(gathered_output, dim=0)
    else:
        gathered_output = []
        for i in range(len(output)):
            gathered_output.append(output[i]['pred1']['pts3d'])
        return torch.cat(gathered_output, dim=0)

def eval_tapvid3d(
        output_dir,
        num_frames,
        load_offline_data=None,
        is_filtered_tracks=False,
        args=None,
        model=None,
        device=None,
        data_type='pstudio',
        eval_recon=False,
        visualize=False,
        visualize_all=False,  # New flag to control visualization frequency
        dyn_static=True,
        save_predictions=False,  # New flag to save predictions
        sim3_for_recon=False,
        data_root='./data/worldtrack_release',  # Configurable data root path
    ):
    """
    Distributed evaluation for TapVid3D data. 
      1) Inference on each npz track
      2) Evaluate using "metric-based" scaling: global & per-trajectory
      3) Evaluate using "non-metric" thresholds: fixed & adaptive
      4) Optionally evaluate dynamic vs. static subsets (via dyn_static)
      5) Gather & average results
      6) Optionally save predictions for later re-evaluation
    """

    # Setup visualization directory if needed
    if visualize:
        vis_dir = os.path.join(output_dir, 
                              f"recon_eval_{data_type}" if eval_recon else f"track_eval_{data_type}")
        os.makedirs(vis_dir, exist_ok=True)

    # Setup directory for saved predictions if needed
    if save_predictions:
        pred_dir = os.path.join(output_dir, f"saved_predictions_{data_type}")
        os.makedirs(pred_dir, exist_ok=True)

    # Gather .npz files
    track_list = glob.glob(f'{data_root}/{data_type}/*.npz')

    track_list = natsorted(track_list)
    if len(track_list) == 0:
        print(f"No .npz files found in {data_root}/{data_type}.")
        return []

    # Determine rank-based splitting (if distributed)
    sub_track_list = track_list

    # Loop over sub-list to gather local_results
    local_results = []
    saved_prediction_paths = []  # To store paths of saved predictions

    for track_idx, track_npz in enumerate(tqdm(sub_track_list)):
        
        # Load data (either from recon or normal .npz)
        if eval_recon:
            filelist, depth_map, intrinsics, recon_xyz_world, recon_xyz_cam, \
                visibility, video_name, extrinsics_w2c = \
                load_npz_data_recon(track_npz, num_frames=num_frames)

        else:
            filelist, tracks_xyz_cam, tracks_uv, intrinsics, tracks_xyz_world, \
                visibility, video_name, extrinsics_w2c = \
                load_npz_data(track_npz, num_frames=num_frames)

        if load_offline_data is None:
            assert model is not None, "model is required if load_offline_data is False"
            pred_tracks = get_inference_output(args, model, device, filelist, eval_recon=eval_recon)
            
            # Save predictions if requested
            if save_predictions:
                save_path = save_prediction_as_npy(pred_tracks, pred_dir, video_name, eval_recon)
                saved_prediction_paths.append(save_path)
        else:

            handle = track_npz.split("/")[-1].split(".")[0]
            assert handle in load_offline_data[track_idx], f"{handle} not in {load_offline_data[track_idx]}"
            print(f"Processing {track_npz}, {load_offline_data[track_idx]}")
            # if it is a npy file, load it
            if load_offline_data[track_idx].endswith('.npy'):
                pred_tracks = np.load(load_offline_data[track_idx])
            elif os.path.isdir(load_offline_data[track_idx]):
                # load all npy files in the directory
                if eval_recon:
                    pred_tracks = np.stack([np.load(f) for f in natsorted(glob.glob(os.path.join(load_offline_data[track_idx], "pts3d2_*.npy")))], axis=0)[...,:3]
                else:
                    pred_tracks = np.stack([np.load(f) for f in natsorted(glob.glob(os.path.join(load_offline_data[track_idx], "pts3d1_*.npy")))], axis=0)[...,:3]
            else:
                raise ValueError(f"Unknown file type: {load_offline_data[track_idx]}")

            pred_tracks = torch.from_numpy(pred_tracks)
        pred_tracks = pred_tracks[:num_frames]

        # Initialize placeholders for dynamic fraction dictionaries
        fractions_sim3_closed_dyn = {}
        avg_pts_sim3_closed_dyn = float('nan')

        # Compute metrics for reconstruction or track-based data
        if eval_recon:
            # Evaluate reconstruction

            pred_tracks = F.interpolate(pred_tracks.permute(0, 3, 1, 2),
                                        size=recon_xyz_world.shape[1:3],
                                        mode='bilinear',
                                        align_corners=False).permute(0, 2, 3, 1)
            gt_tracks = recon_xyz_world[:num_frames]  # shape (T, H, W, 3)

            # based on depth map, get the visibility
            depth_maps = depth_map[:num_frames]
            visibility = get_visibility_from_depth(depth_maps)
            gt_tracks_filtered = gt_tracks[visibility]  # shape (N, 3)
            pred_tracks_filtered = pred_tracks[visibility]

            # Expand dims to shape (1, N, 3) for compute_average_pts_within_thresh
            gt_tracks_filtered = np.expand_dims(gt_tracks_filtered, 0)  # shape (1, N, 3)
            pred_tracks_filtered = pred_tracks_filtered.unsqueeze(0)   # shape (1, N, 3)

            if sim3_for_recon:
                # Add sim3 alignment calculation
                avg_pts_sim3, pred_aligned_sim3, fractions_sim3, (s_s, R_s, t_s), epe_sim3 = \
                    compute_average_pts_within_thresh(
                        gt_tracks_filtered,
                        pred_tracks_filtered,
                        scaling="sim3",
                        intrinsics_params=intrinsics,
                        compute_epe=True
                    )
            else:
                avg_pts_sim3 = float('nan')
                fractions_sim3 = {}
                epe_sim3 = float('nan')

            # Still keep per-traj as NaN since it's not meaningful for reconstruction evaluation
            avg_pts_pertraj = float('nan')
            epe_pertraj = float('nan')
            fractions_pertraj = {}
            
            # Dynamic point evaluation related metrics remain as NaN
            fractions_sim3_closed_dyn = {}
            avg_pts_sim3_closed_dyn = float('nan')
            epe_sim3_closed_dyn = float('nan')
            avg_pts_global_dyn = float('nan')
            avg_pts_sim3_dyn = float('nan')
            epe_sim3_dyn = float('nan')
            fractions_global_dyn = {}
            epe_global_dyn = float('nan')

        else:
            # Evaluate standard track data

            # Extract the visible subset from the first frame
            visibility_mask = visibility[0]  # shape (N,)
            if visibility_mask.sum() == 0:
                print(f"Warning: No visible points in {video_name}")
                continue

            query_uv = np.array(tracks_uv)[0, visibility_mask]  # (M, 2)
            gt_tracks_filtered = tracks_xyz_world[:num_frames, visibility_mask]  # (T, M, 3)
            if not is_filtered_tracks:

                # Adjust uv coords to match pred_track resolution
                gt_image_size = filelist[0].size        # e.g. (width, height) from PIL
                pred_image_size = pred_tracks.shape[1:3][::-1]  # (width, height) from the tensor shape
                query_uv = query_uv * np.array(pred_image_size) / np.array(gt_image_size)

                # Filter out-of-bounds points
                oob_mask = (
                    (query_uv[:, 0] >= 0) &
                    (query_uv[:, 0] < pred_image_size[0]) &
                    (query_uv[:, 1] >= 0) &
                    (query_uv[:, 1] < pred_image_size[1])
                )
                if oob_mask.sum() < len(query_uv):
                    print(f"Warning: {len(query_uv) - oob_mask.sum()} out-of-bounds points in {video_name}, "
                        f"{query_uv.max(0)}")
                    query_uv = query_uv[oob_mask]
                    gt_tracks_filtered = gt_tracks_filtered[:, oob_mask]

                # Gather predicted 3D for those uv coords
                pred_tracks_filtered = pred_tracks[
                    :num_frames,
                    query_uv[:, 1].astype(int),  # row
                    query_uv[:, 0].astype(int),  # col
                ]  # shape (T, M, 3)
            else:
                pred_tracks_filtered = pred_tracks

            # Identify dynamic subset if requested
            if dyn_static:
                total_motion = gt_tracks_filtered[1:] - gt_tracks_filtered[:-1]
                total_motion_norm = np.linalg.norm(total_motion, axis=-1).sum(0)
                dyn_mask = (total_motion_norm > 0.01)
                print(
                    f"fraction of dynamic points: {dyn_mask.mean()}, "
                    f"number of dynamic points: {dyn_mask.sum()}, "
                    f"maximum motion: {total_motion_norm.max()}"
                )
            else:
                dyn_mask = None

            # Compute the standard metrics
            # check shape match
            if gt_tracks_filtered.shape != pred_tracks_filtered.shape:
                print(f"Warning: gt_tracks_filtered.shape != pred_tracks_filtered.shape, {gt_tracks_filtered.shape}, {pred_tracks_filtered.shape}")

            avg_pts_pertraj, pred_aligned_pertraj, fractions_pertraj, (s_p, R_p, t_p), epe_pertraj = \
                compute_average_pts_within_thresh(
                    gt_tracks_filtered,
                    pred_tracks_filtered,
                    scaling="per_traj",
                    intrinsics_params=intrinsics,
                    compute_epe=True
                )
            if sim3_for_recon:
                avg_pts_sim3, pred_aligned_sim3, fractions_sim3, (s_s, R_s, t_s), epe_sim3 = \
                compute_average_pts_within_thresh(
                    gt_tracks_filtered,
                    pred_tracks_filtered,
                    scaling="sim3",
                    intrinsics_params=intrinsics,
                    compute_epe=True
                )
            else:
                avg_pts_sim3 = float('nan')
                fractions_sim3 = {}
                epe_sim3 = float('nan')

            # ------------------------------------------------------------------
            # *Dynamic subset evaluation* (sim3_closed only, as your code does)
            # ------------------------------------------------------------------
            avg_pts_sim3_closed_dyn = float('nan')
            fractions_sim3_closed_dyn = {}
            epe_sim3_closed_dyn = float('nan')
            avg_pts_global_dyn = float('nan')
            fractions_global_dyn = {}
            avg_pts_sim3_dyn = float('nan')
            epe_sim3_dyn = float('nan')
            fractions_sim3_dyn = {}
            epe_global_dyn = float('nan')
            if dyn_static and dyn_mask is not None and not eval_recon:
                # Evaluate only if we have dynamic points
                if dyn_mask.sum() > 0:
        
                    avg_pts_sim3_closed_dyn, pred_aligned_sim3_closed_dyn, fractions_sim3_closed_dyn, _, epe_sim3_closed_dyn = \
                        compute_average_pts_within_thresh(
                            gt_tracks_filtered[:, dyn_mask],
                            pred_tracks_filtered[:, dyn_mask],
                            scaling="sim3_closed",
                            use_fixed_metric_threshold=True,
                            intrinsics_params=intrinsics,
                            pred_aligned=None,  # We'll align from scratch or pass aligned
                        )
                    # global alignment
                    avg_pts_global_dyn, pred_aligned_global_dyn, fractions_global_dyn, _, epe_global_dyn = \
                        compute_average_pts_within_thresh(
                            gt_tracks_filtered[:, dyn_mask],
                            pred_tracks_filtered[:, dyn_mask],
                            scaling="global",
                            use_fixed_metric_threshold=True,
                            intrinsics_params=intrinsics,
                            pred_aligned=None,  # We'll align from scratch or pass aligned
                        )
                    if sim3_for_recon:
                        avg_pts_sim3_dyn, _, fractions_sim3_dyn, _, epe_sim3_dyn = \
                        compute_average_pts_within_thresh(
                            gt_tracks_filtered[:, dyn_mask],
                            pred_tracks_filtered[:, dyn_mask],
                            scaling="sim3",
                            use_fixed_metric_threshold=True,
                            intrinsics_params=intrinsics,
                            pred_aligned=None,  # We'll align from scratch or pass aligned
                        )
                    else:
                        avg_pts_sim3_dyn = float('nan')
                        fractions_sim3_dyn = {}
                        epe_sim3_dyn = float('nan')

        # ----------------------------------------------------------------------
        # *Compute "global" and "sim3_closed" metrics for all (even if recon)*
        # ----------------------------------------------------------------------
        avg_pts_global, pred_aligned_global, fractions_global, (s_g, R_g, t_g), epe_global = \
            compute_average_pts_within_thresh(
                gt_tracks_filtered,
                pred_tracks_filtered,
                scaling="global",
                intrinsics_params=intrinsics,
                compute_epe=True
            )

        avg_pts_sim3_closed, pred_aligned_sim3_closed, fractions_sim3_closed, (s_sc, R_sc, t_sc), epe_sim3_closed = \
            compute_average_pts_within_thresh(
                gt_tracks_filtered,
                pred_tracks_filtered,
                scaling="sim3_closed",
                intrinsics_params=intrinsics,
                compute_epe=True
            )

        # -------------------------------------------------
        # (d) Visualization if needed
        # -------------------------------------------------
        should_visualize = visualize and (
            visualize_all or track_idx % 10 == 0
        )

        if should_visualize:
            video_name = os.path.splitext(os.path.basename(track_npz))[0]
            vis_subdir = os.path.join(vis_dir, video_name)
            os.makedirs(vis_subdir, exist_ok=True)

            if not eval_recon:
                visualize_results(
                    filelist, 
                    gt_tracks_filtered, 
                    pred_aligned_sim3_closed, 
                    extrinsics_w2c, 
                    intrinsics, 
                    save_path=os.path.join(vis_subdir, f"track_vis_{avg_pts_sim3_closed:.4f}_dyn{dyn_mask.mean():.4f}")
                )
            else:
                # count the number of points in gt_tracks
                num_points = gt_tracks.shape[0]
                print(f"number of points in gt_tracks per frame: {num_points // num_frames}")

                T, H, W, _ = pred_tracks.shape
                pred_tracks_aligned = (
                    s_g
                    * (torch.from_numpy(R_g).to(pred_tracks.device).float() @ pred_tracks.reshape(-1, 3).T).T
                    + torch.from_numpy(t_g).to(pred_tracks.device).float()
                )
                pred_tracks_aligned = pred_tracks_aligned.reshape(T, H, W, 3)

                visualize_results_recon(
                    filelist[:num_frames], 
                    gt_tracks,
                    pred_tracks_aligned, 
                    save_path=os.path.join(vis_subdir, f"recon_vis_global_{avg_pts_global:.4f}")
                )

                pred_tracks_aligned_sim3_closed = (
                    s_sc
                    * (torch.from_numpy(R_sc).to(pred_tracks.device).float() @ pred_tracks.reshape(-1, 3).T).T
                    + torch.from_numpy(t_sc).to(pred_tracks.device).float()
                )
                pred_tracks_aligned_sim3_closed = pred_tracks_aligned_sim3_closed.reshape(T, H, W, 3)
                
                visualize_results_recon(
                    filelist[:num_frames], 
                    gt_tracks,
                    pred_tracks_aligned_sim3_closed, 
                    save_path=os.path.join(vis_subdir, f"recon_vis_sim3_closed_{avg_pts_sim3_closed:.4f}")
                )
                
                if sim3_for_recon:
                    # Add sim3 alignment visualization
                    pred_tracks_aligned_sim3 = (
                        s_s
                        * (torch.from_numpy(R_s).to(pred_tracks.device).float() @ pred_tracks.reshape(-1, 3).T).T
                        + torch.from_numpy(t_s).to(pred_tracks.device).float()
                    )
                    pred_tracks_aligned_sim3 = pred_tracks_aligned_sim3.reshape(T, H, W, 3)
                
                    visualize_results_recon(
                        filelist[:num_frames], 
                        gt_tracks,
                        pred_tracks_aligned_sim3, 
                        save_path=os.path.join(vis_subdir, f"recon_vis_sim3_{avg_pts_sim3:.4f}")
                    )

        # -------------------------------------------------
        # (e) Construct local result dict
        # -------------------------------------------------
        result_dict = {
            'video_name': video_name,

            # Scalar metrics
            'avg_pts_global':      float(avg_pts_global),
            'avg_pts_pertraj':     float(avg_pts_pertraj),
            'avg_pts_sim3':        float(avg_pts_sim3),
            'avg_pts_sim3_dyn':    float(avg_pts_sim3_dyn),
            'avg_pts_sim3_closed': float(avg_pts_sim3_closed),
            'avg_pts_sim3_closed_dyn': float(avg_pts_sim3_closed_dyn),
            'avg_pts_global_dyn': float(avg_pts_global_dyn),

            # EPE metrics
            'epe_global':      float(epe_global),
            'epe_pertraj':     float(epe_pertraj),
            'epe_sim3':        float(epe_sim3),
            'epe_sim3_dyn':    float(epe_sim3_dyn),
            'epe_sim3_closed': float(epe_sim3_closed),
            'epe_sim3_closed_dyn': float(epe_sim3_closed_dyn),
            'epe_global_dyn': float(epe_global_dyn),

            # Fraction dictionaries
            'fractions_global':      dict(fractions_global),  
            'fractions_pertraj':     dict(fractions_pertraj),
            'fractions_sim3':        dict(fractions_sim3),
            'fractions_sim3_closed': dict(fractions_sim3_closed),

            # Dynamic subset metrics (only relevant if dyn_static && !eval_recon)
            'avg_pts_sim3_closed_dyn': float(avg_pts_sim3_closed_dyn),
            'fractions_sim3_closed_dyn': dict(fractions_sim3_closed_dyn),
        }

        local_results.append(result_dict)

        # Print intermediate results
        for key, value in result_dict.items():
            print(f"{key}: {value}")
        print()

    ############################################################################
    # 4) Gather results across all ranks
    ############################################################################

    combined_results = local_results

    # --------------------------------------------------------------------------
    # (A) Aggregate scalar metrics
    # --------------------------------------------------------------------------
    sum_global = 0.0
    sum_pertraj = 0.0
    sum_sim3 = 0.0
    sum_sim3_dyn = 0.0
    sum_sim3_closed = 0.0
    sum_sim3_closed_dyn = 0.0  # dynamic subset
    sum_global_dyn = 0.0

    sum_epe_global = 0.0
    sum_epe_pertraj = 0.0
    sum_epe_sim3 = 0.0
    sum_epe_sim3_dyn = 0.0
    sum_epe_sim3_closed = 0.0
    sum_epe_sim3_closed_dyn = 0.0
    sum_epe_global_dyn = 0.0

    # Counters
    cnt_global = 0
    cnt_pertraj = 0
    cnt_sim3 = 0
    cnt_sim3_dyn = 0
    cnt_sim3_closed = 0
    cnt_sim3_closed_dyn = 0
    cnt_global_dyn = 0
    # For EPE, we'll just count the total number of results
    # (you could do separate counters if you want to handle missing data differently)
    cnt_epe = 0

    # --------------------------------------------------------------------------
    # (B) Aggregate fraction metrics
    # --------------------------------------------------------------------------
    fraction_keys = [
        'fractions_global',
        'fractions_pertraj',
        'fractions_sim3',
        'fractions_sim3_closed',
        'fractions_sim3_closed_dyn',  # dynamic subset
    ]

    fraction_aggregator = {
        fkey: defaultdict(list) for fkey in fraction_keys
    }

    # Loop over all results
    for res in combined_results:
        # 1. Accumulate scalar sums
        if not math.isnan(res['avg_pts_global']):
            sum_global += res['avg_pts_global']
            cnt_global += 1

        if not math.isnan(res['avg_pts_pertraj']):
            sum_pertraj += res['avg_pts_pertraj']
            cnt_pertraj += 1

        if not math.isnan(res['avg_pts_sim3']):
            sum_sim3 += res['avg_pts_sim3']
            cnt_sim3 += 1

        if not math.isnan(res['avg_pts_sim3_dyn']):
            sum_sim3_dyn += res['avg_pts_sim3_dyn']
            cnt_sim3_dyn += 1

        if not math.isnan(res['avg_pts_sim3_closed']):
            sum_sim3_closed += res['avg_pts_sim3_closed']
            cnt_sim3_closed += 1

        if not math.isnan(res['avg_pts_sim3_closed_dyn']):
            sum_sim3_closed_dyn += res['avg_pts_sim3_closed_dyn']
            cnt_sim3_closed_dyn += 1

        if not math.isnan(res['avg_pts_global_dyn']):
            sum_global_dyn += res['avg_pts_global_dyn']
            cnt_global_dyn += 1

        # EPE
        if not math.isnan(res['epe_global']):
            sum_epe_global += res['epe_global']
            cnt_epe += 1
        if not math.isnan(res['epe_pertraj']):
            sum_epe_pertraj += res['epe_pertraj']
        if not math.isnan(res['epe_sim3']):
            sum_epe_sim3 += res['epe_sim3']
        if not math.isnan(res['epe_sim3_dyn']):
            sum_epe_sim3_dyn += res['epe_sim3_dyn']
        if not math.isnan(res['epe_sim3_closed']):
            sum_epe_sim3_closed += res['epe_sim3_closed']
        if not math.isnan(res['epe_sim3_closed_dyn']):
            sum_epe_sim3_closed_dyn += res['epe_sim3_closed_dyn']
        if not math.isnan(res['epe_global_dyn']):
            sum_epe_global_dyn += res['epe_global_dyn']

        # 2. Accumulate fraction dicts
        for fkey in fraction_keys:
            frac_dict = res.get(fkey, {})
            if frac_dict is None:
                continue
            for thr, val in frac_dict.items():
                fraction_aggregator[fkey][thr].append(val)

    # --------------------------------------------------------------------------
    # (C) Final averages for scalar metrics
    # --------------------------------------------------------------------------
    final_global = sum_global / cnt_global if cnt_global > 0 else float('nan')
    final_pertraj = sum_pertraj / cnt_pertraj if cnt_pertraj > 0 else float('nan')
    final_sim3 = sum_sim3 / cnt_sim3 if cnt_sim3 > 0 else float('nan')
    final_sim3_dyn = sum_sim3_dyn / cnt_sim3_dyn if cnt_sim3_dyn > 0 else float('nan')
    final_sim3_closed = sum_sim3_closed / cnt_sim3_closed if cnt_sim3_closed > 0 else float('nan')
    final_sim3_closed_dyn = sum_sim3_closed_dyn / cnt_sim3_closed_dyn if cnt_sim3_closed_dyn > 0 else float('nan')
    final_global_dyn = sum_global_dyn / cnt_global_dyn if cnt_global_dyn > 0 else float('nan')

    # EPE averages (shared denominators for EPE is somewhat of a design choice)
    final_epe_global = sum_epe_global / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_pertraj = sum_epe_pertraj / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_sim3 = sum_epe_sim3 / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_sim3_dyn = sum_epe_sim3_dyn / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_sim3_closed = sum_epe_sim3_closed / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_sim3_closed_dyn = sum_epe_sim3_closed_dyn / cnt_epe if cnt_epe > 0 else float('nan')
    final_epe_global_dyn = sum_epe_global_dyn / cnt_epe if cnt_epe > 0 else float('nan')

    # --------------------------------------------------------------------------
    # (D) Final averages for fraction dicts
    # --------------------------------------------------------------------------
    final_fractions = {}
    for fkey in fraction_keys:
        final_fractions[fkey] = {}
        for thr, val_list in fraction_aggregator[fkey].items():
            final_fractions[fkey][thr] = float(np.mean(val_list))

    # --------------------------------------------------------------------------
    # (E) Print or return final results
    # --------------------------------------------------------------------------

    log_str = "\n=== Final Results ===\n"
    
    # Scalar metrics
    log_str += "\nThreshold-based metrics:\n"
    log_str += f"  Global alignment:     {final_global:.4f}\n"
    log_str += f"  Per-traj alignment:   {final_pertraj:.4f}\n"
    log_str += f"  Sim3 alignment:       {final_sim3:.4f}\n"
    log_str += f"  Sim3 alignment (dyn): {final_sim3_dyn:.4f}\n"
    log_str += f"  Sim3 closed:          {final_sim3_closed:.4f}\n"
    log_str += f"  Sim3 closed (dyn):    {final_sim3_closed_dyn:.4f}\n"
    log_str += f"  Global (dyn):         {final_global_dyn:.4f}\n"

    # EPE metrics
    log_str += "\nEnd Point Error (EPE):\n"
    log_str += f"  Global alignment:     {final_epe_global:.4f}\n"
    log_str += f"  Per-traj alignment:   {final_epe_pertraj:.4f}\n"
    log_str += f"  Sim3 alignment:       {final_epe_sim3:.4f}\n"
    log_str += f"  Sim3 alignment (dyn): {final_epe_sim3_dyn:.4f}\n"
    log_str += f"  Sim3 closed:          {final_epe_sim3_closed:.4f}\n"
    log_str += f"  Sim3 closed (dyn):    {final_epe_sim3_closed_dyn:.4f}\n"
    log_str += f"  Global (dyn):         {final_epe_global_dyn:.4f}\n"

    # Fraction metrics for each threshold
    log_str += "\nDetailed fraction metrics:\n"
    for fkey, thr_dict in final_fractions.items():
        log_str += f"\n{fkey}:\n"
        for thr, val in sorted(thr_dict.items()):
            log_str += f"  threshold={thr}: {val:.4f}\n"

    print(log_str)

    # Save to log file if output_dir is specified
    if output_dir:
        log_file = os.path.join(output_dir, f"track_eval_{data_type}.txt" if not eval_recon else f"recon_eval_{data_type}.txt")
        with open(log_file, "a") as f:
            f.write(f"\n=== Evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(log_str)

    # Save mapping of NPZ files to prediction files if we saved predictions
    if save_predictions and saved_prediction_paths:
        mapping_file = os.path.join(output_dir, f"prediction_mapping_{data_type}.json")
        mapping = {}
        for i, (npz_path, pred_path) in enumerate(zip(sub_track_list, saved_prediction_paths)):
            mapping[npz_path] = {
                "prediction_path": pred_path,
                "video_name": os.path.splitext(os.path.basename(npz_path))[0],
                "eval_recon": eval_recon
            }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved prediction mapping to {mapping_file}")

    # For compatibility with existing code, return tuple of main metrics
    return (
        final_global,      # Global alignment threshold metric
        final_pertraj,     # Per-trajectory threshold metric
        final_sim3,        # Sim3 threshold metric
        final_sim3_closed, # Sim3 closed-form threshold metric
        final_sim3_closed_dyn,  # Sim3 closed-form dynamic threshold metric
        final_epe_global,  # Global alignment EPE
        final_epe_pertraj, # Per-trajectory EPE
        final_epe_sim3,    # Sim3 EPE
        final_epe_sim3_closed  # Sim3 closed-form EPE
    )

def load_and_eval_saved_predictions(mapping_file, output_dir=None, visualize=False, dyn_static=True, data_root='./data/worldtrack_release'):
    """
    Load saved predictions from a mapping file and re-evaluate them.
    
    Args:
        mapping_file: Path to the JSON mapping file
        output_dir: Directory to save evaluation results (defaults to directory containing mapping file)
        visualize: Whether to visualize results
        dyn_static: Whether to evaluate dynamic vs static subsets
    
    Returns:
        Evaluation results
    """
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(mapping_file)
    
    # Extract NPZ paths and prediction paths
    npz_paths = list(mapping.keys())
    pred_paths = [item["prediction_path"] for item in mapping.values()]
    eval_recon = mapping[npz_paths[0]]["eval_recon"]  # Assume all entries have same eval_recon value
    
    # Get data_type from the first NPZ path
    data_type = os.path.basename(os.path.dirname(npz_paths[0]))
    
    # Call eval_tapvid3d with the saved predictions
    return eval_tapvid3d(
        output_dir=output_dir,
        num_frames=None,  # Will be determined from the loaded data
        load_offline_data=pred_paths,
        is_filtered_tracks=False,
        args=None,
        data_type=data_type,
        eval_recon=eval_recon,
        visualize=visualize,
        visualize_all=False,
        dyn_static=dyn_static,
        data_root=data_root
    )

if __name__ == "__main__":

    
    result_list_path = ""
    data_types = ["pstudio_mini"]
    
    for data_type in data_types:
        result_list = natsorted(glob.glob(os.path.join(f"results_tta_single_{data_type}_mast3r", "*/tta_eval_epoch_3/*/")))
        print(len(result_list), "results")
        output_dir = f"./tmp_test_mast3r_tta_3"
        os.makedirs(output_dir, exist_ok=True)
        eval_tapvid3d(
            output_dir=output_dir,
            num_frames=64,
            load_offline_data=result_list,
            is_filtered_tracks=False,
            args=None,
            data_type=data_type,
            eval_recon=False,
            visualize=False,
            visualize_all=True,
            dyn_static=True,
            save_predictions=False
        )