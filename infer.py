# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import math
from unittest import result
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy
import cv2
import time
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt.oneref_viewer import oneref_viewer_wrapper
import matplotlib.pyplot as pl
import glob
from tqdm import tqdm
from dust3r.datasets.tapvid3d import load_npz_data
import shutil

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights")
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp_st4rtrack', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument("--input_dir", type=str, help="Path to input images directory", default='./assets/feng.mp4')
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for video processing")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for video processing")
    parser.add_argument("--mid_anchor", action='store_true', default=False, help="Use mid anchor for inference")
    
    # HuggingFace model loading
    parser.add_argument('--hf_model', type=str, default=None, help='HuggingFace model repo (e.g., yupengchengg147/St4RTrack)')
    parser.add_argument('--hf_variant', type=str, default='seq', choices=['seq', 'pair'], help='HuggingFace model variant')
    parser.add_argument('--hf_force_download', action='store_true', default=False, help='Force download from HuggingFace')
    return parser

def clean_hf_cache():
    """Clean HuggingFace cache to force fresh download"""
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

def get_reconstructed_scene(
    filelist, 
    model, 
    device, 
    batch_size=16, 
    image_size=512, 
    output_dir='./output', 
    seq_name='sequence', 
    silent=False, 
    start_frame=0, 
    step_size=1, 
    fps=0, 
    num_frames=200,
    dynamic_mask_path=None,
    mid_anchor=False
):
    assert batch_size > 1, "Batch size must be greater than 1"
    """
    Process a list of images through the model and save 3D reconstruction results.
    
    Args:
        filelist: List of image paths or a single .npz file path
        model: Pre-loaded model
        device: Device to run inference on
        batch_size: Batch size for inference
        image_size: Size to resize images to
        output_dir: Directory to save results
        seq_name: Name of the sequence
        silent: Whether to suppress output
        start_frame: First frame to process
        step_size: Frame sampling interval
        fps: FPS for video processing
        num_frames: Maximum number of frames to process
        dynamic_mask_path: Path to dynamic masks
    
    Returns:
        Path to the saved results
    """
    model.eval()
    
    # Handle dynamic mask path
    if dynamic_mask_path is None and seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'

    # Handle tapvid3d data
    if len(filelist) == 1 and filelist[0].endswith('.npz'):
        filelist, tracks_xyz_cam, tracks_uv, intrinsics, tracks_xyz_world, visibility, video_name, extrinsics_w2c = load_npz_data(filelist[0])

    # Load images
    imgs = load_images(
        filelist, 
        size=image_size, 
        verbose=not silent, 
        dynamic_mask_root=dynamic_mask_path, 
        fps=fps, 
        num_frames=num_frames,
        start_frame=start_frame, 
        step_size=step_size
    )
    
    # Handle single image case
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        if mid_anchor:
            output = inference(imgs, model, device, batch_size=batch_size, verbose=not silent, anchor_view=len(imgs)//2)
        else:
            output = inference(imgs, model, device, batch_size=batch_size, verbose=not silent, anchor_view=0)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time} seconds")

    # Create output directory
    save_folder = os.path.join(output_dir, seq_name)
    os.makedirs(save_folder, exist_ok=True)

    # Process and save results
    id = 0
    for idx in range(len(output)):
        view1, view2, pred1, pred2 = output[idx]['view1'], output[idx]['view2'], output[idx]['pred1'], output[idx]['pred2']
        pts1 = pred1['pts3d'].detach().cpu().numpy()
        pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
        
        for batch_idx in range(len(view1['img'])):
            colors1 = rgb(view1['img'][batch_idx])
            colors2 = rgb(view2['img'][batch_idx])
            
            # Save 3D points with colors
            xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)
            xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
            np.save(os.path.join(save_folder, f'pts3d1_p{id}.npy'), xyzrgb1)
            np.save(os.path.join(save_folder, f'pts3d2_p{id}.npy'), xyzrgb2)
            
            # Save confidence maps
            conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
            conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
            np.save(os.path.join(save_folder, f'conf1_p{id}.npy'), conf1)
            np.save(os.path.join(save_folder, f'conf2_p{id}.npy'), conf2)
            
            # Save images
            img1 = colors1 * 255
            img2 = colors2 * 255
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_folder, f'img1_p{id}.png'), img1_rgb)
            cv2.imwrite(os.path.join(save_folder, f'img2_p{id}.png'), img2_rgb)
            
            id += 1
            
    return save_folder


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if args.seq_name == "NULL":
        args.seq_name = '_'.join(args.weights.split('/')[-2:]) + '_' + args.input_dir.split('/')[-2]+args.input_dir.split('/')[-1] + '_' + str(args.start_frame) + '_' + str(args.step_size) + '_' + str(args.batch_size)

    if args.mid_anchor:
        args.seq_name += '_midanchor'
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        tempfile.tempdir = args.output_dir

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    # Load model - prioritize HF model if specified
    if args.hf_model is not None:
        print(f'Loading HuggingFace model: {args.hf_model} (variant: {args.hf_variant})')
        model = load_hf_model(args.hf_model, args.hf_variant, args.device, args.hf_force_download)
    else:
        # Use local weights or model_name
        if args.weights is not None and os.path.exists(args.weights):
            weights_path = args.weights
        else:
            weights_path = args.model_name
        model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    if not args.silent:
        print('Outputting results in', args.output_dir)
    
    if args.input_dir is not None:
        # Process images in the input directory
        if os.path.isdir(args.input_dir):
            input_files = [os.path.join(args.input_dir, fname) for fname in sorted(os.listdir(args.input_dir))]
        else:
            input_files = [args.input_dir]
        
        # Call the function with parameters from args
        save_folder = get_reconstructed_scene(
            filelist=input_files,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            output_dir=args.output_dir,
            seq_name=args.seq_name,
            silent=args.silent,
            start_frame=args.start_frame,
            step_size=args.step_size,
            fps=args.fps,
            num_frames=args.num_frames,
            dynamic_mask_path=None if not args.use_gt_davis_masks else f'data/davis/DAVIS/masked_images/480p/{args.seq_name}',
            mid_anchor=args.mid_anchor
        )
        
        print(f"Processing completed. Output saved in {save_folder}")
