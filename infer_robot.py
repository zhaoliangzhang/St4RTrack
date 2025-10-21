# --------------------------------------------------------
# Robot inference script
# --------------------------------------------------------

import argparse
import os
import torch
import numpy as np
import time
from pathlib import Path
import cv2
from dust3r.model_robot import AsymmetricCroCo3DStereoRobot
from dust3r.datasets.robot import RobotDUSt3R
from dust3r.robot_losses import RobotConfLoss
from dust3r.inference import loss_of_one_batch
from dust3r.utils.image import rgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import inf

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def get_args_parser():
    parser = argparse.ArgumentParser('Robot inference', add_help=False)
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to robot model checkpoint')
    parser.add_argument('--model', default="AsymmetricCroCo3DStereoRobot(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', \
                        img_size=(256, 144), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), \
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')",
                        type=str, help="Robot model architecture")
    
    # Dataset
    parser.add_argument('--dataset', default="RobotDUSt3R(dataset_location='./robots/allegro/', dset='test', resolution=[(256, 144)], S=16, strides=[1,2,3], clip_step=32)", 
                        type=str, help="Robot dataset")
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size for inference")
    parser.add_argument('--num_workers', default=4, type=int, help="Number of workers")
    
    # Output
    parser.add_argument('--output_dir', default='./robot_inference_output/', type=str, help="Output directory")
    parser.add_argument('--data_index', default=None, type=int, help="Specific data index to test (if specified, only test this index)")
    
    # Device
    parser.add_argument('--device', default='cuda', type=str, help="Device to run on")
    
    return parser

def load_robot_model(checkpoint_path, model_config, device):
    """Load robot model from checkpoint"""
    print(f'Loading robot model from: {checkpoint_path}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = eval(model_config)
    model.to(device)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print(f'Model loaded successfully on {device}')
    
    return model

def create_robot_dataset(dataset_config, batch_size, num_workers):
    """Create robot dataset"""
    from dust3r.datasets import get_data_loader
    
    print(f'Building robot dataset: {dataset_config}')
    loader = get_data_loader(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False
    )
    print(f"Dataset length: {len(loader)}")
    return loader

def save_robot_results(view1, view2, pred1, pred2, save_dir, sample_idx):
    """Save robot inference results exactly like infer.py lines 215-245"""
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get 3D points exactly like infer.py
    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    
    # Use simple id counter like infer.py
    id = 0
    view1['img'] = view1['img'].squeeze(0)
    view2['img'] = view2['img'].squeeze(0)
    for batch_idx in range(len(view1['img'])):
        colors1 = rgb(view1['img'][batch_idx])
        colors2 = rgb(view2['img'][batch_idx])
        
        # Save 3D points with colors exactly like infer.py
        xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)
        xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
        np.save(os.path.join(save_dir, f'pts3d1_p{id}.npy'), xyzrgb1)
        np.save(os.path.join(save_dir, f'pts3d2_p{id}.npy'), xyzrgb2)
        
        # Save confidence maps exactly like infer.py
        conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
        conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'conf1_p{id}.npy'), conf1)
        np.save(os.path.join(save_dir, f'conf2_p{id}.npy'), conf2)
        
        # Save images exactly like infer.py
        img1 = colors1 * 255
        img2 = colors2 * 255
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir, f'img1_p{id}.png'), img1_rgb)
        cv2.imwrite(os.path.join(save_dir, f'img2_p{id}.png'), img2_rgb)
        
        id += 1

def run_robot_inference(model, data_loader, device, output_dir, args):
    """Run robot inference using DataLoader properly"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize loss criterion for evaluation
    criterion = RobotConfLoss(traj_weight=0.5, align3d_weight=5.0, pred_intrinsics=True, cotracker=True).to(device)
    
    results = []
    total_loss = 0.0
    num_samples = 0
    
    if args.data_index is not None:
        print(f"Running inference on specific data index: {args.data_index}")
        # Get specific batch directly
        batch = data_loader.dataset[args.data_index]
        # Convert to batch format and move to device
        view1, view2 = batch
        for key in view1:
            if isinstance(view1[key], torch.Tensor):
                view1[key] = view1[key].unsqueeze(0).to(device)
            elif isinstance(view1[key], np.ndarray):
                view1[key] = torch.from_numpy(view1[key]).unsqueeze(0).to(device)
        for key in view2:
            if isinstance(view2[key], torch.Tensor):
                view2[key] = view2[key].unsqueeze(0).to(device)
            elif isinstance(view2[key], np.ndarray):
                view2[key] = torch.from_numpy(view2[key]).unsqueeze(0).to(device)
        
        batch_indices = [args.data_index]
    else:
        print(f"Running inference on {len(data_loader)} batches...")
        batch_indices = list(range(min(len(data_loader), 10)))
    
    with torch.no_grad():
        if args.data_index is not None:
            # Process single batch
            batch_idx = args.data_index
            try:
                # Restructure view1 and view2 to match infer.py pattern
                # Robot dataset: views1=[img0, img1], views2=[img1, img0]
                # Target: views1=[img0, img0], views2=[img0, img1]
                
                # Get the data from robot dataset
                # views1: [img0, img1], views2: [img1, img0]
                for key in view1:
                    if isinstance(view1[key], torch.Tensor) and len(view1[key].shape) >= 2:
                        # Get img0 from views1[0] and img1 from views1[1]
                        img0_data = view1[key][0][0].clone()  # img0 from views1[0]
                        img1_data = view1[key][0][1].clone()  # img1 from views1[1]
                        
                        # Restructure: views1=[img0, img0], views2=[img0, img1]
                        view1[key][0][0] = img0_data  # img0
                        view1[key][0][1] = img0_data  # img0 (same as [0])
                        view2[key][0][0] = img0_data  # img0
                        view2[key][0][1] = img1_data  # img1
                
                # Forward pass
                pred1, pred2 = model(view1, view2)
                
                # Compute loss for evaluation
                loss, loss_details = criterion(view1, view2, pred1, pred2)
                
                # Store results
                batch_result = {
                    'batch_idx': batch_idx,
                    'loss': float(loss),
                    'loss_details': loss_details,
                    'view1': view1,
                    'view2': view2,
                    'pred1': pred1,
                    'pred2': pred2
                }
                results.append(batch_result)
                
                total_loss += float(loss)
                num_samples += 1
                
                # Save results for this batch
                batch_output_dir = os.path.join(output_dir, f'batch_{batch_idx:03d}')
                save_robot_results(view1, view2, pred1, pred2, batch_output_dir, batch_idx)
                
                # Save loss details
                with open(os.path.join(batch_output_dir, 'loss_details.txt'), 'w') as f:
                    f.write(f"Total Loss: {float(loss):.6f}\n")
                    for key, value in loss_details.items():
                        f.write(f"{key}: {value:.6f}\n")
                
                print(f"Processed batch {batch_idx}, loss: {float(loss):.6f}")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Process multiple batches
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
                if batch_idx >= 10:  # Default max 10 samples
                    break
                
                try:
                    view1, view2 = batch
                    
                    # Move tensors to device
                    for key in view1:
                        if isinstance(view1[key], torch.Tensor):
                            view1[key] = view1[key].to(device)
                    for key in view2:
                        if isinstance(view2[key], torch.Tensor):
                            view2[key] = view2[key].to(device)
                    
                    # Restructure view1 and view2 to match infer.py pattern
                    # Robot dataset: views1=[img0, img1], views2=[img1, img0]
                    # Target: views1=[img0, img0], views2=[img0, img1]
                    
                    # Get the data from robot dataset
                    # views1: [img0, img1], views2: [img1, img0]
                    for key in view1:
                        if isinstance(view1[key], torch.Tensor) and len(view1[key].shape) >= 2:
                            # Get img0 from views1[0] and img1 from views1[1]
                            img0_data = view1[key][batch_idx, 0]  # img0 from views1[0]
                            img1_data = view1[key][batch_idx, 1]  # img1 from views1[1]
                            
                            # Restructure: views1=[img0, img0], views2=[img0, img1]
                            view1[key][batch_idx, 0] = img0_data  # img0
                            view1[key][batch_idx, 1] = img0_data  # img0 (same as [0])
                            view2[key][batch_idx, 0] = img0_data  # img0
                            view2[key][batch_idx, 1] = img1_data  # img1
                    
                    # Forward pass
                    pred1, pred2 = model(view1, view2)
                    
                    # Compute loss for evaluation
                    loss, loss_details = criterion(view1, view2, pred1, pred2)
                    
                    # Store results
                    batch_result = {
                        'batch_idx': batch_idx,
                        'loss': float(loss),
                        'loss_details': loss_details,
                        'view1': view1,
                        'view2': view2,
                        'pred1': pred1,
                        'pred2': pred2
                    }
                    results.append(batch_result)
                    
                    total_loss += float(loss)
                    num_samples += 1
                    
                    # Save results for this batch
                    batch_output_dir = os.path.join(output_dir, f'batch_{batch_idx:03d}')
                    save_robot_results(view1, view2, pred1, pred2, batch_output_dir, batch_idx)
                    
                    # Save loss details
                    with open(os.path.join(batch_output_dir, 'loss_details.txt'), 'w') as f:
                        f.write(f"Total Loss: {float(loss):.6f}\n")
                        for key, value in loss_details.items():
                            f.write(f"{key}: {value:.6f}\n")
                    
                    print(f"Processed batch {batch_idx}, loss: {float(loss):.6f}")
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Save summary
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    summary = {
        'total_batches': len(data_loader),
        'processed_batches': num_samples,
        'average_loss': avg_loss,
        'total_loss': total_loss
    }
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("Robot Inference Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total batches: {summary['total_batches']}\n")
        f.write(f"Processed batches: {summary['processed_batches']}\n")
        f.write(f"Average loss: {summary['average_loss']:.6f}\n")
        f.write(f"Total loss: {summary['total_loss']:.6f}\n")
    
    print(f"\nInference completed!")
    print(f"Processed {num_samples} batches")
    print(f"Average loss: {avg_loss:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return results, summary

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("Robot Inference Script")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    if args.data_index is not None:
        print(f"Data index: {args.data_index}")
    
    # Load model
    model = load_robot_model(args.checkpoint, args.model, args.device)
    
    # Create dataset
    data_loader = create_robot_dataset(args.dataset, args.batch_size, args.num_workers)
    
    # Run inference
    results, summary = run_robot_inference(model, data_loader, args.device, args.output_dir, args)
    
    print("\nInference completed successfully!")

if __name__ == '__main__':
    main()