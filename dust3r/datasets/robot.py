import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
import torchvision.transforms as transforms
import glob
import cv2
import dust3r.datasets.utils.cropping as cropping
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
ToTensor = transforms.ToTensor()
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
from typing import Tuple
import json
import yaml

class RobotDUSt3R(BaseStereoViewDataset):
    def __init__(self, # only keyword arguments
                 dataset_location = "./robots/allegro/",
                 clip_step=1,  # Step between pairs
                 quick=False,
                 num_frames=None,
                 dset='train',
                 S=2,  # Number of frames in sequence (like other datasets)
                 strides=[1, 2, 3, 4, 5],  # strides parameter (like other datasets)
                 training_mode='pair',  # training mode (seq or pair)
                 pair_strides=[1, 2, 4],  # strides for pair mode
                 view_strides=[2, 4, 6, 8, 10],  # allowed view number differences
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        print('loading Robot dataset...')

        self.dataset_label = 'Robot'
        self.dataset_location = dataset_location
        self.clip_step = clip_step
        self.S = S  # Store the S parameter
        self.strides = strides  # Store the strides parameter
        self.training_mode = training_mode  # Store the training mode
        self.pair_strides = pair_strides
        self.view_strides = view_strides  # Store the view stride parameter
        
        # Load YAML config
        path_parts = self.dataset_location.split('/')
        robots_idx = path_parts.index('robots')
        robot_name = path_parts[robots_idx+1]
        self.config_path = 'configs/dataset/' + robot_name + '.yaml'
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use all available views for sequence-based splitting
        self.target_views = self.config['train_view'] + self.config['test_view']
        print(f"Loading all views for sequence-based splitting: {self.target_views}")
        
        # Store dataset split type for later use
        self.dset = dset

        with open(os.path.join(self.dataset_location, 'transforms.json'), "r") as f:
            self.transforms = json.load(f)
        self.cameras = self.transforms["cameras"]
        self.frames = self.transforms["frames"]

        # Initialize lists for clip-based structure (similar to PointOdyssey)
        self.frame_indices = []  # Store frame indices for each clip/pair
        self.sample_stride = []
        self.stride_counts = {}
        self.stride_idxs = {}
        
        # Initialize stride statistics for pair mode
        for stride in self.pair_strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []

        # Group frames by view and sequence
        self._load_robot_sequences()

        # Calculate stride statistics
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)

        print('stride counts:', self.stride_counts)
        print(f'collected {len(self.frame_indices)} pairs')

    def _process_joint_positions(self, joint_pos_raw):
        """
        Process joint positions based on config settings.
        
        Args:
            joint_pos_raw: Raw joint positions from frame data
            
        Returns:
            np.ndarray: Processed joint positions (only active joints)
        """
        if joint_pos_raw is None:
            return None
            
        joint_pos = np.array(joint_pos_raw, dtype=np.float32)
        
        # Check if joint positions match expected total joints
        expected_joints = self.config.get('num_total_joints', None)
        if expected_joints is not None and len(joint_pos) != expected_joints:
            print(f"Warning: Joint positions length ({len(joint_pos)}) doesn't match "
                  f"expected num_total_joints ({expected_joints})")
            return None
        
        # Get disabled joints from config
        disabled_joints = self.config.get('disabled_joints', [])
        
        if disabled_joints:
            # Create mask for active joints (not disabled)
            active_mask = np.ones(len(joint_pos), dtype=bool)
            for disabled_idx in disabled_joints:
                if 0 <= disabled_idx < len(joint_pos):
                    active_mask[disabled_idx] = False
            
            # Return only active joints
            return joint_pos[active_mask]
        else:
            # No disabled joints, return all positions
            return joint_pos

    def _load_robot_sequences(self):
        """
        Direct approach: collect all valid frame pairs and filter by constraints.
        """
        print('Loading robot sequences...')
        
        # Collect all frames with metadata
        all_frame_data = {}
        for frame_idx, frame in enumerate(tqdm(self.frames, desc="Processing frames")):
            path_parts = frame['file_path'].split('/')
            if len(path_parts) < 3:
                continue
                
            view_name = path_parts[-3]
            if not view_name.startswith('view_'):
                continue
                
            view_num = int(view_name.split('_')[1])
            if view_num not in self.target_views:
                continue
                
            filename = path_parts[-1].replace('.png', '')
            parts = filename.split('_')
            if len(parts) != 2:
                continue
                
            sequence_id = int(parts[0])
            frame_index = int(parts[1])
            
            if sequence_id not in all_frame_data.keys():
                all_frame_data[sequence_id] = []
            all_frame_data[sequence_id].append((frame_idx, frame_index, view_num))
        
        # Get train and test sequences
        sequence_ids = sorted(all_frame_data.keys())
        split_idx = int(len(sequence_ids) * 0.8)
        
        if self.dset == 'train':
            selected_sequence_ids = sequence_ids[:split_idx]
            print(f"Training: Using sequences {selected_sequence_ids[0]}-{selected_sequence_ids[-1]} ({len(selected_sequence_ids)} sequences)")
        elif self.dset == 'test':
            selected_sequence_ids = sequence_ids[split_idx:]
            print(f"Test: Using sequences {selected_sequence_ids[0]}-{selected_sequence_ids[-1]} ({len(selected_sequence_ids)} sequences)")
        else:
            selected_sequence_ids = sequence_ids
            print(f"Using all sequences: {len(selected_sequence_ids)} sequences")
        
        for sequence_id in selected_sequence_ids:
            frames = all_frame_data[sequence_id]
            for i in range(len(frames) - 1):
                for j in range(i + 1, len(frames)):
                    frame1 = frames[i]
                    frame2 = frames[j]
                    frame1_idx = frame1[0]
                    frame2_idx = frame2[0]
                    frame1_index = frame1[1]
                    frame2_index = frame2[1]
                    frame1_view = frame1[2]
                    frame2_view = frame2[2]
                    view_diff = abs(frame1_view - frame2_view)
                    frame_diff = abs(frame1_index - frame2_index)
                    if view_diff not in self.view_strides or frame_diff not in self.pair_strides:
                        continue
                    self.frame_indices.append([frame1_idx, frame2_idx])
                    self.sample_stride.append(frame_diff)

        print(f'Created {len(self.frame_indices)} cross-view pairs')
        
    def __len__(self):
        return len(self.frame_indices)

    def _get_views(self, index, resolution=(512, 288), rng=None):
        # /robots/allegro/view_0/rgb/00524_00001.png corrupted
        # Get frame indices for this pair (always 2 frames)
        frame_indices = self.frame_indices[index]  # List of 2 frame indices in self.frames
        
        B = 2  # Always 2 frames
        W, H = resolution
        supervised_label = torch.tensor(0.0, dtype=torch.float32)
        
        # Load data on-demand from frame indices
        rgb_paths = []
        depth_paths = []
        joint_positions = []
        camera_indices = []
        
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            # Construct full path by joining with dataset_location
            full_rgb_path = os.path.join(self.dataset_location, frame['file_path'])
            rgb_paths.append(full_rgb_path)
            
            # Load depth file path
            depth_file_path = frame.get('depth_file_path', None)
            if depth_file_path:
                full_depth_path = os.path.join(self.dataset_location, depth_file_path)
                depth_paths.append(full_depth_path)
            else:
                depth_paths.append(None)
            
            # Process joint positions
            joint_pos_raw = frame.get('joint_pos', None)
            joint_pos_processed = self._process_joint_positions(joint_pos_raw)
            joint_positions.append(joint_pos_processed)
            
            # Get camera index for this frame
            camera_idx = frame.get('camera_idx', 0)
            camera_indices.append(camera_idx)

        # Calculate joint positions tensor size
        joint_pos_size = 0
        if joint_positions[0] is not None:
            joint_pos_size = len(joint_positions[0])
        
        # Load camera intrinsics and extrinsics for each frame
        camera_intrinsics_list = []
        camera_extrinsics_list = []
        
        for i, camera_idx in enumerate(camera_indices):
            # Get camera parameters from the cameras array
            camera = self.cameras[camera_idx]
            
            # Extract intrinsic parameters
            fl_x = camera['fl_x']
            fl_y = camera['fl_y'] 
            cx = camera['cx']
            cy = camera['cy']
            orig_h = camera['h']
            orig_w = camera['w']
            
            # Scale intrinsics to target resolution
            # Original resolution: (orig_w, orig_h), Target resolution: (W, H)
            scale_x = W / orig_w
            scale_y = H / orig_h
            
            # Create camera intrinsics matrix with proper scaling
            # Focal lengths and principal points need to be scaled by the resolution change
            # This maintains the same field of view and optical center relative to the new image size
            camera_intrinsics = torch.tensor([
                [fl_x * scale_x, 0.0, cx * scale_x],  # [fx, 0, cx]
                [0.0, fl_y * scale_y, cy * scale_y],  # [0, fy, cy]
                [0.0, 0.0, 1.0]                       # [0, 0, 1]
            ], dtype=torch.float32)
            camera_intrinsics_list.append(camera_intrinsics)
            
            # Extract camera extrinsics (transform matrix)
            transform_matrix = torch.tensor(camera['transform_matrix'], dtype=torch.float32)
            camera_extrinsics_list.append(transform_matrix)
        
        # Stack camera intrinsics and extrinsics
        camera_intrinsics_tensor = torch.stack(camera_intrinsics_list)  # Shape: (B, 3, 3)
        camera_extrinsics_tensor = torch.stack(camera_extrinsics_list)  # Shape: (B, 4, 4)
        
        # views1: Bidirectional pairs [view0, view1]
        views1 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'depth'            : torch.zeros((B, H, W), dtype=torch.float32),
            'joint_pos'        : torch.zeros((B, joint_pos_size), dtype=torch.float32) if joint_pos_size > 0 else None,
            'dataset'          : self.dataset_label,
            'supervised_label' : torch.zeros(B, dtype=torch.float32),
            'file_path'        : [''] * B,  # Add file paths for debugging
            'camera_intrinsics': camera_intrinsics_tensor,  # Shape: (B, 3, 3)
            'camera_pose': camera_extrinsics_tensor,  # Shape: (B, 4, 4)
        }
        
        # views2: Bidirectional pairs [view1, view0] 
        views2 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'depth'            : torch.zeros((B, H, W), dtype=torch.float32),
            'joint_pos'        : torch.zeros((B, joint_pos_size), dtype=torch.float32) if joint_pos_size > 0 else None,
            'dataset'          : self.dataset_label,
            'supervised_label' : torch.zeros(B, dtype=torch.float32),
            'file_path'        : [''] * B,  # Add file paths for debugging
            'camera_intrinsics': camera_intrinsics_tensor,  # Shape: (B, 3, 3)
            'camera_pose': camera_extrinsics_tensor,  # Shape: (B, 4, 4)
        }

        # Process each frame in the pair (only 2 frames)
        for i in range(B):
            rgb_path_frame = rgb_paths[i]
            depth_path_frame = depth_paths[i]
            
            # Load RGB image
            try:
                rgb_image = imread_cv2(rgb_path_frame)
            except Exception as e:
                print(f"Warning: Failed to load corrupted image {rgb_path_frame}: {e}")
                # Try to load a different sample by recursively calling with a different index
                # This will skip the corrupted sample and try the next one
                import random
                alternative_idx = random.randint(0, len(self.frame_indices) - 1)
                print(f"Trying alternative sample at index {alternative_idx}")
                return self._get_views(alternative_idx, resolution, rng)
            
            # Load depth image if available
            depth_image = None
            if depth_path_frame is not None:
                try:
                    depth_image = cv2.imread(depth_path_frame, cv2.IMREAD_UNCHANGED)
                    if depth_image is not None:
                        # Resize depth to target resolution
                        depth_image = cv2.resize(depth_image, (W, H), interpolation=cv2.INTER_NEAREST)
                        # Convert to float32 and normalize if needed
                        if depth_image.dtype == np.uint16:
                            depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
                        else:
                            depth_image = depth_image.astype(np.float32)
                    else:
                        print(f"Warning: Failed to load depth image {depth_path_frame}")
                except Exception as e:
                    print(f"Warning: Failed to load depth image {depth_path_frame}: {e}")
            
            # Resize RGB image to target resolution
            rgb_image = cv2.resize(rgb_image, (W, H))
            
            # Store frame data
            if i == 0:
                rgb_image_0 = rgb_image.copy()
                depth_image_0 = depth_image
                joint_pos_0 = joint_positions[0]
            elif i == 1:
                rgb_image_1 = rgb_image.copy()
                depth_image_1 = depth_image
                joint_pos_1 = joint_positions[1]
        
        # Fill views1 with bidirectional pairs: [view0, view1]
        views1['img'][0] = self.transform(rgb_image_0)  # First entry: view0
        views1['img'][1] = self.transform(rgb_image_1)  # Second entry: view1
        views1['supervised_label'][0] = supervised_label
        views1['supervised_label'][1] = supervised_label
        views1['file_path'][0] = rgb_paths[0]  # First entry: view0 path (full path)
        views1['file_path'][1] = rgb_paths[1]  # Second entry: view1 path (full path)
        if views1['joint_pos'] is not None and joint_pos_0 is not None:
            views1['joint_pos'][0] = torch.from_numpy(joint_pos_0)
        if views1['joint_pos'] is not None and joint_pos_1 is not None:
            views1['joint_pos'][1] = torch.from_numpy(joint_pos_1)
        
        # Add depth data to views1
        if depth_image_0 is not None:
            views1['depth'][0] = torch.from_numpy(depth_image_0)
        if depth_image_1 is not None:
            views1['depth'][1] = torch.from_numpy(depth_image_1)
        
        # Fill views2 with bidirectional pairs: [view1, view0] 
        views2['img'][0] = self.transform(rgb_image_1)  # First entry: view1
        views2['img'][1] = self.transform(rgb_image_0)  # Second entry: view0
        views2['supervised_label'][0] = supervised_label
        views2['supervised_label'][1] = supervised_label
        views2['file_path'][0] = rgb_paths[1]  # First entry: view1 path (full path)
        views2['file_path'][1] = rgb_paths[0]  # Second entry: view0 path (full path)
        if views2['joint_pos'] is not None and joint_pos_1 is not None:
            views2['joint_pos'][0] = torch.from_numpy(joint_pos_1)
        if views2['joint_pos'] is not None and joint_pos_0 is not None:
            views2['joint_pos'][1] = torch.from_numpy(joint_pos_0)
        
        # Add depth data to views2 (swapped order)
        if depth_image_1 is not None:
            views2['depth'][0] = torch.from_numpy(depth_image_1)
        if depth_image_0 is not None:
            views2['depth'][1] = torch.from_numpy(depth_image_0)
        # print(f'In robot dataset, views1 image shape: {views1["img"].shape}')
        # print(f'In robot dataset, views2 image shape: {views2["img"].shape}')

        return [views1, views2]


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import gradio as gr
    import random

    dataset = RobotDUSt3R(quick=False, resolution=[(512,288)])
    print(len(dataset), 'sequences')
    # for idx in range(700):
    #     print(dataset[idx][0]['file_path'])
    # def visualize(idx, frame=0):
    #     views = dataset[idx]
    #     viz = SceneViz()
    #     views1, views2 = views
    #     pts3d = views2['pts3d_moge'][frame]
    #     valid_mask = views2['valid_mask'][frame]
    #     colors = rgb(views2['img'][frame])
    #     viz.add_pointcloud(pts3d, colors, valid_mask)
    #     os.makedirs('./tmp/custom', exist_ok=True)
    #     path = f'./tmp/custom/{idx}_f{frame}.glb'

    #     # visualize the img
    #     img_save_path = f'./tmp/custom/{idx}_f{frame}.png'
    #     img = views2['img_org'][frame].permute(1,2,0).numpy() * 255
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(img_save_path, img)

    #     # visualize the depthmap
    #     depth_save_path = f'./tmp/custom/{idx}_f{frame}_depth.png'
    #     depth = views2['depthmap'][frame].numpy()
    #     depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    #     depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    #     cv2.imwrite(depth_save_path, depth)

    #     return viz.save_glb(path)

    # for idx in range(1):
    #     visualize(idx, frame=1)
    #     print(f'visualized {idx}')
    
