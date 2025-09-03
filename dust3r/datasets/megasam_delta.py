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
import pickle
import dust3r.datasets.utils.cropping as cropping
from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import torch.nn.functional as F
ToTensor = transforms.ToTensor()
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
from typing import Tuple
import matplotlib.pyplot as plt

def depthmap_gradient_filter(depthmap):

    # input: np.ndarray, shape: (B, H, W, 1)
    # output: np.ndarray, shape: (B, H, W, 1)

    # compute the gradient of the depthmap
    grad_x = np.gradient(depthmap, axis=2)
    grad_y = np.gradient(depthmap, axis=1)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    threshold = 1
    grad_magnitude[grad_magnitude > threshold] = 0
    depthmap[grad_magnitude == 0] = -1
    return depthmap

class MegasamDeltaDUSt3R(BaseStereoViewDataset):
    def __init__(self, # only keyword arguments
                 dataset_location = ["/is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth/*/", 
                                    "/is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth_mevis_lasot3k/*/",
                                    "/is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth_openvid30k_vimeo15k_vmimik5k/*/",
                                    "/is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth_vmimik30k_vos3k/*/"],  
                 S_min=16,
                 S=32,
                 strides=[2, 1],
                 quick=False,
                 reject_sampling=False,
                 accumulate_motion_masks=False,
                 depth_filter=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        print('loading Megasam-Delta dataset...')

        self.dataset_label = 'Megasam-Delta'
        self.dataset_location = dataset_location
        self.S = S
        self.S_min = S_min
        self.strides_all = strides  # Added strides for multiple iterations
        self.reject_sampling = reject_sampling
        self.accumulate_motion_masks = accumulate_motion_masks
        self.depth_filter = depth_filter
        self.sequences = []
        for dataset_location in self.dataset_location:
            self.sequences.extend(sorted(glob.glob(dataset_location)))

        if quick:
            self.sequences = self.sequences[22000:23000]

        print(f"Found {len(self.sequences)} sequences") # Found 71078 sequences

        self.megasam_paths = []     # /is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth/Animal-Kingdom_action_AADBUVKA/sgd_cvd_hr.npz
        self.delta_paths = []       # /is/cluster/fast/groups/ps-pbrgaussians/megasam/vis_depth/Animal-Kingdom_action_AADBUVKA/delta_results.pkl
        self.org_fps = []
        for seq in tqdm(self.sequences):
            megasam_path = os.path.join(seq, "sgd_cvd_hr.npz")
            delta_path = os.path.join(seq, "delta_results.pkl")
            if not os.path.exists(megasam_path) or not os.path.exists(delta_path):
                continue
            self.megasam_paths.append(megasam_path)
            self.delta_paths.append(delta_path)
            if 'lasot' in seq:
                self.org_fps.append(8)
            else:
                self.org_fps.append(30)

        # Apply 50% dropout for first 6000 sequences
        if len(self.megasam_paths) > 0:

            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            dropout_mask = np.ones(len(self.megasam_paths), dtype=bool)
            n_dropout = min(6000, len(self.megasam_paths))          # drop out for animal kindom
            dropout_mask[:n_dropout] = rng.random(n_dropout) > 0.9
            
            self.megasam_paths = [p for i, p in enumerate(self.megasam_paths) if dropout_mask[i]]
            self.delta_paths = [p for i, p in enumerate(self.delta_paths) if dropout_mask[i]]

        self.idx_ssl = len(self.megasam_paths) - 1 # dummy
        print(f"Finally {len(self.megasam_paths)} sequences after dropout")

    def __len__(self):
        return len(self.megasam_paths) * len(self.strides_all)
    
    def _get_views(self, index, resolution=(512, 288), rng=None):
        # Calculate which stride and sequence to use
        stride_idx = index // len(self.megasam_paths)
        seq_idx = index % len(self.megasam_paths)
        current_stride = self.strides_all[stride_idx]

        megasam_path = self.megasam_paths[seq_idx]
        delta_path = self.delta_paths[seq_idx]
        megasam_data = np.load(megasam_path)
        delta_data = pickle.load(open(delta_path, "rb"))

        images = megasam_data['images']
        num_frames = images.shape[0]

        # set B to be the num_frames.clip(self.S_min, self.S)
        B = max(self.S_min, min(num_frames, self.S))
        W, H = resolution
        stride = max(1, min(current_stride, num_frames // B))

        images = images[:B*stride:stride]
        depths = megasam_data['depths'][:B*stride:stride]
        intrinsics_ori = megasam_data['intrinsic']
        cam_c2w = megasam_data['cam_c2w'][:B*stride:stride]
        motion_probs = megasam_data['motion_pro'][:B*stride:stride]

        trajs_uv = delta_data['trajs_uv'][0][:B*stride:stride]
        trajs_depth = delta_data['trajs_depth'][0][:B*stride:stride]
        conf_trajs = delta_data['conf'][0][:B*stride:stride]

        if len(images) < B:
            images = np.pad(images, ((0, B-len(images)), (0, 0), (0, 0), (0, 0)), mode='edge')
            depths = np.pad(depths, ((0, B-len(depths)), (0, 0), (0, 0)), mode='edge')
            cam_c2w = np.pad(cam_c2w, ((0, B-len(cam_c2w)), (0, 0), (0, 0)), mode='edge')
            motion_probs = np.pad(motion_probs, ((0, B-len(motion_probs)), (0, 0), (0, 0)), mode='edge')
            trajs_uv = np.pad(trajs_uv, ((0, B-len(trajs_uv)), (0, 0), (0, 0)), mode='edge')
            trajs_depth = np.pad(trajs_depth, ((0, B-len(trajs_depth)), (0, 0), (0, 0)), mode='edge')
            conf_trajs = np.pad(conf_trajs, ((0, B-len(conf_trajs)), (0, 0)), mode='edge')

        if self.depth_filter:
            depths = depthmap_gradient_filter(depths)

        W0, H0 = images[0].shape[1], images[0].shape[0]

        trajs_uv = torch.from_numpy(trajs_uv).reshape(B, 384, 512, 2)
        trajs_depth = torch.from_numpy(trajs_depth).reshape(B, 384, 512, 1)
        conf_trajs = torch.from_numpy(conf_trajs).reshape(B, 384, 512)
        motion_probs = torch.from_numpy(motion_probs)

        trajs_uv = F.interpolate(trajs_uv.permute(0, 3, 1, 2), size=(H0, W0), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        trajs_depth = F.interpolate(trajs_depth.permute(0, 3, 1, 2), size=(H0, W0), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)   
        conf_trajs = F.interpolate(conf_trajs.unsqueeze(1), size=(H0, W0), mode='bilinear', align_corners=False).squeeze(1)
        motion_probs = F.interpolate(motion_probs.unsqueeze(1), size=(H0, W0), mode='bilinear', align_corners=False).squeeze(1)

        views1 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'depthmap'         : torch.zeros((B, H, W),   dtype=torch.float32),
            'traj_ptc'         : torch.zeros((B, H, W, 3),dtype=torch.float32),
            'traj_mask'        : torch.ones((B, H, W),   dtype=torch.bool),
            'camera_pose'      : torch.zeros((B, 4, 4),   dtype=torch.float32),
            'camera_intrinsics': torch.zeros((B, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * B,
            'instance'         : [''] * B,
            'supervised_label' : torch.ones(B, dtype=torch.float32),
            'pts3d'            : torch.zeros((B, H, W, 3), dtype=torch.float32),
            'valid_mask'       : torch.zeros((B, H, W),    dtype=torch.bool),
            'motion_mask'      : torch.zeros((B, H, W),    dtype=torch.float32),
        }

        views2 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'img_org'          : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'depthmap'         : torch.zeros((B, H, W),   dtype=torch.float32),
            'camera_pose'      : torch.zeros((B, 4, 4),   dtype=torch.float32),
            'camera_intrinsics': torch.zeros((B, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * B,
            'instance'         : [''] * B,
            'supervised_label' : torch.ones(B, dtype=torch.float32),
            'pts3d'            : torch.zeros((B, H, W, 3), dtype=torch.float32),
            'valid_mask'       : torch.zeros((B, H, W),    dtype=torch.bool),
        }

        # Pre-allocate and convert tensors once before the loop
        # depth_maps_tensor = torch.from_numpy(depths)
        cam_c2w_tensor = torch.from_numpy(cam_c2w)
        
        # Pre-compute transformations for all images at once
        # images_tensor = torch.stack([self.transform(img) for img in images])
        # images_org_tensor = torch.stack([ToTensor(img) for img in images])

        for i in range(B):
            depth_map = depths[i]
            rgb_image = images[i]
            intrinsics = intrinsics_ori


            rgb_image, depth_map, intrinsics, (l, t, r, b) = cropping.rescale_image_depthmap_crop(
                rgb_image, depth_map, intrinsics, resolution)


            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depth_map, intrinsics, camera_pose=cam_c2w[i], proj_mode='depth'
            )

            trajs_uv_single = trajs_uv[i][t:b, l:r]
            trajs_depth_single = trajs_depth[i][t:b, l:r]
            trajs_uv_single[..., 0] = trajs_uv_single[..., 0] - l
            trajs_uv_single[..., 1] = trajs_uv_single[..., 1] - t

            traj_3d_proj = torch.cat((trajs_uv_single, trajs_depth_single), dim=-1)

            pts3d_traj, _ = depthmap_to_absolute_camera_coordinates(
                traj_3d_proj.reshape(-1, 3).numpy(), intrinsics, camera_pose=cam_c2w[i], proj_mode='ptc')

            # pts3d_traj is of shape (N, 3) where N should equal H*W
            total_size = pts3d_traj.shape[0]
            expected_size = H * W
            assert total_size == expected_size, f"Size mismatch in {megasam_path}: expected {expected_size} points (H={H} * W={W}), got {total_size} points, traj_3d_proj.shape: {traj_3d_proj.shape}, \
                trajs_uv_single.shape: {trajs_uv_single.shape}, trajs_depth_single.shape: {trajs_depth_single.shape}, t is {t}, l is {l}, r is {r}, b is {b}, rgb_image.size: {rgb_image.size}, \
                images.shape: {images.shape}, depths.shape: {depths.shape}, cam_c2w.shape: {cam_c2w.shape}, motion_probs.shape: {motion_probs.shape}, trajs_uv.shape: {trajs_uv.shape}, trajs_depth.shape: {trajs_depth.shape}, conf_trajs.shape: {conf_trajs.shape}"
            
            # Reshape the points to (H, W, 3)
            pts3d_traj = torch.from_numpy(pts3d_traj).reshape(-1, 3)  # Ensure it's (N, 3)
            pts3d_traj = pts3d_traj.reshape(H, W, 3)

            if i == 0:
                rgb_image_0 = rgb_image.copy()
                depth_map_0 = depth_map.copy()
                intrinsics_0 = torch.from_numpy(intrinsics).clone()
                pts3d_0 = torch.from_numpy(pts3d).clone()
                valid_mask_0 = torch.from_numpy(valid_mask).clone()
                if self.accumulate_motion_masks:
                    # Initialize motion mask and votes tensors
                    motion_mask_0 = torch.zeros((H0, W0), dtype=torch.float32)
                    votes = torch.zeros((H0, W0), dtype=torch.float32)
                    ones = torch.ones((H0, W0), dtype=torch.float32)
                    
                    for j in range(B):
                        # Reshape grid for batch dimension [1, H, W, 2]
                        grid = ((trajs_uv[j] / torch.tensor([W0, H0])) - 0.5) * 2
                        grid = grid.unsqueeze(0)  # Add batch dimension
                        
                        # Add batch dimension to ones [1, 1, H, W]
                        ones_batch = ones.unsqueeze(0).unsqueeze(0)
                        
                        # Add batch and channel dimensions to motion_probs [1, 1, H, W]
                        motion_probs_batch = motion_probs[j].unsqueeze(0).unsqueeze(0)
                        
                        # Update votes and motion_mask_0
                        votes += F.grid_sample(ones_batch, grid, mode='nearest', align_corners=False).squeeze(0).squeeze(0)
                        motion_mask_0 += F.grid_sample(motion_probs_batch, grid, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    
                    # Avoid division by zero and normalize the motion mask
                    motion_mask_0 = torch.where(votes > 0, motion_mask_0 / votes, torch.zeros_like(motion_mask_0))
                    
                    # Set where motion_mask_0 is 0 to the min of the non-zero values in motion_mask_0
                    min_value = motion_mask_0[motion_mask_0 > 0].min()
                    motion_mask_0 = torch.where(motion_mask_0 == 0, min_value, motion_mask_0).clip(0, 1)
                    # Convert to boolean if needed
                    # motion_mask_0_bool = motion_mask_0 > torch.mean(motion_mask_0)
                else:
                    motion_mask_0 = motion_probs[0].clone()
                    # motion_mask_0_bool = motion_mask_0 > torch.mean(motion_mask_0)
    
                motion_mask_0 = motion_mask_0[t:b, l:r]
                pts3d_traj_0 = pts3d_traj.clone()
            
            if self.reject_sampling:
                # set where motion_mask_0 is more than mean, the pts3d_traj to be pts3d_traj_0
                # motion_mask_0 > torch.mean(motion_mask_0), means static regions
                pts3d_traj[motion_mask_0 > torch.mean(motion_mask_0)] = pts3d_traj_0[motion_mask_0 > torch.mean(motion_mask_0)]
            

            views1['motion_mask'][i] = motion_mask_0
            views1['img'][i] = self.transform(rgb_image_0)
            views1['depthmap'][i] = torch.from_numpy(depth_map_0)
            views1['traj_ptc'][i] = pts3d_traj
            views1['traj_mask'][i] = torch.ones((H, W), dtype=torch.bool) & valid_mask_0
            views1['camera_pose'][i] = cam_c2w_tensor[0]
            views1['camera_intrinsics'][i] = intrinsics_0
            views1['label'][i] = megasam_path.split('/')[-2]
            views1['instance'][i] = f'frame{0}'
            views1['supervised_label'][i] = torch.ones(1, dtype=torch.float32)
            views1['pts3d'][i] = pts3d_0
            views1['valid_mask'][i] = valid_mask_0

            views2['img'][i] = self.transform(rgb_image)
            views2['img_org'][i] = ToTensor(rgb_image)
            views2['depthmap'][i] = torch.from_numpy(depth_map)
            views2['camera_pose'][i] = cam_c2w_tensor[i]
            views2['camera_intrinsics'][i] = torch.from_numpy(intrinsics)
            views2['label'][i] = megasam_path.split('/')[-2]
            views2['instance'][i] = f'frame{i*stride}'
            views2['supervised_label'][i] = torch.ones(1, dtype=torch.float32)
            views2['pts3d'][i] = torch.from_numpy(pts3d)
            views2['valid_mask'][i] = torch.from_numpy(valid_mask)

        return [views1, views2]

if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import random
    import time
    
    dataset = MegasamDeltaDUSt3R(S=24,
                                strides=[2,1],
                                resolution=[(512,288)],
                                depth_filter=False,
                                accumulate_motion_masks=True,
                                reject_sampling=True)
    print(len(dataset), 'sequences')
    
    def visualize(idx, frame=0):
        # count the time
        start_time = time.time()
        views = dataset[idx]
        end_time = time.time()
        print(f'time: {end_time - start_time}s')
        
        # Initialize the visualizer and get data
        viz = SceneViz()
        views1, views2 = views
        pts3d = views2['pts3d'][frame]
        valid_mask = views2['valid_mask'][frame]
        colors = rgb(views2['img'][frame])
        colors = np.clip(colors, 0, 1)
        viz.add_pointcloud(pts3d, colors, valid_mask)
        
        # Ensure output directory exists
        os.makedirs('./tmp_megasam_delta_ptc', exist_ok=True)
        
        # Define trajectory offsets and generate rainbow colors for them
        traj_offsets = [4, 8, 12, 16, 20, 23]
        cmap = plt.get_cmap('rainbow')
        # Evenly space colors over the number of trajectories
        rainbow_colors = [tuple(int(255 * c) for c in cmap(i / (len(traj_offsets)-1))[:3])
                        for i in range(len(traj_offsets))]
        
        # Loop over each trajectory offset:
        for offset, color in zip(traj_offsets, rainbow_colors):
            if frame + offset < len(views2['pts3d']):
                traj = views2['pts3d'][frame + offset]
                viz.add_pointcloud(traj, color, valid_mask)
                
                # Save corresponding RGB frame for this trajectory offset
                rgb_save_path = f'./tmp_megasam_delta_ptc/{idx}_f{frame+offset}.png'
                rgb_img = views2['img_org'][frame + offset].permute(1, 2, 0).numpy() * 255
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_save_path, rgb_img)
        
        # Save the valid mask
        valid_mask_save_path = f'./tmp_megasam_delta_ptc/{idx}_f{frame}_valid_mask.png'
        valid_mask_img = valid_mask.numpy() * 255
        valid_mask_img = cv2.cvtColor(valid_mask_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(valid_mask_save_path, valid_mask_img)
        
        # Save the motion mask if available
        if 'motion_mask' in views1:
            motion_mask_save_path = f'./tmp_megasam_delta_ptc/{idx}_f{frame}_motion_mask.png'
            motion_mask = views1['motion_mask'][frame].numpy()
            motion_mask = (motion_mask - motion_mask.min()) / (motion_mask.max() - motion_mask.min()) * 255
            motion_mask = cv2.cvtColor(motion_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(motion_mask_save_path, motion_mask)

        # Save the main frame image
        img_save_path = f'./tmp_megasam_delta_ptc/{idx}_f{frame}.png'
        img = views2['img_org'][frame].permute(1, 2, 0).numpy() * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_save_path, img)
        
        # Save the depthmap
        depth_save_path = f'./tmp_megasam_delta_ptc/{idx}_f{frame}_depth.png'
        depth = views2['depthmap'][frame].numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(depth_save_path, depth)
        
        # Save the scene visualization file
        path = f'./tmp_megasam_delta_ptc/{idx}_f{frame}.glb'
        viz.save_glb(path)
        
        # Print the label/sequence name
        label = views2['label'][frame]
        print(f"Sequence: {label}")

    # Set random seed for reproducibility
    random.seed(125)
    # Sample random indices for visualization
    random_samples = random.sample(range(len(dataset)), 100)[20:]
    
    # Visualize each sampled index
    for idx in tqdm(random_samples):
        print(f'visualizing {idx}')
        visualize(idx, frame=0)
        