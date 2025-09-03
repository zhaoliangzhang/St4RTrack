import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, GaussianBlur
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import zipfile
import glob
import cv2


import dust3r.datasets.utils.cropping as cropping
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution
ToTensor = transforms.ToTensor()
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

def reprojection(points, K, RT, eps=1e-8):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / (XYZ[:, 2:]+eps)
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z

def inverse_projection(depth, K, RT):
    h, w = depth.shape
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    uv_homogeneous = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))

    K_inv = np.linalg.inv(K)

    # use max depth as 10m for visualization
    depth = depth.flatten()
    mask = depth < 10

    XYZ = K_inv @ uv_homogeneous * depth

    XYZ = np.vstack((XYZ, np.ones(XYZ.shape[1])))
    world_coordinates = np.linalg.inv(RT) @ XYZ
    world_coordinates = world_coordinates[:3, :].T
    world_coordinates = world_coordinates[mask]

    return world_coordinates

class PointOdysseyDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/pointodyssey',
                 dset='train',
                 use_augs=False,
                 S=16,
                 N=16,
                 strides=[2,3,4],
                 clip_step=32,
                 verbose=False,
                 clip_step_last_skip = 0,
                 skip_fog = False,
                 training_mode='seq', # or pair
                 po_fog_list_path = None,
                 *args, 
                 **kwargs
                 ):

        print('loading pointodyssey dataset...')
        assert training_mode in ['seq', 'pair'], 'training_mode must be either seq or pair'

        super().__init__(*args, **kwargs)
        self.training_mode = training_mode
        self.dataset_label = 'pointodyssey'
        self.split = dset
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose

        self.use_augs = use_augs
        self.dset = dset

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.trajs_3d_data = []      
        self.traj_visibs_data = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))
        
        
        if po_fog_list_path is None:
            po_fog_list_path = "./data/po_fog_list.txt"
        po_fog_list = []
        with open(po_fog_list_path, 'r') as f:
            for line in f:
                po_fog_list.append(line.strip())

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-2]
                if not seq_name.startswith('ani') and not seq_name.startswith('char') and not seq_name.startswith('r') and (not skip_fog or (not seq_name in po_fog_list)): # if skip_fog, skip foggy sequences
                    self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))

        ## load trajectories
        print('loading trajectories...')
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')
            
            if os.path.isfile(info_path) and os.path.isfile(annotations_path):
                try:
                    # If BadZipFile occurs here (e.g., CRC error), it will be caught by the except below
                    annotations = np.load(annotations_path, allow_pickle=True)
                    trajs_3d = annotations['trajs_3d']
                    traj_visibs = annotations['visibs']
                    del annotations
                except zipfile.BadZipFile:
                    print(f'[{seq}] anno.npz file is corrupted (BadZipFile), skipping this sequence.')
                    continue
                except Exception as e:
                    if self.verbose:
                        print(f'[{seq}] unknown error when reading anno.npz: {e}, skipping this sequence.')
                    continue

                trajs_3d_shape = trajs_3d.shape

                if len(trajs_3d_shape) and trajs_3d_shape[1] > self.N:
                    for stride in strides:
                        for ii in range(0, len([f for f in os.listdir(rgb_path) if f.endswith('.jpg')]) - self.S*max(stride, clip_step_last_skip) + 1, clip_step):
                            full_idx = ii + np.arange(self.S)*stride # sampled frame idx in each clip
                            self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                            self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                            self.normal_paths.append([os.path.join(seq, 'normals', 'normal_%05d.jpg' % idx) for idx in full_idx])
                            self.trajs_3d_data.append(trajs_3d[full_idx])
                            self.traj_visibs_data.append(traj_visibs[full_idx])
                            self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                            self.full_idxs.append(full_idx)
                            self.sample_stride.append(stride)
                        if self.verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                elif self.verbose:
                    print('rejecting seq for missing 3d')
            elif self.verbose:
                print('rejecting seq for missing info or anno')
        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        
        print('stride counts:', self.stride_counts)

        for stride, count in self.stride_counts.items():
            assert count>0, f"dataset PointOdyssey stride {stride} has no clips"
        
        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def visualize_traj(self, rgb_image, traj_3d_proj, i, H, W):

        # visualize whether the traj_3d_proj align with the rgb_image
        # Create a visualization image by copying the RGB image
        vis_img = np.array(rgb_image).copy()
        
        # Get normalized coordinates based on image dimensions
        points = traj_3d_proj.numpy()
        x_norm = points[:, 0] / W  # Normalize by image width
        y_norm = points[:, 1] / H  # Normalize by image height
        
        # Convert normalized coordinates to rainbow colors
        if i==0:
            colors = np.zeros((len(points), 3))
            colors[:, 0] = 255 * (1 - x_norm)  # Red varies with x
            colors[:, 1] = 255 * y_norm        # Green varies with y
            colors[:, 2] = 255 * x_norm        # Blue varies with x
            self.traj_color = colors
        else:
            colors = self.traj_color
        
        # Draw points with their corresponding colors
        for point, color in zip(points, colors):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(vis_img, (x, y), radius=2, 
                            color=(int(color[0]), int(color[1]), int(color[2])), 
                            thickness=-1)
        
        # Save visualization to file
        os.makedirs('./test_vis', exist_ok=True)
        cv2.imwrite(f'./test_vis/frame_{i}_traj.png', vis_img)

    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[np.argmax(dist)]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()
        
        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.normal_paths = [self.normal_paths[i] for i in resampled_idxs]
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths) # num of clips
    
    def _get_views(self, index, resolution, rng):

        # -- 1) get paths and annotations --
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        traj_3d_seq = self.trajs_3d_data[index]
        traj_visibs_seq = self.traj_visibs_data[index]

        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        pix_T_cams = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['extrinsics'][full_idx].astype(np.float32)

        # -- 2) supervised for training datasets
        supervised_label = torch.tensor(1.0, dtype=torch.float32)
        
        # -- 3) prepare views1 and views2 --
        B = self.S
        if self.training_mode == 'seq':
            tensor_size = B
        elif self.training_mode == 'pair':
            tensor_size = B-1 # should be 1
        else:
            raise ValueError(f'training_mode must be either seq or pair, got {self.training_mode}')
        
        W, H = resolution

        # views1: reference frame (0th frame) repeated content, and its trajectory related fields
        views1 = {
            'img'              : torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'depthmap'         : torch.zeros((tensor_size, H, W),   dtype=torch.float32),
            'traj_ptc'         : torch.zeros((tensor_size, H, W, 3),dtype=torch.float32),
            'traj_mask'        : torch.zeros((tensor_size, H, W),   dtype=torch.bool),
            'camera_pose'      : torch.zeros((tensor_size, 4, 4),   dtype=torch.float32),
            'camera_intrinsics': torch.zeros((tensor_size, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * tensor_size,
            'instance'         : [''] * tensor_size,
            'supervised_label' : torch.ones(tensor_size, dtype=torch.float32),
            'pts3d'            : torch.zeros((tensor_size, H, W, 3), dtype=torch.float32),
            'valid_mask'       : torch.zeros((tensor_size, H, W),    dtype=torch.bool),
        }

        # views2: own real data of each frame, without traj_ptc / traj_mask (can be kept/deleted according to needs)
        views2 = {
            'img'              : torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'img_org'          : torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'depthmap'         : torch.zeros((tensor_size, H, W),   dtype=torch.float32),
            'camera_pose'      : torch.zeros((tensor_size, 4, 4),   dtype=torch.float32),
            'camera_intrinsics': torch.zeros((tensor_size, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * tensor_size,
            'instance'         : [''] * tensor_size,
            'supervised_label' : torch.ones(tensor_size, dtype=torch.float32),
            'pts3d'            : torch.zeros((tensor_size, H, W, 3), dtype=torch.float32),
            'valid_mask'       : torch.zeros((tensor_size, H, W),    dtype=torch.bool),
        }

        # -- 4) loop over B frames, read data, project, process depth, construct traj, etc. --
        for i in range(B):
            impath      = rgb_paths[i]
            depthpath   = depth_paths[i]
            extrinsics  = cams_T_world[i]   
            intrinsics  = pix_T_cams[i]     
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3] 
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R.T
            camera_pose[:3, 3] = -R.T @ t

            # read image and depth
            rgb_image = imread_cv2(impath)
            depth16   = cv2.imread(depthpath, cv2.IMREAD_ANYDEPTH)
            depthmap  = depth16.astype(np.float32) / 65535.0 * 1000.0  # according to your dataset rule

            # get visible trajectory points
            traj_3d_proj = traj_3d_seq[i]     # (N,3) 3D points
            # project to pixel coordinates
            uv, Z = reprojection(traj_3d_proj, intrinsics, extrinsics)
            traj_3d_proj = np.concatenate((uv, Z), axis=1)  # shape [N,3]
            traj_3d_proj = torch.from_numpy(traj_3d_proj)

            # filter out points out of range
            H_img, W_img = rgb_image.shape[:2]
            if i == 0:
                # only decide the final visible mask at i=0
                f0_traj_vis  = traj_visibs_seq[i]  # shape [N]
                valid_mask_traj = (
                    (traj_3d_proj[:, 0] >= 0) & (traj_3d_proj[:, 0] < W_img) &
                    (traj_3d_proj[:, 1] >= 0) & (traj_3d_proj[:, 1] < H_img)
                )
                f0_traj_vis = valid_mask_traj & f0_traj_vis
                f0_traj_vis = f0_traj_vis.to(torch.bool)
            traj_3d_proj = traj_3d_proj[f0_traj_vis]

            # --- resize / rescale ---
            rgb_image, depthmap, intrinsics = cropping.rescale_image_depthmap(
                rgb_image, depthmap, intrinsics, resolution
            )
            traj_3d_proj[..., 0] = traj_3d_proj[..., 0] * resolution[0] / float(W_img)
            traj_3d_proj[..., 1] = traj_3d_proj[..., 1] * resolution[1] / float(H_img)
            pts3d_traj, _ = depthmap_to_absolute_camera_coordinates(
                traj_3d_proj.numpy(), intrinsics, camera_pose, proj_mode='ptc'
            )

            # construct a (H,W,3) empty tensor, put the trajectory points into it
            W_new, H_new = resolution
            traj_mask_with_pts = torch.zeros((H_new, W_new, 3), dtype=torch.float32)
            if i == 0:
                traj_mask = torch.zeros((H_new, W_new), dtype=torch.bool)
                indices = traj_3d_proj[:, :2].round().long()
                indices[:, 0] = indices[:, 0].clamp(0, W_new - 1)
                indices[:, 1] = indices[:, 1].clamp(0, H_new - 1)
                traj_mask[indices[:, 1], indices[:, 0]] = True
            traj_mask_with_pts[indices[:, 1], indices[:, 0]] = torch.from_numpy(pts3d_traj)

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap, intrinsics, camera_pose, proj_mode='depth'
            )
            # filter out inf / nan
            valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)

            if i == 0:
                rgb_image_0     = rgb_image.copy()
                depthmap_0      = depthmap.copy()
                camera_pose_0   = torch.from_numpy(camera_pose).clone()
                intrinsics_0    = torch.from_numpy(intrinsics).clone()
                pts3d_0         = torch.from_numpy(pts3d).clone()
                valid_mask_0    = torch.from_numpy(valid_mask).clone()
                traj_mask_0     = traj_mask.clone()

            # for pair mode, dont save frame 0 where views1 is the same as views2
            should_save = (i > 0) if self.training_mode == 'pair' else True
            if should_save:
                save_idx = i - 1 if self.training_mode == 'pair' else i

                # -- A) Fill views1 --
                views1['img'][save_idx] = self.transform(rgb_image_0)
                views1['depthmap'][save_idx] = torch.from_numpy(depthmap_0)
                views1['traj_ptc'][save_idx] = traj_mask_with_pts
                views1['traj_mask'][save_idx] = traj_mask_0
                views1['camera_pose'][save_idx] = camera_pose_0
                views1['camera_intrinsics'][save_idx] = intrinsics_0
                views1['label'][save_idx] = rgb_paths[0].split('/')[-3]
                views1['instance'][save_idx] = osp.split(rgb_paths[0])[1]
                views1['supervised_label'][save_idx] = supervised_label
                views1['pts3d'][save_idx] = pts3d_0
                views1['valid_mask'][save_idx] = valid_mask_0

                # -- B) Fill views2 --
                views2['img'][save_idx] = self.transform(rgb_image)
                views2['img_org'][save_idx] = ToTensor(rgb_image)
                views2['depthmap'][save_idx] = torch.from_numpy(depthmap)
                views2['camera_pose'][save_idx] = torch.from_numpy(camera_pose)
                views2['camera_intrinsics'][save_idx] = torch.from_numpy(intrinsics)
                views2['label'][save_idx] = rgb_paths[i].split('/')[-3]
                views2['instance'][save_idx] = osp.split(rgb_paths[i])[1]
                views2['supervised_label'][save_idx] = supervised_label
                views2['pts3d'][save_idx] = torch.from_numpy(pts3d)
                views2['valid_mask'][save_idx] = torch.from_numpy(valid_mask) & torch.from_numpy(np.isfinite(pts3d).all(axis=-1))

        return [views1, views2]
