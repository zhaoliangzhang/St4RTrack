import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms
import glob
import cv2

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution

ToTensor = transforms.ToTensor()
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

def undistort_depthmap(depthmap, K):
    """
    Convert Euclidean distances (sqrt(x^2 + y^2 + z^2)) in depthmap
    to actual Z values in camera coordinate system.

    Parameters:
    - depthmap: numpy array of shape (H, W), each pixel stores sqrt(x^2 + y^2 + z^2).
    - K: camera intrinsic matrix [3x3], in the form [[fx, 0, cx],
                                                      [0,  fy, cy],
                                                      [0,   0,  1]].

    Returns:
    - real_depth: numpy array of shape (H, W), each pixel stores actual depth Z.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    H, W = depthmap.shape

    # (u, v) pixel grid coordinates
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    x_factor = (uu - cx) / fx
    y_factor = (vv - cy) / fy
    Q = x_factor**2 + y_factor**2

    real_depth = depthmap / np.sqrt(1 + Q)

    return real_depth

def to_opencv_extrinsic(E_old):
    """
    Convert extrinsic matrix E_old from old camera coordinate system (X left, Y up, Z inward)
    to extrinsic matrix E_new in OpenCV coordinate system (X right, Y down, Z outward).

    Parameters:
        E_old: np.ndarray, shape=[4, 4]
               Homogeneous transformation matrix for transforming points from old camera 
               coordinate system to world coordinate system

    Returns:
        E_new: np.ndarray, shape=[4, 4]
               Homogeneous transformation matrix for transforming points from OpenCV 
               coordinate system (new coordinate system) to world coordinate system
    """
    # Construct flip matrix T (new -> old), i.e., -I in the 3×3 coordinate part
    T = np.eye(4)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    # Explanation: first transform points from new coordinate system to old coordinate system, then use E_old to transform to world system
    E_new =  E_old @ T
    return E_new


def convert_intrinsics_to_pixel_space(intrinsics, H, W):

    intrinsics = -(intrinsics) 
    intrinsics[0,0] = -intrinsics[0,0]
    intrinsics[0, :] = intrinsics[0, :] * W
    intrinsics[1, :] = intrinsics[1, :] * H
    return intrinsics



def project_point(extrinsics, intrinsics, point3d, num_frames):
    """Compute the image space coordinates [0, 1] for a set of points.

    Args:
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      point3d: Points in 3D world coordinates.  it has shape [num_frames,
        num_points, 3].
      num_frames: The number of frames in the video.

    Returns:
      Image coordinates in 2D.  The last coordinate is an indicator of whether
        the point is behind the camera.
    """

    homo_transform = torch.inverse(extrinsics)
    homo_intrinsics = torch.zeros((num_frames, 3, 1), dtype=torch.float32)
    homo_intrinsics = torch.cat([intrinsics, homo_intrinsics], dim=2)

    point4d = torch.cat([point3d, torch.ones_like(point3d[:, :, 0:1])], dim=2)
    point4d_cam = torch.matmul(point4d, homo_transform.transpose(1, 2))
    point3d_cam = point4d_cam[:, :, :3].clone()

    projected = torch.matmul(point4d_cam, homo_intrinsics.transpose(1, 2))
    image_coords = projected / projected[:, :, 2:3]
    image_coords = torch.cat([image_coords[:, :, :2], torch.sign(projected[:, :, 2:])], dim=2)
    return image_coords, point3d_cam


def unproject(coord, extrinsics, intrinsics, depth):
    """Unproject points.

    Args:
      coord: Points in 2D coordinates.  it has shape [num_points, 2].  Coord is in
        integer (y,x) because of the way meshgrid happens.
      cam: The camera parameters, as returned by kubric.  'matrix_world' and
        'intrinsics' have a leading axis num_frames.
      depth: Depth map for the scene.

    Returns:
      Image coordinates in 3D.
    """

    shp = torch.tensor(depth.shape)
    idx = coord[:, 0] * shp[1] + coord[:, 1]
    coord = coord[..., ::-1].float()
    shp = shp[1::-1].float().unsqueeze(0)

    # Need to convert from pixel to raster coordinate.
    projected_pt = (coord + 0.5) / shp

    projected_pt = torch.cat(
        [
            projected_pt,
            torch.ones_like(projected_pt[:, -1:]),
        ],
        dim=-1,
    )

    camera_plane = projected_pt @ torch.inverse(intrinsics.transpose(0, 1))
    camera_ball = camera_plane / torch.sqrt(
        torch.sum(
            torch.square(camera_plane),
            dim=1,
            keepdim=True,
        ),
    )
    camera_ball *= depth.reshape(-1)[idx].unsqueeze(1)

    camera_ball = torch.cat(
        [
            camera_ball,
            torch.ones_like(camera_plane[:, 2:]),
        ],
        dim=1,
    )
    points_3d = camera_ball @ extrinsics.transpose(0, 1)
    return points_3d[:, :3] / points_3d[:, 3:]

class KubrickDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/datasets/kubric_full_frame_fp32_cam_pos_batch',
                 dset='train',
                 use_augs=False,
                 S=16,
                 N=1,
                 strides=[1],
                 clip_step=1,
                 verbose=False,
                 clip_step_last_skip = 0,
                 training_mode='seq', # or pair
                 *args, 
                 **kwargs
                 ):

        print('loading kubrick dataset...')
        assert training_mode in ['seq', 'pair'], 'training_mode must be either seq or pair'
        super().__init__(*args, **kwargs)
        self.dataset_label = 'kubrick'
        self.split = dset
        self.S = S # stride
        self.N = N # min num points
        self.verbose = verbose
        self.training_mode = training_mode

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
        self.segment_masks = []
        self.strides = strides

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        self.sequences = sorted(glob.glob(f'{dataset_location}/*/'))
        self.sequences2 = sorted(glob.glob(f'{dataset_location.replace("_batch", "")}/*/'))

        self.sequences = sorted(self.sequences)
        all_subnames = [s.split('/')[-2] for s in self.sequences]
        for seq in self.sequences2[:-10]:
            if seq.split('/')[-2] not in all_subnames:
                self.sequences.append(seq)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')


        for seq in self.sequences:
            seq_name = seq.split('/')[-2]
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'frames')
            annotations_path = os.path.join(seq, '{}_dense.npy'.format(seq_name))

            for stride in strides:
                for ii in range(0, len([f for f in os.listdir(rgb_path) if f.endswith('.png')]) - self.S*max(stride, clip_step_last_skip) + 1, clip_step):
                    full_idx = ii + np.arange(self.S)*stride
                    self.rgb_paths.append([os.path.join(seq, 'frames', '%03d.png' % idx) for idx in full_idx])
                    self.depth_paths.append([os.path.join(seq, 'depths', '%03d.png' % idx) for idx in full_idx])
                    self.segment_masks.append([os.path.join(seq, 'segments', '%03d.png' % idx) for idx in full_idx])
                    self.annotation_paths.append(annotations_path)
                    self.full_idxs.append(full_idx)
                    self.sample_stride.append(stride)
                if self.verbose:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        
        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        

        for stride, count in self.stride_counts.items():
            assert count>0, f"dataset Kubrick stride {stride} has no clips"


        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))

    def visualize_traj(self, rgb_image, traj_3d_proj, i, H, W):

        #visualize whether the traj_3d_proj align with the rgb_image
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
        return len(self.rgb_paths)
    
    def _get_views(self, index, resolution, rng):

        # -- 1) get paths and annotations --
        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        full_idx = self.full_idxs[index]
        segment_masks = self.segment_masks[index]

        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True).item()
        pix_T_cams = annotations['camera_intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['camera_extrinsics'][full_idx].astype(np.float32)

        depth_maps = annotations['depth_map'][full_idx].astype(np.float32)

        # -- 2) supervised for training datasets
        supervised_label = torch.tensor(1.0, dtype=torch.float32)

        # -- 3) prepare views1 and views2 --
        B = self.S if self.training_mode == 'seq' else self.S-1
        W, H = resolution

        # views1: reference frame (0th frame) repeated content, and its trajectory related fields
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
        }

        # views2: own real data of each frame, without traj_ptc / traj_mask (can be kept/deleted according to needs)
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

        # set a random seed for crop
        random_seed = np.random.randint(0, 1000000)
        
        # -- 4) loop over B frames, read data, project, process depth, construct traj, etc. --
        for i in range(self.S):
            impath      = rgb_paths[i]
            rgb_image = imread_cv2(impath)
            H_img, W_img = rgb_image.shape[:2]
            depthpath   = depth_paths[i]
            extrinsics  = to_opencv_extrinsic(cams_T_world[i])   
            intrinsics  = pix_T_cams[i]    
            intrinsics = convert_intrinsics_to_pixel_space(intrinsics, H_img, W_img)
            R = extrinsics[:3, :3]
            t = extrinsics[:3, 3]
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R 
            camera_pose[:3, 3] = t
            
            uv = annotations['coords'].transpose(1,0,2)[full_idx[i]]
            Z = annotations['reproj_depth'][...,None].transpose(1,0,2)[full_idx[i]]
            if i == 0:
                uv_flat = uv.reshape(-1, 2)  # Shape: [H*W, 2]
                reorder_indices = np.lexsort((uv_flat[:, 0], uv_flat[:, 1]))

            # Apply reordering and reshape
            uv = uv[reorder_indices].reshape(512, 512, 2)
            Z = Z[reorder_indices].reshape(512, 512)
            Z = undistort_depthmap(Z, intrinsics)
            Z = Z.reshape(512, 512, 1)

            depthmap = depth_maps[i].reshape(512, 512)
            depthmap = undistort_depthmap(depthmap, intrinsics)
            
            segment_mask = cv2.imread(segment_masks[i], cv2.IMREAD_GRAYSCALE)
            segment_mask = segment_mask.astype(np.float32) > 0

            # --- resize / rescale ---
            rgb_image, depthmap, intrinsics, (l, t, r, b) = cropping.rescale_image_depthmap_crop(
                rgb_image, depthmap, intrinsics, resolution, random_crop=True, random_seed=random_seed
            )
            segment_mask = segment_mask[t:b, l:r]
            
            uv = uv[t:b, l:r]
            Z = Z[t:b, l:r]
            traj_3d_proj = np.concatenate((uv, Z), axis=-1)  # shape [N,3]
            traj_3d_proj = torch.from_numpy(traj_3d_proj).to(torch.float32)

            traj_3d_proj[..., 0] = traj_3d_proj[..., 0] - l
            traj_3d_proj[..., 1] = traj_3d_proj[..., 1] - t

            # project to camera coordinates and then to world coordinates (proj_mode='ptc')
            pts3d_traj, _ = depthmap_to_absolute_camera_coordinates(
                traj_3d_proj.reshape(-1,3).numpy(), intrinsics, camera_pose, proj_mode='ptc'
            )


            W_new, H_new = resolution

            traj_mask_with_pts = torch.from_numpy(pts3d_traj).reshape(H_new, W_new, 3)
            # do full 3D conversion on depthmap (proj_mode='depth')
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depthmap, intrinsics, camera_pose, proj_mode='depth'
            )
            # filter out inf / nan
            valid_mask = valid_mask & np.isfinite(pts3d).all(axis=-1)
            # if i=0, copy this frame's data to X_0 for views1
            if i == 0:
                rgb_image_0     = rgb_image.copy()  
                depthmap_0      = depthmap.copy()
                camera_pose_0   = torch.from_numpy(camera_pose).clone()
                intrinsics_0    = torch.from_numpy(intrinsics).clone()
                pts3d_0         = torch.from_numpy(pts3d).clone()
                valid_mask_0    = torch.from_numpy(valid_mask).clone() & torch.from_numpy(segment_mask).clone()
                traj_mask_0     = torch.ones((H, W), dtype=torch.bool) & torch.from_numpy(segment_mask).clone()
                
            
            # for pair mode, dont save frame 0 where views1 is the same as views2
            should_save = (i > 0) if self.training_mode == 'pair' else True
            if should_save:
                save_idx = i - 1 if self.training_mode == 'pair' else i
            
                # -- A) fill views1 (save 0th frame data, but repeated in loop)
                # traj_ptc uses coords from current i frame, but traj_mask uses 0th frame
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

                # -- B) fill views2 (save real data of each frame)
                views2['img'][save_idx] = self.transform(rgb_image)
                views2['img_org'][save_idx] = ToTensor(rgb_image)
                views2['depthmap'][save_idx] = torch.from_numpy(depthmap)
                views2['camera_pose'][save_idx] = torch.from_numpy(camera_pose)
                views2['camera_intrinsics'][save_idx] = torch.from_numpy(intrinsics)
                views2['label'][save_idx] = rgb_paths[i].split('/')[-3]
                views2['instance'][save_idx] = osp.split(rgb_paths[i])[1]
                views2['supervised_label'][save_idx] = supervised_label
                views2['pts3d'][save_idx] = torch.from_numpy(pts3d)
                views2['valid_mask'][save_idx] = torch.from_numpy(valid_mask) & torch.from_numpy(np.isfinite(pts3d).all(axis=-1)) & torch.from_numpy(segment_mask).clone()
        
        
        return [views1, views2]