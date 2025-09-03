import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import torchvision.transforms as transforms

import glob
import cv2
import json
import PIL
from PIL import Image

import gzip
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import dust3r.datasets.utils.cropping as cropping

from dust3r.utils.image import imread_cv2
from dust3r.utils.misc import get_stride_distribution
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
ToTensor = transforms.ToTensor()

def get_pytorch3d_camera(entry_viewpoint, image_size, scale: float
                         ) -> PerspectiveCameras:
    assert entry_viewpoint is not None
    # principal point and focal length
    principal_point = torch.tensor(
        entry_viewpoint['principal_point'], dtype=torch.float
    )
    focal_length = torch.tensor(entry_viewpoint['focal_length'], dtype=torch.float)

    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    # first, we convert from the dataset's NDC convention to pixels
    format = entry_viewpoint['intrinsics_format']
    if format.lower() == "ndc_norm_image_bounds":
        # this is e.g. currently used in CO3D for storing intrinsics
        rescale = half_image_size_wh_orig
    elif format.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {format}")

    # principal point and focal length in pixels
    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    K_pixel = np.array([
        [focal_length_px[0], 0,                principal_point_px[0]],
        [0,                  focal_length_px[1], principal_point_px[1]],
        [0,                  0,                 1]
    ])

    # now, convert from pixels to PyTorch3D v0.5+ NDC convention
    # if self.image_height is None or self.image_width is None:
    out_size = list(reversed(image_size))

    half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
    half_min_image_size_output = half_image_size_output.min()

    # rescaled principal point and focal length in ndc
    principal_point = (
        half_image_size_output - principal_point_px * scale
    ) / half_min_image_size_output
    focal_length = focal_length_px * scale / half_min_image_size_output

    return PerspectiveCameras(
        focal_length=focal_length[None],
        principal_point=principal_point[None],
        R=torch.tensor(entry_viewpoint['R'], dtype=torch.float)[None],
        T=torch.tensor(entry_viewpoint['T'], dtype=torch.float)[None],
    ), K_pixel

def convert_ndc_to_pixel_intrinsics(
    focal_length_ndc, principal_point_ndc, image_width, image_height, intrinsics_format='ndc_isotropic'
):
    f_x_ndc, f_y_ndc = focal_length_ndc
    c_x_ndc, c_y_ndc = principal_point_ndc

    # Compute half image size
    half_image_size_wh_orig = np.array([image_width, image_height]) / 2.0

    # Determine rescale factor based on intrinsics_format
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig  # [image_width/2, image_height/2]
    elif intrinsics_format.lower() == "ndc_isotropic":
        rescale = np.min(half_image_size_wh_orig)  # scalar value
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics_format}")

    # Convert focal length from NDC to pixel coordinates
    if intrinsics_format.lower() == "ndc_norm_image_bounds":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale
    elif intrinsics_format.lower() == "ndc_isotropic":
        focal_length_px = np.array([f_x_ndc, f_y_ndc]) * rescale

    # Convert principal point from NDC to pixel coordinates
    principal_point_px = half_image_size_wh_orig - np.array([c_x_ndc, c_y_ndc]) * rescale

    # Construct the intrinsics matrix in pixel coordinates
    K_pixel = np.array([
        [focal_length_px[0], 0,                principal_point_px[0]],
        [0,                  focal_length_px[1], principal_point_px[1]],
        [0,                  0,                 1]
    ])

    return K_pixel

def load_16big_png_depth(depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

def camera_inv(camera_pose):
    R = camera_pose[:3,:3]
    t = camera_pose[:3,3]
    camera_pose_inv = np.eye(4, dtype=np.float32)
    camera_pose_inv[:3,:3] = R.T
    camera_pose_inv[:3,3] = -R.T @ t
    return camera_pose_inv

class DynamicReplicaDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/dynamic_stereo/dynamic_replica_data/train',
                 use_augs=False,
                 S=6,
                 strides=[4,5,6],
                 clip_step=1,
                 verbose=False,
                 dist_type=None,
                 clip_step_last_skip = 0,
                 sequence_split = 0,
                 mix_split = 2,
                 training_mode='seq',
                 *args, 
                 **kwargs
                 ):

        print('loading dynamic replica dataset...')
        super().__init__(*args, **kwargs)
        self.training_mode = training_mode
        self.dataset_label = 'dynamic_replica'
        self.S = S # stride
        self.verbose = verbose

        self.use_augs = use_augs

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []
        self.sample_stride = []
        self.supervised_label = []
        self.moge_depth_paths = []
        self.moge_depth_mask_paths = []
        self.strides = strides 

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location))

        if 'train' in dataset_location:
            anno_path = os.path.join(dataset_location, 'frame_annotations_train_full.json')
            with open(anno_path, 'r') as f:
                self.anno = json.load(f)
        elif 'test' in dataset_location:
            anno_path = os.path.join(dataset_location, 'frame_annotations_test.jgz')
            with gzip.open(anno_path, 'r') as f:
                self.anno = json.load(f)
        else:
            raise ValueError('Unknown dataset location')


        anno_by_seq = {}
        for a in self.anno:
            seq_name = a['sequence_name']

            if seq_name not in anno_by_seq:
                anno_by_seq[seq_name] = []
            anno_by_seq[seq_name].append(a)

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = anno_by_seq.keys()
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s ' % (len(self.sequences), dataset_location))

        if sequence_split > 0:
            self.sequences = list(self.sequences)[:len(self.sequences) // sequence_split]
        

        for seq_idx, seq in enumerate(self.sequences):
            if self.verbose: 
                print('seq', seq)
            anno = anno_by_seq[seq]

            for stride in strides:
                for ii in range(0, len(anno) - self.S * max(stride, clip_step_last_skip) + 1, clip_step):
                    full_idx = ii + np.arange(self.S) * stride
                    try:
                        rgb_paths = [os.path.join(dataset_location, anno[idx]['image']['path']) for idx in full_idx]
                        depth_paths = [os.path.join(dataset_location, anno[idx]['depth']['path']) for idx in full_idx]
                        traj_paths = [
                            os.path.join(dataset_location, anno[idx]['trajectories']['path'])
                            if anno[idx].get('trajectories') and anno[idx]['trajectories'].get('path') and 'path' in anno[idx]['trajectories']
                            else None
                            for idx in full_idx
                        ]
                    except KeyError as e:
                        # Skip this sample if any key is missing
                        print(f"\nSkipping due to missing key: {e}")
                        continue

                    # Check if all paths are valid, if not, skip
                    if (
                        not all([p and os.path.exists(p) for p in rgb_paths]) or
                        not all([p and os.path.exists(p) for p in depth_paths]) or
                        not all([p and os.path.exists(p) for p in traj_paths])
                    ):
                        continue
                    if seq_idx < len(self.sequences) // mix_split:
                        self.supervised_label.append(torch.tensor(1.))
                    else:
                        self.supervised_label.append(torch.tensor(0.))
                    self.rgb_paths.append(rgb_paths)
                    self.depth_paths.append(depth_paths)
                    self.moge_depth_mask_paths.append([path.replace('.png', '_mask.npy') for path in rgb_paths])
                    self.moge_depth_paths.append([path.replace('.png', '_depth.npy') for path in rgb_paths])
                    self.traj_paths.append(traj_paths)
                    self.annotation_paths.append([anno[idx] for idx in full_idx])
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
        print('stride counts:', self.stride_counts)

        for stride, count in self.stride_counts.items():
            assert count>0, f"dataset DynamicReplica stride {stride} has no clips"
        
        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        print('collected %d clips of length %d in %s' % (
            len(self.rgb_paths), self.S, dataset_location,))

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
        self.traj_paths = [self.traj_paths[i] for i in resampled_idxs]
        self.annotation_paths = [self.annotation_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]
        self.supervised_label = [self.supervised_label[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
    
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

    def _get_views(self, index, resolution, rng):

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        traj_paths = self.traj_paths[index]
        annotations = self.annotation_paths[index]

        supervised_label = torch.tensor(1.)

        views1 = {}
        views2 = {}

        B = self.S
        if self.training_mode == 'seq':
            tensor_size = B
        elif self.training_mode == 'pair':
            tensor_size = B-1 # should be 1
        else:
            raise ValueError(f'training_mode must be either seq or pair, got {self.training_mode}')
        
        W, H = resolution
        views1 = {
            'img': torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'depthmap': torch.zeros(tensor_size, H, W, dtype=torch.float32),
            'traj_ptc': torch.zeros(tensor_size, H, W, 3, dtype=torch.float32),
            'traj_mask': torch.zeros(tensor_size, H, W, dtype=torch.bool),
            'camera_pose': torch.zeros(tensor_size, 4, 4, dtype=torch.float32),
            'camera_intrinsics': torch.zeros(tensor_size, 3, 3, dtype=torch.float32),
            'dataset': self.dataset_label,
            'label': [''] * tensor_size,
            'instance': [''] * tensor_size,
            'supervised_label': torch.ones(tensor_size, dtype=torch.float32),
            'pts3d': torch.zeros(tensor_size, H, W, 3, dtype=torch.float32),
            'valid_mask': torch.zeros(tensor_size, H, W, dtype=torch.bool)
        }

        views2 = {
            'img': torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'img_org': torch.zeros((tensor_size, 3, H, W), dtype=torch.float32),
            'depthmap': torch.zeros(tensor_size, H, W, dtype=torch.float32),
            'camera_pose': torch.zeros(tensor_size, 4, 4, dtype=torch.float32),
            'camera_intrinsics': torch.zeros(tensor_size, 3, 3, dtype=torch.float32),
            'dataset': self.dataset_label,
            'label': [''] * tensor_size,
            'instance': [''] * tensor_size,
            'supervised_label': torch.ones(tensor_size, dtype=torch.float32),
            'pts3d': torch.zeros(tensor_size, H, W, 3, dtype=torch.float32),
            'valid_mask': torch.zeros(tensor_size, H, W, dtype=torch.bool)
        }

        for i in range(B):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            traj_path = traj_paths[i]
            annotation = annotations[i]

            # load image and depth
            rgb_image = imread_cv2(impath)
            H, W = rgb_image.shape[:2]

            viewpoint, k_pixels = get_pytorch3d_camera(
            annotation['viewpoint'],
            annotation['image']['size'],
            scale=1.0,
            )

            R, T, K = opencv_from_cameras_projection(
                viewpoint,
                torch.tensor([H, W])[None],
            )

            # load camera params
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3,:3] = R[0].T
            camera_pose[:3,3] = -R[0].T @ T[0]
            depthmap = load_16big_png_depth(depthpath)
            # #render the depthmap out as colormap

            traj = torch.load(traj_path, weights_only=False)

            traj_3d_w = traj["traj_3d_world"].clone()
            traj_3d_proj = (viewpoint.transform_points_screen(traj_3d_w, image_size=[H, W]))[...,:2]
            #filter out points that are outside the image
            traj_3d_proj_depth=viewpoint.get_world_to_view_transform().transform_points(traj_3d_w)[...,2:3]
            traj_3d_proj = torch.cat([traj_3d_proj, traj_3d_proj_depth], dim=-1)

            if i==0:
                f0_traj_vis = torch.load(traj_paths[0], weights_only=False)['verts_inds_vis']
                valid_mask_traj = (traj_3d_proj[..., 0] >= 0) & (traj_3d_proj[..., 0] < W) & (traj_3d_proj[..., 1] >= 0) & (traj_3d_proj[..., 1] < H)
                #merge f0_traj_vis with valid_mask
                f0_traj_vis = valid_mask_traj & f0_traj_vis

            traj_3d_proj = traj_3d_proj[f0_traj_vis]

            # load intrinsics
            intrinsics = k_pixels
            intrinsics = intrinsics.astype(np.float32)

            if not isinstance(rgb_image, PIL.Image.Image):
                rgb_image = PIL.Image.fromarray(rgb_image)
            
            rgb_image, depthmap, intrinsics = cropping.rescale_image_depthmap(rgb_image, depthmap, intrinsics, resolution)

            #rescale traj_3d_proj's x,y to the new image size
            traj_3d_proj[..., :2] = traj_3d_proj[..., :2] * resolution[0] / W

            pts3d_traj, _ = depthmap_to_absolute_camera_coordinates(traj_3d_proj.numpy(), intrinsics, camera_pose, proj_mode='ptc')
            W, H = resolution

            # Initialize the mask and the mask with points
            traj_mask_with_pts = torch.zeros((H, W, 3), dtype=torch.float32)
            if i==0:
                traj_mask = torch.zeros((H, W), dtype=torch.bool)
                indices = traj_3d_proj[:, :2].round().long()
                indices[:, 0] = indices[:, 0].clamp(0, W - 1)
                indices[:, 1] = indices[:, 1].clamp(0, H - 1)
                traj_mask[indices[:, 1], indices[:, 0]] = True
            
            traj_mask_with_pts[indices[:, 1], indices[:, 0]] = torch.from_numpy(pts3d_traj)

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(depthmap, intrinsics, camera_pose, z_far=self.z_far, proj_mode='depth')

            if i==0:
                rgb_image_0 = rgb_image.copy()
                depthmap_0 = depthmap.copy()
                traj_mask_0 = traj_mask.clone()
                camera_pose_0 = torch.from_numpy(camera_pose).clone()
                intrinsics_0 = torch.from_numpy(intrinsics).clone()
                pts3d_0 = torch.from_numpy(pts3d).clone()
                valid_mask_0 = torch.from_numpy(valid_mask).clone() & np.isfinite(pts3d_0).all(axis=-1)
            
            # Save to views only if we want this frame
            should_save = (i > 0) if self.training_mode == 'pair' else True
            if should_save:
                save_idx = i-1 if self.training_mode == 'pair' else i
                
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

                views2['img'][save_idx] = self.transform(rgb_image)
                views2['img_org'][save_idx] = ToTensor(rgb_image)
                views2['depthmap'][save_idx] = torch.from_numpy(depthmap)
                views2['camera_pose'][save_idx] = torch.from_numpy(camera_pose)
                views2['camera_intrinsics'][save_idx] = torch.from_numpy(intrinsics)
                views2['label'][save_idx] = rgb_paths[i].split('/')[-3]
                views2['instance'][save_idx] = osp.split(rgb_paths[i])[1]
                views2['supervised_label'][save_idx] = supervised_label
                views2['pts3d'][save_idx] = torch.from_numpy(pts3d)
                views2['valid_mask'][save_idx] = torch.from_numpy(valid_mask) & np.isfinite(pts3d).all(axis=-1)

        
        return [views1, views2]