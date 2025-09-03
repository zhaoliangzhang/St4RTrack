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

def intrinsics_to_pixel_space(intrinsics: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    """
    Convert camera intrinsics to pixel space.

    Args:
        intrinsics (np.ndarray): Camera intrinsics in normalized space (3x3 matrix).
        resolution (Tuple[int, int]): The resolution of the image (W, H).

    Returns:
        np.ndarray: Camera intrinsics matrix in pixel space (3x3).
    """
    H, W = resolution
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    # Convert to pixel space by scaling with resolution
    fx_pixel = fx * W
    fy_pixel = fy * H
    cx_pixel = cx * W
    cy_pixel = cy * H



    # Construct the pixel space intrinsics matrix
    pixel_intrinsics = np.array([
        [fx_pixel, 0, cx_pixel],
        [0, fy_pixel, cy_pixel],
        [0, 0, 1]
    ])
    
    return pixel_intrinsics

def extract_moge_depth(rgb_paths, output_path, num_frames=None):
    """
    Extract depth maps using MoGe model for a sequence of images.
    
    Args:
        rgb_paths (list): List of paths to RGB images
        output_path (str): Path to save the depth results
        num_frames (int): Number of frames to process
    """
    import sys
    sys.path.append("third_party/MoGe")  # Make sure MoGe is on Python path
    from moge.model import MoGeModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()
    
    save_dict = {}
    if num_frames is None:
        num_frames = len(rgb_paths)
    
    with torch.no_grad():
        for img_path in tqdm(rgb_paths[:num_frames], desc="Extracting depth"):
            # Read and preprocess image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            img_bgr_resized = img_bgr
            img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            img_tensor = torch.tensor(img_rgb / 255.0, 
                                    dtype=torch.float32,
                                    device=device).permute(2, 0, 1)

            # Run MoGe inference
            output = model.infer(img_tensor)
            
            # Store results
            save_dict[img_path] = {
                "depth": output["depth"].cpu().numpy(),
                "mask": output["mask"].cpu().numpy(),
                "intrinsics": output["intrinsics"].cpu().numpy(),
                "points": output["points"].cpu().numpy()
            }
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, save_dict)
    return output_path

class CustomDUSt3R(BaseStereoViewDataset):
    def __init__(self, # only keyword arguments
                 dataset_location = "./Davis_data/DAVIS/JPEGImages/480p/rollerblade/",
                 depth_path = "./Davis_data/DAVIS/JPEGImages/480p/rollerblade/depth",
                 S=16,
                 stride=2,
                 clip_step=1,  # Added clip_step parameter
                 quick=False,
                 extract_depth=True,
                 num_frames=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        print('loading Custom dataset...')

        self.dataset_label = 'Custom'
        self.dataset_location = dataset_location
        self.depth_path = depth_path
        os.makedirs(self.depth_path, exist_ok=True)
        self.S = S
        self.stride = stride
        self.clip_step = clip_step
        self.sequences = sorted(glob.glob(self.dataset_location))

        if quick:
            self.sequences = self.sequences[200:500]

        print(f"Found {len(self.sequences)} sequences")

        self.rgb_paths = []
        self.depth_paths = []
        self.strides = []
        self.depth_data = {}  # New: store loaded depth data

        for seq in tqdm(self.sequences):
            depth_path = os.path.join(self.depth_path, seq.split('/')[-2] + f"moge_results.npy")
            rgb_paths = (sorted(glob.glob(os.path.join(seq, "*.jpg"))) + sorted(glob.glob(os.path.join(seq, "*.png"))))
            if num_frames is not None:
                rgb_paths = rgb_paths[:num_frames]
            if not rgb_paths:
                print(f"No images found for sequence {seq}")
                continue
                
            if not os.path.exists(depth_path) and extract_depth:
                print(f"Extracting depth for sequence {seq}")
                depth_path = extract_moge_depth(rgb_paths, depth_path)
            elif not os.path.exists(depth_path):
                print(f"Depth path {depth_path} does not exist")
                continue
            
            # Load depth data once during initialization
            try:
                depth = np.load(depth_path, allow_pickle=True).item()
                self.depth_data[depth_path] = depth
            except Exception as e:
                print(f"Error loading depth data for {depth_path}: {e}")
                continue

            keys = list(depth.keys())
            
            num_frames = len(rgb_paths)
            stride = min(self.stride, num_frames // self.S)
            for ii in range(0, num_frames - self.S*stride + 1, self.clip_step):
                frame_indices = ii + np.arange(self.S)*stride
                if frame_indices[-1] >= num_frames:
                    print(f"Not enough frames left for sequence {seq}")
                    continue
                    
                sequence_rgb_paths = [rgb_paths[idx] for idx in frame_indices]
                sequence_keys = [keys[idx] for idx in frame_indices]
                
                if not all(key in depth for key in sequence_keys):
                    print(f"Missing keys: {sequence_keys}, {depth.keys()}")
                    continue
                
                self.rgb_paths.append(sequence_rgb_paths)
                self.depth_paths.append(depth_path)
                self.strides.append(stride)

        self.idx_ssl = len(self.rgb_paths) - 1
        print(f"Finally {len(self.rgb_paths)} sequences")

    def __len__(self):
        return len(self.rgb_paths)

    def _get_views(self, index, resolution=(512, 288), rng=None):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        depth = self.depth_data[depth_path]  # Use pre-loaded depth data

        B = self.S
        W, H = resolution
        supervised_label = torch.tensor(0.0, dtype=torch.float32)

        # views1: Store reference frame (frame 0) repeated content and its trajectory-related fields
        views1 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'camera_intrinsics': torch.zeros((B, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * B,
            'instance'         : [''] * B,
            'supervised_label' : torch.zeros(B, dtype=torch.float32),
        }

        # views2: Store each frame's actual data, excluding traj_ptc / traj_mask (can be kept/removed as needed)
        views2 = {
            'img'              : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'img_org'          : torch.zeros((B, 3, H, W), dtype=torch.float32),
            'depthmap'         : torch.zeros((B, H, W),   dtype=torch.float32),
            'camera_intrinsics': torch.zeros((B, 3, 3),   dtype=torch.float32),
            'dataset'          : self.dataset_label,
            'label'            : [''] * B,
            'instance'         : [''] * B,
            'supervised_label' : torch.zeros(B, dtype=torch.float32),
            'pts3d_moge'            : torch.zeros((B, H, W, 3), dtype=torch.float32),
            'valid_mask'       : torch.zeros((B, H, W),    dtype=torch.bool),
        }

        for i in range(B):
            depth_map = depth[rgb_path[i]]["depth"]  # Use the RGB path as key
            # turn nan to 0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            mask = depth[rgb_path[i]]["mask"]
            # set the depth_map where mask is 0 to -1
            depth_map = depth_map * mask + (-1) * (1 - mask)
            intrinsics = depth[rgb_path[i]]["intrinsics"]

            # Convert intrinsics to pixel unit
            intrinsics = intrinsics_to_pixel_space(intrinsics, depth_map.shape)

            points = depth[rgb_path[i]]["points"]
            impath = rgb_path[i]
            rgb_image = imread_cv2(impath)
        
            size_depth = depth_map.shape
            # resize image to the same size as depth
            rgb_image = cv2.resize(rgb_image, (size_depth[1], size_depth[0]))

        

            rgb_image, depth_map, intrinsics, _ = cropping.rescale_image_depthmap_crop(
                rgb_image, depth_map, intrinsics, resolution)

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(
                depth_map, intrinsics, camera_pose=None, proj_mode='depth')

            if i == 0:
                rgb_image_0 = rgb_image.copy()
                intrinsics_0 = torch.from_numpy(intrinsics).clone()

        

            views1['img'][i] = self.transform(rgb_image_0)
            views1['camera_intrinsics'][i] = intrinsics_0
            views1['label'][i] = rgb_path[0].split('/')[-2]
            views1['instance'][i] = osp.split(rgb_path[0])[1]
            views1['supervised_label'][i] = supervised_label

            views2['img'][i] = self.transform(rgb_image)
            views2['img_org'][i] = ToTensor(rgb_image)
            views2['depthmap'][i] = torch.from_numpy(depth_map)
            views2['camera_intrinsics'][i] = torch.from_numpy(intrinsics)
            views2['label'][i] = rgb_path[i].split('/')[-2]
            views2['instance'][i] = osp.split(rgb_path[i])[1]
            views2['supervised_label'][i] = supervised_label
            views2['pts3d_moge'][i] = torch.from_numpy(pts3d)
            views2['valid_mask'][i] = torch.from_numpy(valid_mask)

        return views1, views2


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import gradio as gr
    import random

    dataset = CustomDUSt3R(quick=False, resolution=[(512,288)])
    print(len(dataset), 'sequences')
    # print(dataset[0])
    def visualize(idx, frame=0):
        views = dataset[idx]
        viz = SceneViz()
        views1, views2 = views
        pts3d = views2['pts3d_moge'][frame]
        valid_mask = views2['valid_mask'][frame]
        colors = rgb(views2['img'][frame])
        viz.add_pointcloud(pts3d, colors, valid_mask)
        os.makedirs('./tmp/custom', exist_ok=True)
        path = f'./tmp/custom/{idx}_f{frame}.glb'

        # visualize the img
        img_save_path = f'./tmp/custom/{idx}_f{frame}.png'
        img = views2['img_org'][frame].permute(1,2,0).numpy() * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_save_path, img)

        # visualize the depthmap
        depth_save_path = f'./tmp/custom/{idx}_f{frame}_depth.png'
        depth = views2['depthmap'][frame].numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(depth_save_path, depth)

        return viz.save_glb(path)

    for idx in range(1):
        visualize(idx, frame=1)
        print(f'visualized {idx}')
    
            