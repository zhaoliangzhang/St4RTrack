import os
import copy
import gzip
import logging
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import os.path as osp
from glob import glob

from PIL import Image
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from pytorch3d.renderer.cameras import PerspectiveCameras

from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)
from cotracker3D.datasets.utils import CoTrackerData
from pytorch3d.utils import (
            opencv_from_cameras_projection,
        )


class Traj3DDataset(data.Dataset):
    def __init__(self):
        self.sample_list = []
        self.depth_eps = 1e-5

    def _load_16big_png_depth(self, depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

    def _get_pytorch3d_camera(
        self, entry_viewpoint, image_size, scale: float
    ) -> PerspectiveCameras:
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
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
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )


    def __len__(self):
        return len(self.sample_list)


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    camera_name: Optional[str] = None
    trajectories: Optional[str] = None


class DynamicReplicaDataset(Traj3DDataset):
    def __init__(
        self,
        root='/fsx-repligen/nikitakaraev/datasets/dynamic_replica',
        split="valid",
        traj_per_sample=256,
        crop_size=None,
        sample_len=-1,
        only_first_n_samples=-1,
        rgbd_input=False,
    ):
        super(DynamicReplicaDataset, self).__init__()
        self.root = root
        self.sample_len = sample_len
        self.split = split
        self.traj_per_sample = traj_per_sample
        self.rgbd_input = rgbd_input
        self.crop_size = crop_size
        frame_annotations_file = f"frame_annotations_{split}.jgz"
        
        with gzip.open(
            os.path.join(root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )
        seq_annot = defaultdict(list)
        for frame_annot in frame_annots_list:
            if frame_annot.camera_name=='left':
                seq_annot[frame_annot.sequence_name].append(
                    frame_annot
                )
        
        for seq_name in seq_annot.keys():
            seq_len = len(seq_annot[seq_name])
        
            print("seq_len", seq_name, seq_len)
            
            step = self.sample_len if self.sample_len > 0 else seq_len
            counter = 0
        
            for ref_idx in range(0, seq_len, step):
                sample = seq_annot[seq_name][ref_idx:ref_idx + step]
                self.sample_list.append(sample)
                counter += 1
                if only_first_n_samples > 0 and counter >= only_first_n_samples:
                    break
    
    def crop(self, rgbs, trajs, depths=None):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        # print('depths',depths.shape,'rgbs',rgbs.shape)
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = (
            0
            if self.crop_size[0] >= H_new
            else (H_new - self.crop_size[0])//2
        )
        x0 = (
            0
            if self.crop_size[1] >= W_new
            else (W_new - self.crop_size[1])//2
        )
        rgbs = [
            rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            for rgb in rgbs
        ]
        if depths is not None:
            depths = [
                depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
                for depth in depths
            ]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return rgbs, trajs, depths

    def __getitem__(self, index):
        sample = self.sample_list[index]
        T = len(sample)
        rgbs, visibilities, traj_3d, traj_3d_world,depths, cameras, traj_masks=[], [], [], [], [], [], []
        
        H, W = sample[0].image.size
        image_size = (H,W)

        for i in range(T):
            # mask_path = os.path.join(self.root, self.split, sample[i].depth.mask_path)
            # masks.append(Image.open(mask_path))
            
            traj_path = os.path.join(self.root, self.split, sample[i].trajectories['path'])
            traj = torch.load(traj_path)
            # print('traj',traj['instances'])
            traj_masks.append((traj['instances']>0).numpy())
            visibilities.append(traj["verts_inds_vis"].numpy())
            
            rgbs.append(traj["img"].numpy()) 
            
            depth = self._load_16big_png_depth(os.path.join(self.root,self.split, sample[i].depth.path))
            # depth_mask = Image.open(os.path.join(root, split, sample[i].depth.mask_path))
            # depth = depth * depth_mask
            depths.append(depth)
            
            
            viewpoint = self._get_pytorch3d_camera(
                        sample[i].viewpoint,
                        sample[i].image.size,
                        scale=1.0,
                    )
            R, T, K = opencv_from_cameras_projection(
                viewpoint,
                torch.tensor([H, W])[None],
            )
            # print('R',R.shape)
            # print('T',T.shape)
            # print('K',K.shape)
            viewpoint_tf = PerspectiveCameras(
                focal_length=torch.tensor([K[0][0][0],K[0][1][1]])[None] ,
                principal_point=torch.tensor([K[0][0][2],K[0][1][2]])[None],
                R=R,
                T=T,
                in_ndc=False
            )
            cameras.append(viewpoint_tf)
            # viewpoint.principal_point=reversed(principal_point)
            # print('traj["traj_3d_world"]',traj["traj_3d_world"].shape)
            traj_3d_w = traj["traj_3d_world"].clone()
            traj_3d_proj = (
                    viewpoint
                    .transform_points_screen(traj_3d_w, image_size=[H, W])
                )[...,:2]
            traj_3d_proj_depth=viewpoint.get_world_to_view_transform().transform_points(traj_3d_w)[...,2:3]
            traj_3d_proj = torch.cat([traj_3d_proj,traj_3d_proj_depth], dim=-1)
            traj_3d_world.append(traj["traj_3d_world"])
            traj_3d.append(traj_3d_proj.numpy())
            # print('traj_2d',traj["traj_2d"].shape)
            # print( torch.allclose(traj_3d_proj[...,:3],traj["traj_2d"],rtol=1e-04, atol=1e-05))
            # print('traj_diff',(traj_3d_proj[...,:2]-traj["traj_2d"][...,:2]).abs().mean())
            

        traj_3d = np.stack(traj_3d)
        traj_3d_world = np.stack(traj_3d_world)
        visibility = np.stack(visibilities)
        traj_masks = np.stack(traj_masks)
        T, N, D = traj_3d.shape
        # subsample trajectories for augmentations
        visible_inds_sampled = torch.randperm(N)[: self.traj_per_sample]
        
        traj_3d = traj_3d[:, visible_inds_sampled]
        traj_3d_world = traj_3d_world[:, visible_inds_sampled]
        visibility = visibility[:, visible_inds_sampled]
        traj_masks = traj_masks[:, visible_inds_sampled]
        
        if self.crop_size is not None:
            rgbs, traj_3d, depths = self.crop(rgbs, traj_3d, depths)
            H, W, _ = rgbs[0].shape
            image_size= self.crop_size

        visibility[traj_3d[:, :, 0] > image_size[1] - 1] = False
        visibility[traj_3d[:, :, 0] < 0] = False
        visibility[traj_3d[:, :, 1] > image_size[0] - 1] = False
        visibility[traj_3d[:, :, 1] < 0] = False
        
        # filter out points that're visible for less than 10 frames
        visible_inds_resampled = visibility.sum(0)>10
        # print('visible_inds_resampled',visible_inds_resampled.shape)
        traj_3d = torch.from_numpy(traj_3d[:, visible_inds_resampled])
        traj_3d_world = torch.from_numpy(traj_3d_world[:, visible_inds_resampled])
        visibility = torch.from_numpy(visibility[:, visible_inds_resampled])
        traj_masks = torch.from_numpy(traj_masks[:, visible_inds_resampled])
        
        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0)
        # masks =  np.stack(masks, 0)
        
        # masks = torch.from_numpy(masks).reshape(T, 1, H, W).float()
        depth = torch.from_numpy(depths).reshape(T, 1, H, W).float()
        video = torch.from_numpy(rgbs).reshape(T, H, W, 3).permute(0, 3, 1, 2).float()

        if self.rgbd_input:
            video = torch.cat([video, depth], dim=1)

        return CoTrackerData(
            video=video,
            # segmentation=masks,
            trajectory=traj_3d,
            traj_masks=traj_masks,
            visibility=visibility,
            depth=depth,
            camera=cameras,
            valid=torch.ones(T,N),
            seq_name=sample[0].sequence_name
        )