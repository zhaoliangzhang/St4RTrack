import numpy as np
import torch
import os
import glob
import cv2
from PIL import Image
import json
import gzip
import io
from tqdm import tqdm
import concurrent.futures
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection

NUM_FRAMES = 128
SUBSAMPLE_FACTOR = 25

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

def process_dynamic_stereo_sequence(seq, output_dir, dataset_location):
    # Load annotations
    anno_path = os.path.join(dataset_location, 'frame_annotations_test.jgz')  # Assuming you are using the test set
    with gzip.open(anno_path, 'r') as f:
        anno = json.load(f)
    
    # Organize annotations by sequence name
    anno_by_seq = {}
    for a in anno:
        seq_name = a['sequence_name']
        if seq_name not in anno_by_seq:
            anno_by_seq[seq_name] = []
        anno_by_seq[seq_name].append(a)

    # Prepare containers for storing sequence data
    images_jpeg_bytes = []
    tracks_xyz_cam = []
    visibility = []
    extrinsics = []
    fx_fy_cx_cy_list = []
    depth_maps = []
    
    # Get sequence name without _source_left/_source_right
    seq_name = seq.split('/')[-2].replace('_source_left', '').replace('_source_right', '')
    
    # Process each frame in the sequence
    for i, a in tqdm(enumerate(anno_by_seq[seq_name])):
        # Get image paths and trajectory paths
        rgb_path = os.path.join(dataset_location, a['image']['path'])
        depth_path = os.path.join(dataset_location, a['depth']['path'])
        try:
            traj_path = os.path.join(dataset_location, a['trajectories']['path'])
        except:
            print(f"Trajectory not found for {a['image']['path']}")
            continue

        # Load trajectory (should be in camera coordinates)
        traj = torch.load(traj_path)
        traj_3d = traj["traj_3d_world"]  # In world coordinates

        # Get camera parameters
        viewpoint, k_pixels = get_pytorch3d_camera(
            a['viewpoint'],
            a['image']['size'],
            scale=1.0,
        )
        R, T, K = opencv_from_cameras_projection(viewpoint, torch.tensor([a['image']['size'][1], a['image']['size'][0]])[None])

        # Store intrinsics
        fx_fy_cx_cy_list.append([k_pixels[0, 0], k_pixels[1, 1], k_pixels[0, 2], k_pixels[1, 2]])
        
        # Calculate camera pose
        extrinsics_w2c = np.eye(4, dtype=np.float32)
        extrinsics_w2c[:3, :3] = R
        extrinsics_w2c[:3, 3] = T

        # Convert 3D trajectory from world to camera coordinates
        traj_3d_homogeneous = np.hstack((traj_3d, np.ones((traj_3d.shape[0], 1))))  # Convert to homogeneous coordinates
        traj_3d_camera = np.dot(extrinsics_w2c, traj_3d_homogeneous.T).T  # Apply transformation
        traj_3d_camera = traj_3d_camera[:, :3]  # Drop the homogeneous coordinate

        # Project 3D points to 2D using camera intrinsics
        visibility_mask = traj['verts_inds_vis']

        if SUBSAMPLE_FACTOR > 1:
            # print(traj_3d_camera.shape, visibility_mask.shape)
            traj_3d_camera = traj_3d_camera[::SUBSAMPLE_FACTOR]
            visibility_mask = visibility_mask[::SUBSAMPLE_FACTOR]

        # Load the RGB and Depth images
        rgb_image = Image.open(rgb_path)
        depth_map = load_16big_png_depth(depth_path)
        depth_maps.append(depth_map)
        
        # Convert RGBA to RGB if necessary before saving as JPEG
        if rgb_image.mode == 'RGBA':
            rgb_image = rgb_image.convert('RGB')

        # Convert the image to JPEG byte array
        with io.BytesIO() as img_byte_arr:
            rgb_image.save(img_byte_arr, format="JPEG")
            img_byte_arr.seek(0)  # Rewind to the beginning of the byte stream
            images_jpeg_bytes.append(img_byte_arr.read())

        # Store the data
        tracks_xyz_cam.append(traj_3d_camera)
        visibility.append(visibility_mask)
        extrinsics.append(extrinsics_w2c)

        # Once we have NUM_FRAMES frames, save them
        if len(images_jpeg_bytes) >= NUM_FRAMES:
            # Check if intrinsics vary throughout the sequence
            fx_fy_cx_cy_array = np.array(fx_fy_cx_cy_list)
            max_variation = np.max(np.std(fx_fy_cx_cy_array, axis=0) / np.mean(fx_fy_cx_cy_array, axis=0))
            
            # If variation is more than 0.1% (relative standard deviation), skip this sequence
            if max_variation > 0.001:
                print(f"Skipping {seq} batch {i//NUM_FRAMES} - camera intrinsics vary too much (max variation: {max_variation:.6f})")
                # Clear the lists for the next batch
                images_jpeg_bytes.clear()
                tracks_xyz_cam.clear()
                visibility.clear()
                extrinsics.clear()
                fx_fy_cx_cy_list.clear()
                depth_maps.clear()
                continue
                
            # Save the current batch
            npz_filename = os.path.join(output_dir, f"{seq.split('/')[-2]}_{i//NUM_FRAMES}.npz")
            # print(tracks_xyz_cam[0].shape, fx_fy_cx_cy[0].shape, visibility[0].shape, extrinsics[0].shape)
            np.savez_compressed(npz_filename,
                                images_jpeg_bytes=images_jpeg_bytes,
                                tracks_XYZ=tracks_xyz_cam,
                                fx_fy_cx_cy=fx_fy_cx_cy_list[0],  # Use first frame's intrinsics
                                visibility=visibility,
                                depth_map=depth_maps,
                                extrinsics_w2c=extrinsics)
            print(f"Saved {npz_filename}")

            # Clear the lists for the next batch
            images_jpeg_bytes.clear()
            tracks_xyz_cam.clear()
            visibility.clear()
            extrinsics.clear()
            fx_fy_cx_cy_list.clear()
            depth_maps.clear()

def convert_dynamic_stereo_to_tapvid3d(dataset_location="data/dynamic_replica_data/test", output_dir="data/tapvid3d_dataset/dynamic_stereo_final"):
    # Get the sequence folders in DynamicStereo dataset
    seq_folders = glob.glob(os.path.join(dataset_location, "*/"))
    seq_folders.sort()

    if SUBSAMPLE_FACTOR > 1:
        print(f"Subsampling factor: {SUBSAMPLE_FACTOR}")
        output_dir += f"_subsample{SUBSAMPLE_FACTOR}"

    # Make dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # print(f"Found {len(seq_folders)} sequences, {seq_folders}")

    # Use concurrent.futures to process sequences in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for seq in seq_folders:
            if "right" not in seq:
                futures.append(executor.submit(process_dynamic_stereo_sequence, seq, output_dir, dataset_location))
        
        # Wait for all threads to finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # To ensure we catch any exceptions raised in threads

    #     # Use single thread for debugging
    # for seq in tqdm(seq_folders):
    #     if "right" not in seq:
    #         process_dynamic_stereo_sequence(seq, output_dir, dataset_location)

# Call the conversion function
convert_dynamic_stereo_to_tapvid3d()
