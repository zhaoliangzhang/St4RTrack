import numpy as np
import torch
import os
import glob
import cv2
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import io

NUM_FRAMES = 128
SUBSAMPLE_FACTOR = 8

def process_sequence(seq, output_dir):
    rgb_paths = sorted(glob.glob(os.path.join(seq, "rgbs/*.jpg")))
    depth_paths = sorted(glob.glob(os.path.join(seq, "depths/*.png")))
    annotations_path = os.path.join(seq, "anno.npz")

    # Skip sequences with fewer than NUM_FRAMES frames
    if len(rgb_paths) < NUM_FRAMES:
        print(f"Skipping {seq} - insufficient frames ({len(rgb_paths)} < {NUM_FRAMES})")
        return

    # Load annotations
    annotations = np.load(annotations_path, allow_pickle=True)
    trajs_3d = annotations['trajs_3d']
    traj_visibs = annotations['visibs']
    extrinsics_w2c = annotations.get('extrinsics', None)
    intrinsics = annotations['intrinsics']

    if SUBSAMPLE_FACTOR > 1:
        # print(trajs_3d.shape, traj_visibs.shape)
        trajs_3d = trajs_3d[:,::SUBSAMPLE_FACTOR]
        traj_visibs = traj_visibs[:,::SUBSAMPLE_FACTOR]

    # Prepare the frames to save every 128 frames
    num_frames = len(rgb_paths)
    for i in range(0, num_frames, NUM_FRAMES):
        frame_end = min(i + NUM_FRAMES, num_frames)
        
        # Skip if we don't have a full batch of NUM_FRAMES
        if frame_end - i < NUM_FRAMES:
            print(f"Skipping partial batch at end of {seq} ({frame_end - i} < {NUM_FRAMES})")
            continue

        # Collect frames between i and frame_end
        images_jpeg_bytes = []
        tracks_xyz_cam = []
        visibility = []
        extrinsics = []
        depth_maps = []
        fx_fy_cx_cy_list = []

        # First collect all intrinsics to check for variation
        for j in range(i, frame_end):
            fx_fy_cx_cy_list.append([
                intrinsics[j, 0, 0], 
                intrinsics[j, 1, 1], 
                intrinsics[j, 0, 2], 
                intrinsics[j, 1, 2]
            ])
        
        # Check if intrinsics vary throughout the sequence
        fx_fy_cx_cy_array = np.array(fx_fy_cx_cy_list)
        max_variation = np.max(np.std(fx_fy_cx_cy_array, axis=0) / np.mean(fx_fy_cx_cy_array, axis=0))
        
        # If variation is more than 0.1% (relative standard deviation), skip this sequence
        if max_variation > 0.001:
            print(f"Skipping {seq} batch {i//NUM_FRAMES} - camera intrinsics vary too much (max variation: {max_variation:.6f})")
            continue
        
        # Use the first frame's intrinsics for the entire sequence
        fx_fy_cx_cy = fx_fy_cx_cy_list[0]

        # Now process the frames
        for j in range(i, frame_end):
            rgb_image = Image.open(rgb_paths[j])
            depth_image = cv2.imread(depth_paths[j], cv2.IMREAD_ANYDEPTH)
            depth_map = depth_image.astype(np.float32) / 65535.0 * 1000.0
            depth_maps.append(depth_map)
            
            # Convert 3D trajectory to camera coordinates
            traj_3d_proj = trajs_3d[j]
            traj_3d_proj_homogeneous = np.hstack((traj_3d_proj, np.ones((traj_3d_proj.shape[0], 1))))  # Add 1 for homogeneous coordinates
            traj_3d_proj_camera_homogeneous = np.dot(extrinsics_w2c[j], traj_3d_proj_homogeneous.T).T  # Apply extrinsics
            traj_3d_proj_camera = traj_3d_proj_camera_homogeneous[:, :3]  # Remove homogeneous coordinate

            # Project the 3D points to 2D
            tracks_xyz_cam.append(traj_3d_proj_camera)
            visibility.append(traj_visibs[j])
            extrinsics.append(extrinsics_w2c[j])

            # Save image as JPEG byte array
            with io.BytesIO() as img_byte_arr:
                rgb_image.save(img_byte_arr, format="JPEG")
                img_byte_arr.seek(0)  # Rewind to the beginning of the byte stream
                images_jpeg_bytes.append(img_byte_arr.read())

        # Save as .npz
        npz_filename = os.path.join(output_dir, f"{seq.split('/')[-2]}_{i//NUM_FRAMES}.npz")
        np.savez_compressed(npz_filename,
                            images_jpeg_bytes=images_jpeg_bytes,
                            tracks_XYZ=tracks_xyz_cam,
                            fx_fy_cx_cy=fx_fy_cx_cy,
                            visibility=visibility,
                            depth_map=depth_maps,
                            extrinsics_w2c=extrinsics)
        print(f"Saved {npz_filename}")

def convert_pointodyssey_to_tapvid3d(dataset_location="data/point_odyssey", output_dir="data/tapvid3d_dataset/po_final"):
    # Get the sequence folders in PointOdyssey dataset
    seq_folders = glob.glob(os.path.join(dataset_location, "test/*/"))
    seq_folders.sort()

    if SUBSAMPLE_FACTOR > 1:
        output_dir += f"_subsample{SUBSAMPLE_FACTOR}"

    # Make dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Found {len(seq_folders)} sequences")

    # Use concurrent.futures to process sequences in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for seq in seq_folders:
            futures.append(executor.submit(process_sequence, seq, output_dir))
        
        # Wait for all threads to finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # To ensure we catch any exceptions raised in threads

# Call the conversion function
convert_pointodyssey_to_tapvid3d()
