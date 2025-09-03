import glob
import os
import shutil
import numpy as np
import cv2
from PIL import Image
import io
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append("dust3r")
sys.path.append("../")
sys.path.append("./")
from dust3r.datasets.tapvid3d import load_npz_data_recon

NUM_FRAMES = 128
# TUM RGB-D camera intrinsics
fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y

def tx_ty_tz_qx_qy_qz_qw_to_extrinsic(tx, ty, tz, qx, qy, qz, qw):
    """Convert translation and quaternion to 4x4 extrinsic matrix."""
    rotation_matrix = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw, tx],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw, ty],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy, tz],
        [0, 0, 0, 1]
    ])
    return rotation_matrix

def process_sequence(seq_name, output_dir):
    """Process a single sequence from the TUM dataset."""
    rgb_dir = os.path.join(seq_name, "rgb_90")
    depth_dir = os.path.join(seq_name, "depth_90")
    pose_path = os.path.join(seq_name, "groundtruth_90.txt")

    if not (os.path.exists(pose_path) and os.path.exists(rgb_dir) and os.path.exists(depth_dir)):
        print(f"Missing data in {seq_name}")
        return

    # Get sorted lists of RGB and depth images
    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    # Load poses
    poses = np.loadtxt(pose_path)
    
    # Process frames in chunks of NUM_FRAMES
    num_frames = min(len(rgb_paths), len(depth_paths), len(poses))
    
    for i in range(0, num_frames, NUM_FRAMES):
        frame_end = min(i + NUM_FRAMES, num_frames)
        
        # Collect frames between i and frame_end
        images_jpeg_bytes = []
        depth_maps = []
        extrinsics = []
        visibility = []
        
        # Camera intrinsics (constant for TUM dataset)
        fx_fy_cx_cy = [fx, fy, cx, cy]

        for j in range(i, frame_end):
            # Load and process RGB image
            rgb_image = Image.open(rgb_paths[j])
            with io.BytesIO() as img_byte_arr:
                rgb_image.save(img_byte_arr, format="JPEG")
                img_byte_arr.seek(0)
                images_jpeg_bytes.append(img_byte_arr.read())

            # Load and process depth image
            depth_image = cv2.imread(depth_paths[j], cv2.IMREAD_ANYDEPTH)
            depth_map = depth_image.astype(np.float32) / 5000.0  # Convert to meters
            depth_maps.append(depth_map)

            # Convert pose to extrinsic matrix
            pose = poses[j]
            extrinsic = tx_ty_tz_qx_qy_qz_qw_to_extrinsic(
                pose[1], pose[2], pose[3],  # translation
                pose[4], pose[5], pose[6], pose[7]  # rotation (quaternion)
            )
            extrinsics.append(np.linalg.inv(extrinsic))

            # where depth_map is not 0, set visibility to True, otherwise False
            visibility.append(depth_map > 0)

        # Save as .npz
        seq_basename = os.path.basename(os.path.normpath(seq_name))
        npz_filename = os.path.join(output_dir, f"{seq_basename}_{i//NUM_FRAMES}.npz")
        
        np.savez_compressed(
            npz_filename,
            images_jpeg_bytes=images_jpeg_bytes,
            depth_map=depth_maps,
            fx_fy_cx_cy=fx_fy_cx_cy,
            visibility=visibility,
            extrinsics_w2c=extrinsics
        )
        print(f"Saved {npz_filename}")

def convert_tum_to_tapvid3d(dataset_location="data/tum/*/", output_dir="data/tapvid3d_dataset/tum"):
    """Convert TUM RGB-D dataset to TapVid3D format."""
    # Get sequence folders
    seq_folders = glob.glob(dataset_location)
    seq_folders.sort()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(seq_folders)} sequences")

    # Process sequences in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for seq in seq_folders:
            futures.append(executor.submit(process_sequence, seq, output_dir))
        
        # Wait for all threads to finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # Catch any exceptions raised in threads

if __name__ == "__main__":
    convert_tum_to_tapvid3d()
    # npz_path = "data/tapvid3d_dataset/tum/rgbd_dataset_freiburg3_sitting_xyz_0.npz"  # adjust path as needed
    # video_list, depth_map, intrinsics, recon_xyz_world, recon_xyz_cam, visibility, video_name, extrinsics_w2c = load_npz_data_recon(npz_path)
    
    # # Create directory if it doesn't exist
    # os.makedirs("tmp_tum", exist_ok=True)

    # # Save images
    # frames_to_save = [0, 5, 10]
    # for i in frames_to_save:
    #     video_list[i].save(f"tmp_tum/{video_name}_{i}.png")

    # # Save all pointclouds in one PLY file with colors
    # ply_file = f"tmp_tum/{video_name}_all.ply"
    # total_points = sum(visibility[i].sum() for i in frames_to_save)
    
    # with open(ply_file, "w") as f:
    #     # Write header
    #     f.write("ply\n")
    #     f.write("format ascii 1.0\n")
    #     f.write("comment Created by dust3r\n")
    #     f.write(f"element vertex {total_points}\n")
    #     f.write("property float x\n")
    #     f.write("property float y\n")
    #     f.write("property float z\n")
    #     f.write("property uchar red\n")
    #     f.write("property uchar green\n")
    #     f.write("property uchar blue\n")
    #     f.write("end_header\n")

    #     # Write points with colors
    #     for i in frames_to_save:
    #         # Get RGB colors from the image
    #         img_array = np.array(video_list[i])
            
    #         # Get valid points
    #         valid_mask = visibility[i]
    #         points = recon_xyz_world[i][valid_mask]
    #         colors = img_array[valid_mask]

    #         # Write points and their colors
    #         for (x, y, z), (r, g, b) in zip(points, colors):
    #             f.write(f"{x} {y} {z} {r} {g} {b}\n")

    # print(f"Saved combined pointcloud to {ply_file}")