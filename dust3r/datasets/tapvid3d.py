import os
import numpy as np
import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import mediapy as media
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

from dust3r.viz import SceneViz



def visualize_results_recon(imgs, gt_tracks_world, pred_tracks_world, save_path=None, frame_stride=4, spatial_stride=4):
    """
    Visualize reconstruction results with both 2D frames and 3D pointclouds.
    Combines multiple frames in a single visualization.
    
    Args:
        imgs: list of T PIL.Image.Image
        gt_tracks_world: T, H, W, 3 torch.Tensor or numpy array
        pred_tracks_world: T, H, W, 3 torch.Tensor or numpy array
        save_path: str, path to save visualization results
    """
    # TODO(jz): temporally for generating paper visualizations
    spatial_stride = 1
    frame_stride = 8
    # Create output directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Convert tensors to numpy if needed
    if torch.is_tensor(gt_tracks_world):
        gt_tracks_world = gt_tracks_world.detach().cpu().numpy()
    if torch.is_tensor(pred_tracks_world):
        pred_tracks_world = pred_tracks_world.detach().cpu().numpy()

    gt_tracks_world = gt_tracks_world[:, ::spatial_stride, ::spatial_stride, :]
    pred_tracks_world = pred_tracks_world[:, ::spatial_stride, ::spatial_stride, :]
    
    # Create Y-axis flip transformation matrix
    y_flip_transform = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Apply Y-axis flip to all point clouds
    for i in range(gt_tracks_world.shape[0]):
        # Apply transformation to GT point cloud
        valid_mask = np.linalg.norm(gt_tracks_world[i], axis=-1) > 0
        gt_tracks_world[i, valid_mask] = gt_tracks_world[i, valid_mask] @ y_flip_transform
        
        # Apply transformation to predicted point cloud
        valid_mask = np.linalg.norm(pred_tracks_world[i], axis=-1) > 0
        pred_tracks_world[i, valid_mask] = pred_tracks_world[i, valid_mask] @ y_flip_transform

    # Save RGB frames
    for i, img in enumerate(imgs):
        img_save_path = os.path.join(save_path, f'frame_{i:03d}.png')
        img.save(img_save_path)

    # Define frames to visualize (e.g., every 4th frame)
    frames_to_viz = list(range(0, len(imgs), frame_stride))
    
    # Initialize visualizers for different views
    viz_combined = SceneViz()  # Combined GT and pred with tinted colors
    viz_gt_only = SceneViz()   # GT points only
    viz_pred_only = SceneViz() # Pred points only
    
    # Create rainbow colormap for different frames
    cmap = plt.get_cmap('rainbow')
    frame_colors = [cmap(i / max(1, len(frames_to_viz)-1))[:3] for i in range(len(frames_to_viz))]
    
    for idx, frame_idx in enumerate(frames_to_viz):
        # Get current frame data
        frame_img = imgs[frame_idx]
        gt_pts3d = gt_tracks_world[frame_idx]
        pred_pts3d = pred_tracks_world[frame_idx]
        
        # Convert PIL image to numpy array and normalize to [0,1]
        rgb_colors = (np.array(frame_img) / 255.0)[::spatial_stride, ::spatial_stride, :]
        
        # Create valid masks (non-zero points)
        gt_valid_mask = np.linalg.norm(gt_pts3d, axis=-1) > 0
        pred_valid_mask = np.linalg.norm(pred_pts3d, axis=-1) > 0

        # Create frame-specific colors
        frame_color = np.array(frame_colors[idx])
        
        # Create tinted colors for GT (blue tint) and pred (red tint)
        gt_tinted_color = rgb_colors.copy()
        pred_tinted_color = rgb_colors.copy()
        
        # Add blue tint to GT points (increase blue channel, reduce others slightly)
        gt_tinted_color[..., 2] = np.minimum(1.0, gt_tinted_color[..., 2] * 1.3)  # Boost blue
        gt_tinted_color[..., 0] = gt_tinted_color[..., 0] * 0.8  # Reduce red
        
        # Add red tint to pred points (increase red channel, reduce others slightly)
        pred_tinted_color[..., 0] = np.minimum(1.0, pred_tinted_color[..., 0] * 1.3)  # Boost red
        pred_tinted_color[..., 2] = pred_tinted_color[..., 2] * 0.8  # Reduce blue
        
        # Add to combined visualization with tinted colors
        viz_combined.add_pointcloud(gt_pts3d, gt_tinted_color, gt_valid_mask)
        viz_combined.add_pointcloud(pred_pts3d, pred_tinted_color, pred_valid_mask)
        
        # Add to GT-only visualization
        viz_gt_only.add_pointcloud(gt_pts3d, rgb_colors, gt_valid_mask)
        
        # Add to pred-only visualization
        viz_pred_only.add_pointcloud(pred_pts3d, rgb_colors, pred_valid_mask)

    # Save all visualizations
    glb_path_combined = os.path.join(save_path, 'pointcloud_all_frames_tinted.glb')
    viz_combined.save_glb(glb_path_combined)
    
    glb_path_gt = os.path.join(save_path, 'pointcloud_all_frames_gt_only.glb')
    viz_gt_only.save_glb(glb_path_gt)
    
    glb_path_pred = os.path.join(save_path, 'pointcloud_all_frames_pred_only.glb')
    viz_pred_only.save_glb(glb_path_pred)

    print(f"Saved visualizations to {save_path}")
    return glb_path_combined  # Return the path to the main visualization

def visualize_results(imgs, gt_tracks_world, pred_tracks_world, extrinsics, intrinsics, save_path=None):
    """
    Visualizes the results and saves two videos:
    1. 2D track visualization (on video frames)
    2. 3D track visualization (in world coordinates)
    
    Args:
    - imgs (list of T PIL.Image.Image): The video frames.
    - gt_tracks_world (T, N, 3 np.array): Ground truth tracks in world coordinates.
    - pred_tracks_world (T, N, 3 np.array): Predicted tracks in world coordinates.
    - extrinsics (T, 4, 4 np.array): Camera extrinsics (world to camera).
    - intrinsics (4,): Camera intrinsic parameters (fx, fy, cx, cy).
    - save_path (str): If provided, save the video to this path.
    """
    # make directory if it doesn't exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Randomly sample 300 points if there are more than 300
    if gt_tracks_world.shape[1] > 300:
        random_indices = np.random.choice(gt_tracks_world.shape[1], 300, replace=False)
    else:
        random_indices = np.arange(gt_tracks_world.shape[1])

    first_frame_y = gt_tracks_world[0, :, 1]  # y-values of the first frame
    random_indices = random_indices[first_frame_y[random_indices].argsort()]  # sort by y-values
    gt_tracks_world = gt_tracks_world[:, random_indices]
    pred_tracks_world = pred_tracks_world[:, random_indices]

    # 1. Project 3D GT and Pred tracks to 2D using extrinsics and intrinsics
    gt_tracks_2d = project_points_to_2d(gt_tracks_world, extrinsics, intrinsics)
    pred_tracks_2d = project_points_to_2d(pred_tracks_world, extrinsics, intrinsics)

    # 2. Visualize 2D Tracks on video frames
    visualize_2d_tracks_on_video(imgs, gt_tracks_2d, pred_tracks_2d, save_path)

    # 3. Visualize 3D Tracks in world coordinates (GT and Pred side-by-side)
    video3d_viz_combined = plot_3d_tracks(gt_tracks_world, pred_tracks_world)

    if save_path:
        media.write_video(save_path + '/3d_tracks.mp4', video3d_viz_combined, fps=15)
    else:
        media.show_video(video3d_viz_combined, fps=15)

def project_points_to_2d(points_3d, extrinsics, intrinsics):
    """
    Project points from world coordinates to 2D image plane using camera extrinsics and intrinsics.
    
    Args:
    - points_3d (T, N, 3 np.array): 3D coordinates of tracks in world coordinates.
    - extrinsics (T, 4, 4 np.array): Camera extrinsics (world to camera transformation).
    - intrinsics (4,): Camera intrinsic parameters (fx, fy, cx, cy).
    
    Returns:
    - points_2d (T, N, 2 np.array): 2D coordinates of tracks in image plane.
    """
    num_frames, num_points = points_3d.shape[:2]
    points_2d = np.zeros((num_frames, num_points, 2))

    for t in range(num_frames):
        R = extrinsics[t, :3, :3]
        tvec = extrinsics[t, :3, 3]

        # Project points to camera space
        points_camera = (points_3d[t] @ R.T) + tvec  # (N, 3)

        # Project to 2D using intrinsics
        u_d = points_camera[:, 0] / (points_camera[:, 2] + 1e-8)
        v_d = points_camera[:, 1] / (points_camera[:, 2] + 1e-8)
        f_u, f_v, c_u, c_v = intrinsics
        u_d = u_d * f_u + c_u
        v_d = v_d * f_v + c_v

        points_2d[t] = np.stack([u_d, v_d], axis=-1)
    
    return points_2d

def visualize_2d_tracks_on_video(imgs, gt_tracks_2d, pred_tracks_2d, save_path=None, tracks_leave_trace=8):
    """
    Visualizes and saves the 2D track visualization on the video frames with alpha blending for track traces.

    Args:
    - imgs (list of T PIL.Image.Image): The video frames.
    - gt_tracks_2d (T, N, 2 np.array): Ground truth tracks in 2D.
    - pred_tracks_2d (T, N, 2 np.array): Predicted tracks in 2D.
    - tracks_leave_trace (int): Number of frames to leave traces of points.
    - save_path (str): Path to save the video.
    
    Returns:
    - None
    """
    num_frames, num_points = gt_tracks_2d.shape[:2]

    # Convert colors to be between 0-255 and use the HSV colormap
    color_map = plt.cm.hsv  # Use matplotlib's color map
    cmap_norm = plt.Normalize(vmin=0, vmax=num_points - 1)

    frames_2d = []
    for t in range(num_frames):
        img = np.array(imgs[t].convert("RGB"))
        frame = img.copy()

        # Plot GT tracks with circles (filled)
        for i in range(num_points):
            if np.linalg.norm(gt_tracks_2d[t, i]) > 1e-8:  # Avoid invalid points
                x, y = int(round(gt_tracks_2d[t, i, 0])), int(round(gt_tracks_2d[t, i, 1]))
                color = color_map(cmap_norm(i))[:3]
                color = [int(255 * c) for c in color]
                cv2.circle(frame, (x, y), 3, tuple(color), -1)

            if np.linalg.norm(pred_tracks_2d[t, i]) > 1e-8:  # Avoid invalid points
                x, y = int(round(pred_tracks_2d[t, i, 0])), int(round(pred_tracks_2d[t, i, 1]))
                cv2.drawMarker(frame, (x, y), tuple(color), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

            # Plot track traces with alpha blending
            trace_gt_points = gt_tracks_2d[max(0, t - tracks_leave_trace): t + 1, i]
            trace_pred_points = pred_tracks_2d[max(0, t - tracks_leave_trace): t + 1, i]
            
            for s in range(len(trace_gt_points) - 1):
                # Alpha blending for GT track lines
                if np.linalg.norm(trace_gt_points[s]) > 1e-8 and np.linalg.norm(trace_gt_points[s + 1]) > 1e-8:
                    x1, y1 = int(round(trace_gt_points[s, 0])), int(round(trace_gt_points[s, 1]))
                    x2, y2 = int(round(trace_gt_points[s + 1, 0])), int(round(trace_gt_points[s + 1, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), tuple(color), 1, cv2.LINE_AA)

                # Alpha blending for Pred track lines
                if np.linalg.norm(trace_pred_points[s]) > 1e-8 and np.linalg.norm(trace_pred_points[s + 1]) > 1e-8:
                    x1, y1 = int(round(trace_pred_points[s, 0])), int(round(trace_pred_points[s, 1]))
                    x2, y2 = int(round(trace_pred_points[s + 1, 0])), int(round(trace_pred_points[s + 1, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), tuple(color), 1, cv2.LINE_AA)

        frames_2d.append(frame)

    frames_2d = np.stack(frames_2d)
    if save_path:
        media.write_video(save_path + '/2d_tracks.mp4', frames_2d, fps=15)
    else:
        media.show_video(frames_2d, fps=15)

def plot_3d_tracks(gt_points, pred_points, tracks_leave_trace=32, max_points=100, marker_size=30, line_width=2):
    """
    Visualizes 3D point trajectories for both GT and Pred tracks.

    Args:
    - gt_points (T, N, 3 np.array): Ground truth 3D points (world coordinates).
    - pred_points (T, N, 3 np.array): Predicted 3D points (world coordinates).
    - tracks_leave_trace (int): Number of frames to leave traces of points.
    - max_points (int): Maximum number of points to visualize. If None, use all points.
    - marker_size (int): Size of the markers for points.
    - line_width (int): Width of the trajectory lines.

    Returns:
    - frames (T, H, W, 3 np.array): List of frames for 3D track visualization.
    """
    num_frames, num_points = gt_points.shape[0:2]
    
    # If number of points exceeds maximum, perform downsampling
    if max_points is not None and num_points > max_points:
        # Randomly select indices for max_points points
        selected_indices = np.random.choice(num_points, max_points, replace=False)
        selected_indices.sort()  # Sort to maintain consistency
        
        # Extract selected points
        gt_points = gt_points[:, selected_indices]
        pred_points = pred_points[:, selected_indices]
        num_points = max_points
    
    color_map = plt.cm.hsv  # Use matplotlib's color map
    cmap_norm = plt.Normalize(vmin=0, vmax=num_points - 1)

    # Set axis ranges for consistent scaling
    x_min, x_max = np.min(gt_points[..., 0]), np.max(gt_points[..., 0])
    y_min, y_max = np.min(gt_points[..., 2]), np.max(gt_points[..., 2])
    z_min, z_max = np.min(gt_points[..., 1]), np.max(gt_points[..., 1])

    # Increase interval to make points more dispersed
    interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    padding = interval * 0.2  # Add 20% padding
    
    x_min = (x_min + x_max) / 2 - interval / 2 - padding
    x_max = x_min + interval + 2 * padding
    y_min = (y_min + y_max) / 2 - interval / 2 - padding
    y_max = y_min + interval + 2 * padding
    z_min = (z_min + z_max) / 2 - interval / 2 - padding
    z_max = z_min + interval + 2 * padding

    frames = []
    for t in tqdm(range(num_frames)):
        # Increase figure size
        fig = plt.figure(figsize=(12, 9))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d')

        # Set larger axis ranges
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        # set the background color to white
        ax.set_facecolor('white')

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Set larger tick intervals to reduce number of ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        # Adjust viewing angle for better 3D effect
        ax.view_init(elev=10, azim=-80)
        
        # Invert Z-axis
        ax.invert_zaxis()

        # Draw reference grid with higher transparency
        # ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.2)})
        # ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.2)})
        # ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 0.2)})

        for i in range(num_points):
            # Draw GT tracks with circles
            gt_line = gt_points[max(0, t - tracks_leave_trace): t + 1, i]
            color = color_map(cmap_norm(i))
            
            # Only draw trajectory lines when there are enough points
            if len(gt_line) > 1:
                ax.plot(xs=gt_line[:, 0], ys=gt_line[:, 2], zs=gt_line[:, 1], 
                        color=color, linewidth=line_width, alpha=0.7)
            
            # Increase point size
            ax.scatter(xs=gt_line[-1, 0], ys=gt_line[-1, 2], zs=gt_line[-1, 1], 
                      color=color, s=marker_size, marker='o', edgecolors='black', linewidth=0.5)

            # Draw Pred tracks with crosses
            pred_line = pred_points[max(0, t - tracks_leave_trace): t + 1, i]
            
            # Only draw trajectory lines when there are enough points
            if len(pred_line) > 1:
                ax.plot(xs=pred_line[:, 0], ys=pred_line[:, 2], zs=pred_line[:, 1], 
                       color=color, linewidth=line_width, linestyle='dotted', alpha=0.7)
            
            # Increase point size, use X marker
            ax.scatter(xs=pred_line[-1, 0], ys=pred_line[-1, 2], zs=pred_line[-1, 1], 
                      color=color, s=marker_size*1.2, marker='x', linewidth=1.5)

        # Adjust subplot position, reduce margins
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        canvas.draw()

        # Get the frame as RGBA and convert to RGB
        rgba = np.asarray(canvas.buffer_rgba())
        rgb = rgba[..., :3]  # Remove the alpha channel
        frames.append(rgb)

    return np.array(frames)

def load_npz_data(npz_path, num_frames=None, normalize_cam=True):
    """
    Loads a .npz file containing:
      - images_jpeg_bytes: compressed frames
      - tracks_XYZ: shape (T, N, 3), camera-space 3D points
      - fx_fy_cx_cy: intrinsics (4,)
      - extrinsics_w2c (optional): shape (T, 4, 4) for world-to-camera
      - visibility: shape (T, N), boolean
      - plus any other custom fields

    Args:
      npz_path (str): Path to the .npz file
      num_frames (int | None): If given, only load up to the first num_frames
      normalize_cam (bool): If True and extrinsics are present, multiply all
          extrinsics by the inverse of the first frame's extrinsic so that 
          the first frame extrinsic becomes identity in W2C space.

    Returns:
      video_list: list of PIL images
      tracks_xyz_cam: (T, N, 3) array in camera space
      tracks_uv: (T, N, 2) array of projected 2D coordinates
      intrinsics: (4,) array [fx, fy, cx, cy]
      tracks_xyz_world: (T, N, 3) array in world space (if extrinsics are present,
                        otherwise same as camera space)
      visibility: (T, N) boolean array
      video_name: string extracted from filename
      extrinsics_w2c: (T, 4, 4) world-to-camera (potentially normalized), or None
    """
    in_npz = np.load(npz_path, allow_pickle=True)

    images_jpeg_bytes = in_npz["images_jpeg_bytes"]
    tracks_xyz_cam = in_npz["tracks_XYZ"]
    intrinsics = in_npz["fx_fy_cx_cy"]
    visibility = in_npz["visibility"]

    # Optional extrinsics
    if 'extrinsics_w2c' in in_npz.files:
        extrinsics_w2c = in_npz['extrinsics_w2c']
    else:
        extrinsics_w2c = None
    print(f"Loaded {len(images_jpeg_bytes)} frames from {npz_path}, subsampled to {num_frames}")
    # If num_frames is set, slice everything
    if num_frames is not None:
        images_jpeg_bytes = images_jpeg_bytes[:num_frames]
        tracks_xyz_cam = tracks_xyz_cam[:num_frames]       # shape (T, N, 3)
        visibility = visibility[:num_frames]               # shape (T, N)
        if extrinsics_w2c is not None:
            extrinsics_w2c = extrinsics_w2c[:num_frames]   # shape (T, 4, 4)
    
    # Decode the images into PIL format
    video_list = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        video_list.append(pil_image)

    # Project 3D camera coords into 2D frame coords
    # (Shape checks: video_list -> T frames; image height -> video_list[0].height, etc.)
    # Example usage of project_points_to_video_frame:
    #   tracks_uv, _ = project_points_to_video_frame(
    #       tracks_xyz_cam, intrinsics,
    #       img_height=video_list[0].height,
    #       img_width=video_list[0].width
    #   )
    h, w = video_list[0].height, video_list[0].width
    tracks_uv, _ = project_points_to_video_frame(
        tracks_xyz_cam, intrinsics, h, w
    )

    # Normalize camera extrinsics if requested
    if normalize_cam and (extrinsics_w2c is not None):
        # We'll left-multiply each extrinsic by inv(extrinsics_w2c[0]),
        # so that the first frame becomes identity.
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        for i in range(extrinsics_w2c.shape[0]):
            extrinsics_w2c[i] = extrinsics_w2c[i] @ first_inv

    # If extrinsics_w2c is available, transform tracks to world coords
    if extrinsics_w2c is not None:
        # Convert W2C to C2W for each frame
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
        tracks_xyz_world = np.zeros_like(tracks_xyz_cam)  # (T, N, 3)
        for i in range(tracks_xyz_cam.shape[0]):
            R = extrinsics_c2w[i, :3, :3]
            t = extrinsics_c2w[i, :3, 3]
            # (R @ cam_coords^T)^T + t
            tracks_xyz_world[i] = (R @ tracks_xyz_cam[i].T).T + t
    else:
        # If no extrinsics, world == cam coords
        tracks_xyz_world = tracks_xyz_cam
        extrinsics_w2c = np.tile(np.eye(4), (num_frames, 1, 1))

    # Get video name from the file path
    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    return (
        video_list,
        tracks_xyz_cam,
        tracks_uv,
        intrinsics,
        tracks_xyz_world,
        visibility,
        video_name,
        extrinsics_w2c
    )

def load_npz_data_recon(npz_path, num_frames=None, normalize_cam=True):

    in_npz = np.load(npz_path, allow_pickle=True)
    images_jpeg_bytes = in_npz["images_jpeg_bytes"]
    depth_map = in_npz["depth_map"]
    intrinsics = in_npz["fx_fy_cx_cy"]
    visibility = in_npz["visibility"]

    # Optional extrinsics
    if 'extrinsics_w2c' in in_npz.files:
        extrinsics_w2c = in_npz['extrinsics_w2c']
    else:
        extrinsics_w2c = None
    print(f"Loaded {len(images_jpeg_bytes)} frames from {npz_path}, subsampled to {num_frames}")
    # If num_frames is set, slice everything
    if num_frames is None:
        num_frames = len(images_jpeg_bytes)
    images_jpeg_bytes = images_jpeg_bytes[:num_frames]
    depth_map = depth_map[:num_frames]
    visibility = visibility[:num_frames]               # shape (T, N)
    if extrinsics_w2c is not None:
        extrinsics_w2c = extrinsics_w2c[:num_frames]   # shape (T, 4, 4)
    
    # Decode the images into PIL format
    video_list = []
    for frame_bytes in images_jpeg_bytes:
        arr = np.frombuffer(frame_bytes, np.uint8)
        image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        video_list.append(pil_image)

    h, w = video_list[0].height, video_list[0].width

    # Normalize camera extrinsics if requested
    if normalize_cam and (extrinsics_w2c is not None):
        # We'll left-multiply each extrinsic by inv(extrinsics_w2c[0]),
        # so that the first frame becomes identity.
        first_inv = np.linalg.inv(extrinsics_w2c[0])
        for i in range(extrinsics_w2c.shape[0]):
            extrinsics_w2c[i] = extrinsics_w2c[i] @ first_inv

    extrinsics_c2w = np.linalg.inv(extrinsics_w2c)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map

    x = (u - intrinsics[2]) * z / intrinsics[0]
    y = (v - intrinsics[3]) * z / intrinsics[1]
    recon_xyz_cam = np.stack([x, y, z], axis=-1)
    recon_xyz_world = np.zeros_like(recon_xyz_cam)  # (T, N, 3)
    # If extrinsics_w2c is available, transform tracks to world coords
    if extrinsics_w2c is not None:
        # Convert W2C to C2W for each frame
        for i in range(recon_xyz_cam.shape[0]):
            R = extrinsics_c2w[i, :3, :3]
            t = extrinsics_c2w[i, :3, 3]
            # (R @ cam_coords^T)^T + t
        
            recon_xyz_world_flat = (R @ recon_xyz_cam[i].reshape(-1,3).T).T + t
            recon_xyz_world[i] = recon_xyz_world_flat.reshape(h, w, 3)
    else:
        recon_xyz_world = recon_xyz_cam

    # Get video name from the file path
    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    return (
        video_list,
        depth_map,
        intrinsics,
        recon_xyz_world,
        recon_xyz_cam,
        visibility,
        video_name,
        extrinsics_w2c
    )

def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
    """Project 3d points to 2d image plane."""
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

    f_u, f_v, c_u, c_v = camera_intrinsics

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Mask of points that are in front of the camera and within image boundary
    masks = camera_pov_points3d[..., 2] >= 1
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks
