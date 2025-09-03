"""St4RTrack 3D Visualizer

Interactive web-based visualization of 4D reconstruction and tracking results.

example usage: 
python visualizer_st4rtrack.py --num_traj_points 500 --max_frames 128 --point_size 0.002 \
--traj_path 
# --mask_folder ./Davis/davis_mask/car-shadow \

"""

import time
from pathlib import Path

import numpy as onp
import tyro
import cv2
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

from glob import glob
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
import psutil

def log_memory_usage(message=""):
    """Log current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    print(f"Memory usage {message}: {memory_mb:.2f} MB")

def visualize_st4rtrack(
    traj_path: str = "results",
    up_dir: str = "-z", # should be +z or -z
    max_frames: int = 100,
    share: bool = False,
    point_size: float = 0.002,
    downsample_factor: int = 1,
    num_traj_points: int = 100,
    conf_thre_percentile: float = 0,
    traj_end_frame: int = 100,
    traj_start_frame: int = 0,
    traj_line_width: float = 3.,
    fixed_length_traj: int = 10,
    server: viser.ViserServer = None,
    use_float16: bool = True,
    preloaded_data: dict = None,
    color_code: str = "jet",
    blue_rgb: tuple[float, float, float] = (0.0, 0.149, 0.463),
    red_rgb: tuple[float, float, float] = (0.769, 0.510, 0.055),
    blend_ratio: float = 0.7,
    mask_folder: str = None,
) -> None:
    log_memory_usage("at start of visualization")
    
    if server is None:
        server = viser.ViserServer()
    if share:
        server.request_share_url()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1e-3, 0.75, -0.1)
        client.camera.look_at = (0, 0, 0)

    # Configure the GUI panel size and layout
    server.gui.configure_theme(
        control_layout="collapsible",
        control_width="small",
        dark_mode=False,
        show_logo=False,
        show_share_button=True
    )

    server.scene.set_up_direction(up_dir)
    print("Setting up visualization!")

    # Use preloaded data if available
    if preloaded_data and preloaded_data['loaded']:
        traj_3d_head1 = preloaded_data['traj_3d_head1']
        traj_3d_head2 = preloaded_data['traj_3d_head2']
        conf_mask_head1 = preloaded_data['conf_mask_head1']
        conf_mask_head2 = preloaded_data['conf_mask_head2']
        masks = preloaded_data['masks']
        print("Using preloaded data!")
    else:
        # Original data loading code (as a fallback)
        print("No preloaded data available, loading from files...")
        # Load both head1 and head2 data
        traj_3d_head1 = None
        traj_3d_head2 = None
        conf_mask_head1 = None
        conf_mask_head2 = None
        masks = None
        if mask_folder is not None:
            masks_paths = sorted(glob(mask_folder + '/*.jpg'))
            masks = [iio.imread(p) for p in masks_paths]
            masks = np.stack(masks, axis=0)
            masks = (masks < 1).astype(np.float32)
            masks = masks.sum(axis=-1) > 2
            print(f"Original masks shape: {masks.shape}")
        
        if Path(traj_path).is_dir():
            # Load head1 data
            traj_3d_paths_head1 = sorted(glob(traj_path + '/pts3d1_p*.npy'), 
                                       key=lambda x: int(x.split('_p')[-1].split('.')[0]))
            conf_paths_head1 = sorted(glob(traj_path + '/conf1_p*.npy'), 
                                    key=lambda x: int(x.split('_p')[-1].split('.')[0]))
            
            # Load head2 data
            traj_3d_paths_head2 = sorted(glob(traj_path + '/pts3d2_p*.npy'), 
                                       key=lambda x: int(x.split('_p')[-1].split('.')[0]))
            conf_paths_head2 = sorted(glob(traj_path + '/conf2_p*.npy'), 
                                    key=lambda x: int(x.split('_p')[-1].split('.')[0]))

            # Process head1
            if traj_3d_paths_head1:

                if use_float16:
                    traj_3d_head1 = onp.stack([onp.load(p).astype(onp.float16) for p in traj_3d_paths_head1], axis=0)
                else:
                    traj_3d_head1 = onp.stack([onp.load(p) for p in traj_3d_paths_head1], axis=0)

                log_memory_usage("after loading head1 data")
                
                h, w, _ = traj_3d_head1.shape[1:]
                num_frames = traj_3d_head1.shape[0]

                # If masks is None, create default masks (all ones)
                if masks is None:
                    masks = np.ones((num_frames, h, w), dtype=bool)
                    print(f"Created default masks with shape: {masks.shape}")
                else:
                    masks_resized = np.zeros((masks.shape[0], h, w), dtype=bool)
                    for i in range(masks.shape[0]):
                        masks_resized[i] = cv2.resize(
                            masks[i].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    print(f"Resized masks shape: {masks_resized.shape}")
                    masks = masks_resized
                
                # Reshape trajectory data
                traj_3d_head1 = traj_3d_head1.reshape(traj_3d_head1.shape[0], -1, 6)
                
                if conf_paths_head1:
                    conf_head1 = onp.stack([onp.load(p).astype(onp.float16) for p in conf_paths_head1], axis=0)
                    conf_head1 = conf_head1.reshape(conf_head1.shape[0], -1)
                    conf_head1 = conf_head1.mean(axis=0)
                    conf_head1 = np.tile(conf_head1, (num_frames, 1))
                    conf_thre = np.percentile(conf_head1.astype(np.float32), conf_thre_percentile)
                    conf_mask_head1 = conf_head1 > conf_thre

            # Process head2
            if traj_3d_paths_head2:
                if use_float16:
                    traj_3d_head2 = onp.stack([onp.load(p).astype(onp.float16) for p in traj_3d_paths_head2], axis=0)
                else:
                    traj_3d_head2 = onp.stack([onp.load(p) for p in traj_3d_paths_head2], axis=0)
                    
                log_memory_usage("after loading head2 data")
                
                traj_3d_head2 = traj_3d_head2.reshape(traj_3d_head2.shape[0], -1, 6)
                if conf_paths_head2:
                    conf_head2 = onp.stack([onp.load(p).astype(onp.float16) for p in conf_paths_head2], axis=0)
                    conf_head2 = conf_head2.reshape(conf_head2.shape[0], -1)
                    conf_thre = np.percentile(conf_head2.astype(np.float32), conf_thre_percentile, axis=1)
                    conf_mask_head2 = conf_head2 > conf_thre[:, None]

    # Add visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_head1 = server.gui.add_checkbox("Tracking Points", True)
        gui_show_head2 = server.gui.add_checkbox("Recon Points", True)
        gui_show_trajectories = server.gui.add_checkbox("Trajectories", True)
        gui_use_color_tint = server.gui.add_checkbox("Use Color Tint", True)

    # Process and center point clouds
    center_point = None
    if traj_3d_head1 is not None:
        xyz_head1 = traj_3d_head1[:, :, :3]
        rgb_head1 = traj_3d_head1[:, :, 3:6]
        
        if center_point is None:
            center_point = onp.mean(xyz_head1, axis=(0, 1), keepdims=True)
        xyz_head1 -= center_point
        if rgb_head1.sum(axis=(-1)).max() > 125:
            rgb_head1 /= 255.0

    if traj_3d_head2 is not None:
        xyz_head2 = traj_3d_head2[:, :, :3]
        rgb_head2 = traj_3d_head2[:, :, 3:6]
            
        if center_point is None:
            center_point = onp.mean(xyz_head2, axis=(0, 1), keepdims=True)
        xyz_head2 -= center_point
        if rgb_head2.sum(axis=(-1)).max() > 125:
            rgb_head2 /= 255.0

    # Determine number of frames
    F = max(
        traj_3d_head1.shape[0] if traj_3d_head1 is not None else 0,
        traj_3d_head2.shape[0] if traj_3d_head2 is not None else 0
    )
    num_frames = min(max_frames, F)
    traj_end_frame = min(traj_end_frame, num_frames)
    print(f"Number of frames: {num_frames}")
    xyz_head1 = xyz_head1[:num_frames]
    xyz_head2 = xyz_head2[:num_frames]
    rgb_head1 = rgb_head1[:num_frames]
    rgb_head2 = rgb_head2[:num_frames]

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=20
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30")
        )
        gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
        gui_stride = server.gui.add_slider(
            "Stride",
            min=1,
            max=num_frames,
            step=1,
            initial_value=1,
            disabled=True,  # Initially disabled
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
        gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
        gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        if not gui_show_all_frames.value:
            with server.atomic():
                if gui_show_head1.value:
                    frame_nodes_head1[current_timestep].visible = True
                    frame_nodes_head1[prev_timestep].visible = False
                if gui_show_head2.value:
                    frame_nodes_head2[current_timestep].visible = True
                    frame_nodes_head2[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Show or hide all frames based on the checkbox.
    @gui_show_all_frames.on_update
    def _(_) -> None:
        gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
        if gui_show_all_frames.value:
            # Show frames with stride
            stride = gui_stride.value
            with server.atomic():
                for i, (node1, node2) in enumerate(zip(frame_nodes_head1, frame_nodes_head2)):
                    node1.visible = gui_show_head1.value and (i % stride == 0)
                    node2.visible = gui_show_head2.value and (i % stride == 0)
            # Disable playback controls
            gui_playing.disabled = True
            gui_timestep.disabled = True
            gui_next_frame.disabled = True
            gui_prev_frame.disabled = True
        else:
            # Show only the current frame
            current_timestep = gui_timestep.value
            with server.atomic():
                for i, (node1, node2) in enumerate(zip(frame_nodes_head1, frame_nodes_head2)):
                    node1.visible = gui_show_head1.value and (i == current_timestep)
                    node2.visible = gui_show_head2.value and (i == current_timestep)
            # Re-enable playback controls
            gui_playing.disabled = False
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

    # Update frame visibility when the stride changes.
    @gui_stride.on_update
    def _(_) -> None:
        if gui_show_all_frames.value:
            # Update frame visibility based on new stride
            stride = gui_stride.value
            with server.atomic():
                for i, (node1, node2) in enumerate(zip(frame_nodes_head1, frame_nodes_head2)):
                    node1.visible = gui_show_head1.value and (i % stride == 0)
                    node2.visible = gui_show_head2.value and (i % stride == 0)

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes_head1: list[viser.FrameHandle] = []
    frame_nodes_head2: list[viser.FrameHandle] = []

    # Extract RGB components for tinting
    blue_r, blue_g, blue_b = blue_rgb
    red_r, red_g, red_b = red_rgb
    
    # Create frames for each timestep
    frame_nodes_head1 = []
    frame_nodes_head2 = []
    for i in tqdm(range(num_frames)):
        # Process head1
        if traj_3d_head1 is not None:
            frame_nodes_head1.append(server.scene.add_frame(f"/frames/t{i}/head1", show_axes=False))
            position = xyz_head1[i]
            color = rgb_head1[i]
            if conf_mask_head1 is not None:
                position = position[conf_mask_head1[i]]
                color = color[conf_mask_head1[i]]
            
            # Add point cloud for head1 with optional blue tint
            color_head1 = color.copy()
            if gui_use_color_tint.value:
                color_head1 *= blend_ratio
                color_head1[:, 0] = onp.clip(color_head1[:, 0] + blue_r * (1 - blend_ratio), 0, 1)  # R
                color_head1[:, 1] = onp.clip(color_head1[:, 1] + blue_g * (1 - blend_ratio), 0, 1)  # G
                color_head1[:, 2] = onp.clip(color_head1[:, 2] + blue_b * (1 - blend_ratio), 0, 1)  # B
            
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/head1/point_cloud",
                points=position[::downsample_factor],
                colors=color_head1[::downsample_factor],
                point_size=point_size,
                point_shape="rounded",
            )

        # Process head2
        if traj_3d_head2 is not None:
            frame_nodes_head2.append(server.scene.add_frame(f"/frames/t{i}/head2", show_axes=False))
            position = xyz_head2[i]
            color = rgb_head2[i]
            if conf_mask_head2 is not None:
                position = position[conf_mask_head2[i]]
                color = color[conf_mask_head2[i]]
            
            # Add point cloud for head2 with optional red tint
            color_head2 = color.copy()
            if gui_use_color_tint.value:
                color_head2 *= blend_ratio
                color_head2[:, 0] = onp.clip(color_head2[:, 0] + red_r * (1 - blend_ratio), 0, 1)  # R
                color_head2[:, 1] = onp.clip(color_head2[:, 1] + red_g * (1 - blend_ratio), 0, 1)  # G
                color_head2[:, 2] = onp.clip(color_head2[:, 2] + red_b * (1 - blend_ratio), 0, 1)  # B
            
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/head2/point_cloud",
                points=position[::downsample_factor],
                colors=color_head2[::downsample_factor],
                point_size=point_size,
                point_shape="rounded",
            )

    # Update visibility based on checkboxes
    @gui_show_head1.on_update
    def _(_) -> None:
        with server.atomic():
            for frame_node in frame_nodes_head1:
                frame_node.visible = gui_show_head1.value and (
                    gui_show_all_frames.value
                    or (not gui_show_all_frames.value )
                )

    @gui_show_head2.on_update
    def _(_) -> None:
        with server.atomic():
            for frame_node in frame_nodes_head2:
                frame_node.visible = gui_show_head2.value and (
                    gui_show_all_frames.value
                    or (not gui_show_all_frames.value )
                )

    # Initial visibility
    for i, (node1, node2) in enumerate(zip(frame_nodes_head1, frame_nodes_head2)):
        if gui_show_all_frames.value:
            node1.visible = gui_show_head1.value and (i % gui_stride.value == 0)
            node2.visible = gui_show_head2.value and (i % gui_stride.value == 0)
        else:
            node1.visible = gui_show_head1.value and (i == gui_timestep.value)
            node2.visible = gui_show_head2.value and (i == gui_timestep.value)

    # Process and visualize trajectories for head1
    if traj_3d_head1 is not None:
        # Get points over time
        xyz_head1_centered = xyz_head1.copy()
        
        
        num_points = xyz_head1.shape[1]
        points_to_visualize = min(num_points, num_traj_points)
        
        # Get the mask for the first frame and reshape it to match point cloud dimensions
        first_frame_mask = masks[0].reshape(-1)
        
        # Calculate trajectory lengths for each point
        trajectories = xyz_head1_centered[traj_start_frame:traj_end_frame]  # Shape: (num_frames, num_points, 3)
        traj_diffs = np.diff(trajectories, axis=0)  # Differences between consecutive frames
        

        traj_lengths = np.sum(np.sqrt(np.sum(traj_diffs**2, axis=-1)), axis=0)  # Sum of distances for each point
        
        # Get points that are within the mask
        valid_indices = np.where(first_frame_mask)[0]
        
        if len(valid_indices) > 0:
            # Calculate average trajectory length for masked points
            masked_traj_lengths = traj_lengths[valid_indices]
            avg_traj_length = np.mean(masked_traj_lengths)
            
            if mask_folder is not None:
                long_traj_indices = valid_indices
            else:
                long_traj_indices = valid_indices[masked_traj_lengths >= avg_traj_length]
            
            # Randomly sample from the filtered points
            if len(long_traj_indices) > 0:
                selected_indices = np.random.choice(
                    len(long_traj_indices),
                    min(points_to_visualize, len(long_traj_indices)),
                    replace=False
                )
                valid_point_indices = long_traj_indices[np.sort(selected_indices)]
            else:
                valid_point_indices = np.array([])
        else:
            valid_point_indices = np.array([])
        
        if len(valid_point_indices) > 0:
            trajectories = xyz_head1_centered[traj_start_frame:traj_end_frame, valid_point_indices]
            N_point = trajectories.shape[1]
            if color_code == "rainbow":
                point_colors = plt.cm.rainbow(np.linspace(0, 1, N_point))[:, :3]
            elif color_code == "jet":
                point_colors = plt.cm.jet(np.linspace(0, 1, N_point))[:, :3]
            for i in range(traj_end_frame - traj_start_frame):
                actual_length = min(fixed_length_traj, i + 1)
                
                if actual_length > 1:
                    start_idx = max(0, i - actual_length + 1)
                    end_idx = i + 1
                    
                    traj_slice = trajectories[start_idx:end_idx]
                    line_points = np.stack([traj_slice[:-1], traj_slice[1:]], axis=2)
                    line_points = line_points.reshape(-1, 2, 3)
                    
                    line_colors = np.tile(point_colors, (actual_length-1, 1))
                    line_colors = np.stack([line_colors, line_colors], axis=1)
                    
                    server.scene.add_line_segments(
                        name=f"/frames/t{i+traj_start_frame}/head1/trajectory",
                        points=line_points,
                        colors=line_colors,
                        line_width=traj_line_width,
                        visible=gui_show_trajectories.value
                    )

    @gui_show_trajectories.on_update
    def _(_) -> None:
        with server.atomic():
            for i in range(num_frames):
                try:
                    server.scene.remove_by_name(f"/frames/t{i}/head1/trajectory")
                except KeyError:
                    pass
            
            if gui_show_trajectories.value and traj_3d_head1 is not None:
                last_frame_mask = masks[traj_end_frame-1].reshape(-1)
                
                trajectories = xyz_head1_centered[traj_start_frame:traj_end_frame]
                traj_diffs = np.diff(trajectories, axis=0)
                traj_lengths = np.sum(np.sqrt(np.sum(traj_diffs**2, axis=-1)), axis=0)
                
                valid_indices = np.where(last_frame_mask)[0]
                
                if len(valid_indices) > 0:
                    masked_traj_lengths = traj_lengths[valid_indices]
                    avg_traj_length = np.mean(masked_traj_lengths)
                    long_traj_indices = valid_indices[masked_traj_lengths >= avg_traj_length]
                    
                    if len(long_traj_indices) > 0:
                        selected_indices = np.random.choice(
                            len(long_traj_indices),
                            min(points_to_visualize, len(long_traj_indices)),
                            replace=False
                        )
                        valid_point_indices = long_traj_indices[np.sort(selected_indices)]
                    else:
                        valid_point_indices = np.array([])
                else:
                    valid_point_indices = np.array([])

                if len(valid_point_indices) > 0:
                    trajectories = xyz_head1_centered[traj_start_frame:traj_end_frame, valid_point_indices]
                    N_point = trajectories.shape[1]

                    if color_code == "rainbow":
                        point_colors = plt.cm.rainbow(np.linspace(0, 1, N_point))[:, :3]
                    elif color_code == "jet":
                        point_colors = plt.cm.jet(np.linspace(0, 1, N_point))[:, :3]
                    
                    for i in range(traj_end_frame - traj_start_frame):
                        actual_length = min(fixed_length_traj, i + 1)
                        
                        if actual_length > 1:
                            start_idx = max(0, i - actual_length + 1)
                            end_idx = i + 1
                            
                            traj_slice = trajectories[start_idx:end_idx]
                            line_points = np.stack([traj_slice[:-1], traj_slice[1:]], axis=2)
                            line_points = line_points.reshape(-1, 2, 3)
                            
                            line_colors = np.tile(point_colors, (actual_length-1, 1))
                            line_colors = np.stack([line_colors, line_colors], axis=1)
                            
                            server.scene.add_line_segments(
                                name=f"/frames/t{i+traj_start_frame}/head1/trajectory",
                                points=line_points,
                                colors=line_colors,
                                line_width=traj_line_width,
                                visible=True
                            )

    @gui_use_color_tint.on_update
    def _(_) -> None:
        with server.atomic():
            for i in range(num_frames):
                if traj_3d_head1 is not None:
                    position = xyz_head1[i]
                    color = rgb_head1[i]
                    if conf_mask_head1 is not None:
                        position = position[conf_mask_head1[i]]
                        color = color[conf_mask_head1[i]]
                    
                    color_head1 = color.copy()
                    if gui_use_color_tint.value:
                        color_head1 *= blend_ratio
                        color_head1[:, 0] = onp.clip(color_head1[:, 0] + blue_r * (1 - blend_ratio), 0, 1)  # R
                        color_head1[:, 1] = onp.clip(color_head1[:, 1] + blue_g * (1 - blend_ratio), 0, 1)  # G
                        color_head1[:, 2] = onp.clip(color_head1[:, 2] + blue_b * (1 - blend_ratio), 0, 1)  # B
                    
                    server.scene.remove_by_name(f"/frames/t{i}/head1/point_cloud")
                    server.scene.add_point_cloud(
                        name=f"/frames/t{i}/head1/point_cloud",
                        points=position[::downsample_factor],
                        colors=color_head1[::downsample_factor],
                        point_size=point_size,
                        point_shape="rounded",
                    )
                
                if traj_3d_head2 is not None:
                    position = xyz_head2[i]
                    color = rgb_head2[i]
                    if conf_mask_head2 is not None:
                        position = position[conf_mask_head2[i]]
                        color = color[conf_mask_head2[i]]
                    
                    color_head2 = color.copy()
                    if gui_use_color_tint.value:
                        color_head2 *= blend_ratio
                        color_head2[:, 0] = onp.clip(color_head2[:, 0] + red_r * (1 - blend_ratio), 0, 1)  # R
                        color_head2[:, 1] = onp.clip(color_head2[:, 1] + red_g * (1 - blend_ratio), 0, 1)  # G
                        color_head2[:, 2] = onp.clip(color_head2[:, 2] + red_b * (1 - blend_ratio), 0, 1)  # B
                    
                    server.scene.remove_by_name(f"/frames/t{i}/head2/point_cloud")
                    server.scene.add_point_cloud(
                        name=f"/frames/t{i}/head2/point_cloud",
                        points=position[::downsample_factor],
                        colors=color_head2[::downsample_factor],
                        point_size=point_size,
                        point_shape="rounded",
                    )

    log_memory_usage("before starting playback loop")
    
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value and not gui_show_all_frames.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(visualize_st4rtrack)
