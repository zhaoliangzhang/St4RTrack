import torch

from dust3r.losses import MultiLoss, track_loss_fn
from dust3r.camera_solver import CameraLoss


class RobotConfLoss(MultiLoss):
    """CoTracker-based trajectory and 3D alignment loss for robot pair mode.

    This loss uses only CoTracker-derived 2D tracks as supervision:
    - 2D trajectory loss: model-projected tracks vs CoTracker tracks
    - 3D alignment loss: consistency between head1 and head2 3D trajectories
    It intentionally skips confidence-weighted regression and depth losses.
    """

    def __init__(self, pixel_loss=None, alpha=1, pair_mode=True,
                 traj_weight=0.0, align3d_weight=0.0, pred_intrinsics=False,
                 cotracker=True, **kwargs):
        super().__init__()
        assert pair_mode, 'RobotConfLoss currently supports pair_mode only'
        self.traj_weight = float(traj_weight)
        self.align3d_weight = float(align3d_weight)
        self.pose_loss = CameraLoss(pred_intrinsics=pred_intrinsics) if self.traj_weight != 0 else None
        self.cotracker = cotracker
        if self.cotracker:
            # Lazy load CoTracker like ConfLoss
            self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()
            self.cotracker.eval()
            self.grid_size = 32

    def get_name(self):
        return f'RobotConfLoss(CoTracker)'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def get_tracks_2d(self, gt1, gt2, pred1, pred2, **kw):
        """
        Projects 3D points from pred1['pts3d'] onto image plane using GT camera poses.
        Uses gt1['camera_pose'] and gt2['camera_pose'] instead of solving for poses.
        
        Args:
            gt1, gt2: Ground truth dicts with 'camera_pose' (w2c format)
            pred1, pred2: Model predictions
            
        Returns:
            x2d: [B, H, W, 2] - 2D projections
            gt_poses: [B, 4, 4] - GT camera poses (w2c)
        """
        device = pred1['pts3d'].device
        B, H, W, _ = pred1['pts3d'].shape
        
        # Get GT camera poses (w2c format) and ensure they're on the same device
        gt_poses1 = gt1['camera_pose'].to(device)  # [B, S, 4, 4] - w2c for view1
        gt_poses2 = gt2['camera_pose'].to(device)  # [B, S, 4, 4] - w2c for view2
        
        # Flatten batch and sequence dimensions for processing
        B, S = gt_poses1.shape[:2]
        gt_poses1_flat = gt_poses1.view(B * S, 4, 4)  # [B*S, 4, 4]
        gt_poses2_flat = gt_poses2.view(B * S, 4, 4)  # [B*S, 4, 4]
        
        # Get relative pose: T_cam2_to_cam1 = T_cam2_to_w @ T_w_to_cam1
        # Since poses are w2c, we need: T_cam2_to_cam1 = T_cam2_to_w @ T_w_to_cam1
        # T_cam2_to_w = inv(T_w_to_cam2) = inv(gt_poses2)
        # T_w_to_cam1 = gt_poses1
        relative_poses = torch.linalg.inv(gt_poses2_flat) @ gt_poses1_flat  # [B*S, 4, 4]
        # Get 3D points from head1 and flatten to match pose dimensions
        tracks = pred1['pts3d']  # [B, H, W, 3]
        tracks_flat = tracks.view(B * S, H, W, 3)  # [B*S, H, W, 3]
        
        # Convert to homogeneous coordinates
        ones = torch.ones_like(tracks_flat[..., :1])
        tracks_hom = torch.cat([tracks_flat, ones], dim=-1)  # [B*S, H, W, 4]
        tracks_hom = tracks_hom.view(B * S, -1, 4).transpose(1, 2)  # [B*S, 4, H*W]
        
        # Transform to camera2 coordinate system: T_cam1_to_cam2 @ tracks_hom
        # T_cam1_to_cam2 = inv(T_cam2_to_cam1) = inv(relative_poses)
        x_in_cam = torch.linalg.inv(relative_poses) @ tracks_hom  # [B*S, 4, H*W]
        
        # Normalize homogeneous coordinates
        x_in_cam = x_in_cam / (x_in_cam[:, 3:4, :] + 1e-8)
        x_in_cam_3d = x_in_cam[:, :3, :]  # [B*S, 3, H*W]
        
        # Project to 2D using camera intrinsics
        gt_intrinsics = gt2['camera_intrinsics'].to(device)  # [B, S, 3, 3]
        gt_intrinsics_flat = gt_intrinsics.view(B * S, 3, 3)  # [B*S, 3, 3]
        x2d = gt_intrinsics_flat @ x_in_cam_3d  # [B*S, 3, H*W]
        x2d = x2d / (x2d[:, 2:3, :] + 1e-8)  # Normalize by depth
        x2d = x2d[:, :2, :]  # Take x,y coordinates
        
        # Reshape to [B*S, H, W, 2]
        x2d = x2d.transpose(1, 2).view(B * S, H, W, 2)
        
        return x2d, relative_poses

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        """Robot-specific CoTracker losses: track from view1 to view2 for each image pair."""

        device = pred1['pts3d'].device
        Bz, H, W, _ = pred1['pts3d'].shape

        track_loss = 0
        align3d_loss = 0
        details = {}
        if not self.cotracker or (self.traj_weight == 0 and self.align3d_weight == 0):
            total = pred1['pts3d'].new_zeros(())
            return total, dict(loss_traj_2d=0.0, loss_align3d=0.0)

        # Robot data: view1 has multiple images, view2 has corresponding next frames
        # We need to track from each view1 image to its corresponding view2 frame
        img1 = gt1['img'].to(device)  # [B, S, C, H, W] - multiple different images
        img2 = gt2['img'].to(device)  # [B, S, C, H, W] - corresponding next frames
        
        # Flatten batch and sequence dimensions for processing
        B, S = img1.shape[:2]
        img1_flat = img1.view(B * S, *img1.shape[2:])  # [B*S, C, H, W]
        img2_flat = img2.view(B * S, *img2.shape[2:])  # [B*S, C, H, W]
        
        # For each image pair, create a 2-frame sequence for CoTracker
        all_tracks = []
        all_visibility = []
        
        # Use consistent grid size to avoid dynamic compute graph issues
        grid_size = self.grid_size + torch.randint(-8, 8, (1,), device=device).item()
        
        for i in range(B * S):
            # Create 2-frame sequence: [view1_img, view2_img]
            pair_sequence = torch.stack([img1_flat[i], img2_flat[i]], dim=0)  # [2, C, H, W]
            pair_sequence = pair_sequence.unsqueeze(0) * 255  # [1, 2, C, H, W] for CoTracker
            
            with torch.no_grad():
                cotracker_output = self.cotracker(pair_sequence, grid_size=grid_size)
                
                # Handle different CoTracker output formats
                if len(cotracker_output) == 2:
                    pair_tracks, pair_visibility = cotracker_output
                else:
                    pair_tracks = cotracker_output[0]
                    pair_visibility = cotracker_output[1] if len(cotracker_output) > 1 else None
                
                # pair_tracks: [1, 2, N, 2] -> [2, N, 2]
                pair_tracks = pair_tracks[0]  # [2, N, 2] - tracks from frame 0 to frame 1
                all_tracks.append(pair_tracks)
                
                if pair_visibility is not None:
                    pair_visibility = pair_visibility[0]  # [2, N]
                    all_visibility.append(pair_visibility)
        
        # Stack all tracks: [B*S, 2, N, 2]
        all_tracks = torch.stack(all_tracks, dim=0)  # [B*S, 2, N, 2]
        if all_visibility:
            all_visibility = torch.stack(all_visibility, dim=0)  # [B*S, 2, N] or [B*S, N]
        
        # Get tracks from frame 0 to frame 1 (view1 to view2)
        tracks_0_to_1 = all_tracks[:, 1, :, :]  # [B*S, N, 2] - tracks at frame 1 (view2)
        tracks_0 = all_tracks[:, 0, :, :]  # [B*S, N, 2] - tracks at frame 0 (view1)
        
        # Build mask of valid points at first frame
        rounded_coords = torch.round(tracks_0).long()  # [B*S, N, 2]
        x_coords = rounded_coords[..., 0]
        y_coords = rounded_coords[..., 1]
        valid = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
        
        # Create mask for each image pair
        cotracker_masks = []
        for i in range(B * S):
            mask = torch.zeros((H, W), device=device, dtype=torch.bool)
            valid_i = valid[i]
            if valid_i.any():
                x_coords_i = x_coords[i][valid_i]
                y_coords_i = y_coords[i][valid_i]
                mask[y_coords_i, x_coords_i] = True
            cotracker_masks.append(mask)
        
        cotracker_masks = torch.stack(cotracker_masks, dim=0)  # [B*S, H, W]

        # 2D trajectory loss using CoTracker
        if self.traj_weight != 0:
            pred_tracks_2d, gt_poses = self.get_tracks_2d(gt1, gt2, pred1, pred2)
            # pred_tracks_2d already has shape [B*S, H, W, 2] from get_tracks_2d
            
            # For each image pair, compute loss
            pair_losses = []
            for i in range(B * S):
                mask_i = cotracker_masks[i]  # [H, W]
                if mask_i.any():
                    # Get predicted tracks at valid points
                    pred_tracks_i = pred_tracks_2d[i][mask_i]  # [N_valid, 2]
                    
                    # Get CoTracker tracks at corresponding points
                    valid_i = valid[i]
                    if valid_i.any():
                        cotracker_tracks_i = tracks_0_to_1[i][valid_i]  # [N_valid, 2]
                        
                        if pred_tracks_i.numel() > 0 and cotracker_tracks_i.numel() > 0:
                            # Ensure same number of points
                            min_points = min(pred_tracks_i.shape[0], cotracker_tracks_i.shape[0])
                            if min_points > 0:
                                pair_loss, _ = track_loss_fn(
                                    pred_tracks_i[:min_points], 
                                    cotracker_tracks_i[:min_points], 
                                    H, W, align_scale=True
                                )
                                pair_losses.append(pair_loss)
            
            if pair_losses:
                track_loss = torch.stack(pair_losses).mean()

        # 3D alignment loss using CoTracker-selected points
        if self.align3d_weight != 0:
            # Get tracks from frame 0 to frame 1 (view1 to view2)

            tracks_0_to_1 = all_tracks[:, 1, :, :]  # [B*S, N, 2] - tracks at frame 1 (view2)
            tracks_0 = all_tracks[:, 0, :, :]  # [B*S, N, 2] - tracks at frame 0 (view1)

            # Get visibility for frame 1 (view2)
            # print('all_visibility', all_visibility.dtype, 'shape:', all_visibility.shape)
            if all_visibility is not None and all_visibility.numel() > 0:
                visibility_1 = all_visibility[:, 1, :]  # [B*S, N] - visibility at frame 1
            else:
                visibility_1 = torch.ones(all_tracks.shape[0], all_tracks.shape[2], device=device, dtype=torch.bool)
            # print('visibility_1', visibility_1.dtype, 'shape:', visibility_1.shape)
            
            # Build valid mask for 3D alignment
            rounded_tracks_1 = torch.round(tracks_0_to_1).long()  # [B*S, N, 2]
            valid_mask = (rounded_tracks_1[..., 0] >= 0) & (rounded_tracks_1[..., 0] < W) & \
                        (rounded_tracks_1[..., 1] >= 0) & (rounded_tracks_1[..., 1] < H) & \
                        visibility_1  # [B*S, N]
            
            # Clamp coordinates to valid range
            clamped_tracks = torch.stack([
                rounded_tracks_1[..., 0].clamp(0, W - 1),
                rounded_tracks_1[..., 1].clamp(0, H - 1)
            ], dim=-1)  # [B*S, N, 2]
            
            # Get 3D points from head1 and head2 at corresponding locations
            align3d_losses = []
            for i in range(B * S):
                valid_i = valid_mask[i]  # [N]
                if valid_i.any():
                    h_idx = clamped_tracks[i, :, 1]  # [N]
                    w_idx = clamped_tracks[i, :, 0]  # [N]
                    
                    # Get 3D points from head2 at tracked locations
                    pts3d_head2 = pred2['pts3d_in_other_view'][i, h_idx, w_idx]  # [N, 3]
                    
                    # Get 3D points from head1 at corresponding locations
                    # For robot data, we need to map from view1 to view2 coordinates
                    # This is a simplified version - you might need to adjust based on your specific setup
                    pts3d_head1 = pred1['pts3d'][i, h_idx, w_idx]  # [N, 3]
                    
                    # Compute distance between corresponding 3D points
                    valid_pts1 = pts3d_head1[valid_i]
                    valid_pts2 = pts3d_head2[valid_i]
                    
                    if valid_pts1.numel() > 0 and valid_pts2.numel() > 0:
                        distance = (valid_pts1 - valid_pts2).norm(dim=-1)
                        align3d_losses.append(distance.mean())
            
            if align3d_losses:
                align3d_loss = torch.stack(align3d_losses).mean()
            else:
                align3d_loss = 0

        total_loss = self.traj_weight * track_loss + self.align3d_weight * align3d_loss

        return total_loss, dict(
            loss_traj_2d=float(track_loss) if isinstance(track_loss, torch.Tensor) else float(track_loss),
            loss_align3d=float(align3d_loss) if isinstance(align3d_loss, torch.Tensor) else float(align3d_loss),
        )

    def forward(self, gt1, gt2, pred1, pred2, **kw):
        return self.compute_loss(gt1, gt2, pred1, pred2, **kw)


__all__ = ["RobotConfLoss"]


