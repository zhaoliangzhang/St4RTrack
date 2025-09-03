from dust3r.cloud_opt.cost_fun import AdaptiveHuberPnPCost
from dust3r.cloud_opt.camera import PerspectiveCamera
from dust3r.utils.geometry import inv, geotrf
from dust3r.cloud_opt.levenberg_marquardt import LMSolver, RSLMSolver
from dust3r.cloud_opt.oneref_viewer import pose7d_to_matrix
import torch
import torch.nn as nn
import numpy as np
from dust3r.post_process import estimate_focal_knowing_depth
import cv2

class CameraLoss(nn.Module):
    def __init__(self, pred_intrinsics=False, criterion=None):
        super().__init__()
        self.solver = LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=10,
                num_iter=20))
        self.cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        self.pred_intrinsics = pred_intrinsics

    def build_camera(self, K, H, W):
        # Calculate image boundaries
        allowed_border = 10
        wh_begin = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
        out_res = torch.tensor([H, W], dtype=torch.float32, device=self.device)
        lb = wh_begin - allowed_border
        ub = wh_begin + (out_res - 1) + allowed_border

        # Build camera model
        cam_intrinsic = torch.tensor(K, dtype=torch.float32, device=self.device)
        camera = PerspectiveCamera(
            cam_mats=cam_intrinsic,
            z_min=0.01,
            lb=lb,
            ub=ub
        )
        return camera

    def pose_solver(self, gt_intrinsics, pred2, mask=None, **kwargs):
        #given intrinsics, solve for the pose of the second camera
        
        pts3d_j = pred2['pts3d_in_other_view']
        conf_j  = pred2['conf']
        self.device = pts3d_j.device
        B, H, W = pts3d_j.shape[:3]
        
        # 1. Create containers (keep [B, W*H, ...] shape)
        cost_fun = self.cost_fun
        pixels = torch.stack(
            torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'),
            dim=-1
        ).float().to(self.device)            # [H, W, 2]  -> mesh
        pixels = pixels.reshape(1, W*H, 2)   # [1, W*H, 2]
        
        x3d_all = pts3d_j.reshape(B, W*H, 3).float().to(self.device)                   # [B, W*H, 3]
        x2d_all = pixels.expand(B, -1, -1)                                             # [B, W*H, 2]
        w2d_all = conf_j.reshape(B, W*H, 1).float().to(self.device).expand(-1, -1, 2)  # [B, W*H, 2]

        # If mask is None, no filtering needed
        if mask is not None:
            mask = mask.reshape(B, W*H)  # [B, W*H]

        # 2. Allocate space for output poses
        output_pose = torch.zeros(B, 7, device=self.device)  # Store (tx, ty, tz, qx, qy, qz, qw)

        # 3. Process each batch
        for i in range(B):
            # 3.1 Get all points for batch i
            x3d_i = x3d_all[i]  # [W*H, 3]
            x2d_i = x2d_all[i]  # [W*H, 2]
            w2d_i = w2d_all[i]  # [W*H, 2]

            # 3.2 If mask exists, keep only valid points
            if mask is not None:
                # mask[i] -> [W*H], contains True/False or 1/0
                valid_idx = mask[i].nonzero(as_tuple=True)[0]  # Get valid indices
                x3d_i = x3d_i[valid_idx]
                x2d_i = x2d_i[valid_idx]
                w2d_i = w2d_i[valid_idx]
            
            # 3.3 cost_fun/solver requires (B=1, N, 2/3), add batch dimension
            x3d_i = x3d_i.unsqueeze(0)  # [1, N, 3]
            x2d_i = x2d_i.unsqueeze(0)  # [1, N, 2]
            w2d_i = w2d_i.unsqueeze(0)  # [1, N, 2]

            # 3.4 Set cost_fun parameters
            cost_fun.set_param(x2d_i, w2d_i)

            # 3.5 Build camera and run solver
            camera = self.build_camera(gt_intrinsics[i:i+1], H, W)
            pose_opt, pose_cov, cost, pose_opt_plus = self.solver(
                x3d_i, x2d_i, w2d_i,
                camera, cost_fun,
                pose_init=None,
                cost_init=None,
                with_pose_cov=True,
                force_init_solve=False,
                normalize_override=False,
                with_pose_opt_plus=True,
                **kwargs
            )

            # 3.6 Save results (pose_opt_plus is [1,7], remove batch dim and store in output_pose[i])
            output_pose[i] = pose_opt_plus
            # output_pose[i] = pose_opt

        # 4. Convert 7D pose vector to 4x4 matrix
        output_pose_matrix = pose7d_to_matrix(output_pose)
        return output_pose_matrix


    def criterion(self, gt_poses: torch.Tensor, pred_poses: torch.Tensor):
        """
        Args:
            gt_poses:   Tensor of shape (B, 4, 4) representing the ground truth poses
            pred_poses: Tensor of shape (B, 4, 4) representing the predicted poses

        Returns:
            rot_loss:   Scalar representing the rotation error
            trans_loss: Scalar representing the translation error
        """
        # Extract rotation matrices and translation vectors
        R_gt = gt_poses[:, :3, :3]      # [B, 3, 3]
        R_pred = pred_poses[:, :3, :3]  # [B, 3, 3]
        t_gt = gt_poses[:, :3, 3]       # [B, 3]
        t_pred = pred_poses[:, :3, 3]   # [B, 3]

        # ============= Compute rotation error =============
        # Compute relative rotation R_diff = R_pred^T * R_gt
        # Trace of rotation matrix trace(R_diff) = R_diff[0,0] + R_diff[1,1] + R_diff[2,2]
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)  # [B, 3, 3]
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]  # [B]
        # Cosine value: cos(\theta) = (trace(R_diff) - 1) / 2
        cos_angle = 0.5 * (trace - 1.0)
        # Numerically stable clamp
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        # Compute angle (in radians) as rotation error, can take mean or return as vector
        rot_error = torch.acos(cos_angle)  # [B]

        # ============= Compute translation error =============
        # Compute L2 Norm between the two translation vectors
        trans_error = torch.norm(t_pred - t_gt, dim=-1)  # [B]

        # Return a simple mean as the final error
        rot_loss = rot_error.mean()
        trans_loss = trans_error.mean()

        return rot_loss, trans_loss

    def forward(self, gt2, pred1, pred2, **kw):
        
        gt_intrinsics = self.intrinsics_solver(pred1) if self.pred_intrinsics else gt2['camera_intrinsics'] # B,3,3

        gt_poses = gt2['camera_pose'] # B,4,4
        pred_poses = self.pose_solver(gt_intrinsics, pred2)
        
        gt_relative_poses = gt_poses[0].inverse() @ gt_poses # T_cam2_to_cam1_gt
        pred_relative_poses = pred_poses[0].inverse() @ pred_poses # T_cam2_to_cam1_pred

        rot_loss, trans_loss = self.criterion(gt_relative_poses, pred_relative_poses)

        total_loss = rot_loss + trans_loss
        return {
            'total_loss': total_loss,
            'rot_loss':   rot_loss,
            'trans_loss': trans_loss
        }

    def intrinsics_solver(self, pred1, **kw):
        B, H, W, THREE = pred1['pts3d'].shape
        pts3d_frame0 = pred1['pts3d'][:1]
        assert THREE == 3

        # assume pp is at the center of the image
        pp = torch.tensor([W/2, H/2], device=pts3d_frame0.device)
        focal = estimate_focal_knowing_depth(pts3d_frame0, pp, focal_mode='weiszfeld')

        K = torch.tensor([[focal, 0, pp[0]], [0, focal, pp[1]], [0, 0, 1]], device=pts3d_frame0.device).unsqueeze(0).repeat(B, 1, 1)
        return K

    def get_tracks_2d(self, gt1, gt2, pred1, pred2, **kw):
        """
        Projects the given 3D points (gt1['pts3d']) onto the image plane using the predicted extrinsic parameters (pred_poses).
        
        Returns:
            x2d: A tensor of shape (B, H, W, 2), representing the 2D coordinates in the image plane.
            pred_poses: A tensor of shape (B, 4, 4), representing the predicted extrinsic parameters(T_cam2_to_cam1_pred).
        """
        gt_intrinsics = self.intrinsics_solver(pred1) if self.pred_intrinsics else gt2['camera_intrinsics']  # Intrinsic matrix of shape (B, 3, 3)
        try:
            pred_poses = self.pose_solver(gt_intrinsics, pred2)  # Predicted extrinsic parameters of shape (B, 4, 4)
            pred_poses = pred_poses.inverse()
            # get the relative pose to the first frame
            pred_poses = pred_poses[0].inverse() @ pred_poses # T_cam2_to_cam1_pred
        except Exception as e:
            B = pred2['pts3d_in_other_view'].shape[0]
            print(f"Error in pose_solver: {e}")
            pred_poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(pred2['pts3d_in_other_view'].device)

        tracks = pred1['pts3d']  # 3D points of shape (B, H, W, 3)
        B, H, W, _ = tracks.shape

        # Convert 3D points to homogeneous coordinates
        ones = torch.ones_like(tracks[..., :1])  
        tracks_hom = torch.cat([tracks, ones], dim=-1) 
        tracks_hom = tracks_hom.view(B, -1, 4)          

        #  Transform to camera coordinate system
        x_in_cam = torch.linalg.inv(pred_poses).bmm(tracks_hom.transpose(1, 2))  # T_cam1_to_cam2_pred @ tracks_hom

        # Normalize homogeneous coordinates by dividing by the last dimension (w)
        x_in_cam = x_in_cam / (x_in_cam[:, 3:4, :] + 1e-8) 
        x_in_cam_3d = x_in_cam[:, :3, :]  #head1 output in pred_cam2 coordinate

        #  Project onto image plane: x2d = K * X_cam_normalized
        x2d = gt_intrinsics.bmm(x_in_cam_3d)  
        x2d = x2d / (x2d[:, 2:3, :] + 1e-8)   
        x2d = x2d[:, :2, :]                   

        # Reshape to (B, H, W, 2)
        x2d = x2d.permute(0, 2, 1).view(B, H, W, 2) # projected head1 output in pred_cam2 img plane

        return x2d, pred_poses

    def get_tracks_2d_gt(self, gt1, gt2, **kw):
        """
        Projects the given 3D points (gt1['pts3d']) onto the image plane using the predicted extrinsic parameters (pred_poses).
        
        Returns:
            x2d: A tensor of shape (B, H, W, 2), representing the 2D coordinates in the image plane.
        """
        gt_intrinsics = gt2['camera_intrinsics']  # Intrinsic matrix of shape (B, 3, 3)

        in_camera1 = inv(gt1['camera_pose']) # T_cam1_to_w_gt
        gt_poses = gt2['camera_pose']
        gt_poses = gt_poses[0].inverse() @ gt_poses # T_cam2_to_cam1_gt

        tracks = geotrf(in_camera1, gt1['traj_ptc']) #head1 output in world coordinate
        
        B, H, W, _ = tracks.shape

        # 1. Convert 3D points to homogeneous coordinates =============
        ones = torch.ones_like(tracks[..., :1])  
        tracks_hom = torch.cat([tracks, ones], dim=-1)  
        tracks_hom = tracks_hom.view(B, -1, 4)   

        # ============= 2. Transform to camera coordinate system =============
        x_in_cam = torch.linalg.inv(gt_poses).bmm(tracks_hom.transpose(1, 2))  # T_cam1_to_cam2_gt @ tracks_gt_hom

        # Normalize homogeneous coordinates by dividing by the last dimension (w)
        x_in_cam = x_in_cam / (x_in_cam[:, 3:4, :] + 1e-8)  
        x_in_cam_3d = x_in_cam[:, :3, :]                   

        # ============= 3. Project onto image plane: x2d = K * X_cam_normalized =============
        x2d = gt_intrinsics.bmm(x_in_cam_3d)  
        x2d = x2d / (x2d[:, 2:3, :] + 1e-8)   
        x2d = x2d[:, :2, :]                   

        # Reshape to (B, H, W, 2)
        x2d = x2d.permute(0, 2, 1).view(B, H, W, 2)

        return x2d

    def get_depth_head2(self, gt2, pred1, pred2, pred_poses = None, **kw):
        """
        Transforms the 3D points (pts3d_in_other_view) in pred2 to the camera coordinate system
        using the predicted extrinsic parameters and returns their depth (i.e., Z-coordinate).
        
        Returns:
            depth_map: A tensor of shape (B, H, W), representing the depth value for each pixel
                    after projection into the camera coordinate system.
        """
        if pred_poses is None:
            gt_intrinsics = self.intrinsics_solver(pred1) if self.pred_intrinsics else gt2['camera_intrinsics']
            pred_poses = self.pose_solver(gt_intrinsics, pred2)  # Predicted extrinsic parameters of shape (B, 4, 4)
            pred_poses = pred_poses.inverse()
            # get the relative pose to the first frame
            pred_poses = pred_poses[0].inverse() @ pred_poses
        
        pts3d_in_head2 = pred2['pts3d_in_other_view']  # 3D points in the other view, shape (B, H, W, 3)
        B, H, W, _ = pts3d_in_head2.shape

        #Convert 3D points to homogeneous coordinates 
        ones = torch.ones_like(pts3d_in_head2[..., :1])  
        x3d_hom = torch.cat([pts3d_in_head2, ones], dim=-1)  
        x3d_hom = x3d_hom.view(B, -1, 4)                    

        #Transform to camera coordinate system 
        x_in_cam = torch.linalg.inv(pred_poses).bmm(x3d_hom.transpose(1, 2))  

        # Normalize and extract depth 
        x_in_cam = x_in_cam / (x_in_cam[:, 3:4, :] + 1e-8)  
        depth = x_in_cam[:, 2, :]                           

        depth_map = depth.view(B, H, W)
        return depth_map

    def get_depth_head2_gt(self, gt2, **kw):
        """
        Transforms the 3D points (pts3d_in_other_view) in pred2 to the camera coordinate system
        using the predicted extrinsic parameters and returns their depth (i.e., Z-coordinate).
        
        Returns:
            depth_map: A tensor of shape (B, H, W), representing the depth value for each pixel
                    after projection into the camera coordinate system.
        """
        if 'depthmap' in gt2 and gt2['depthmap'].sum() > 0:
            depth_map = gt2['depthmap']
            return depth_map

        elif 'pts3d' in gt2:
            gt_poses = gt2['camera_pose']
            gt_poses = gt_poses[0].inverse() @ gt_poses #T_cam2_to_cam1_gt

            pts3d_in_head2 = geotrf(inv(gt2['camera_pose']), gt2['pts3d']) #head2 output in world coordinate
            B, H, W, _ = pts3d_in_head2.shape

            # Convert 3D points to homogeneous coordinates
            ones = torch.ones_like(pts3d_in_head2[..., :1])  
            x3d_hom = torch.cat([pts3d_in_head2, ones], dim=-1)  
            x3d_hom = x3d_hom.view(B, -1, 4)                    

            # Transform to camera coordinate system
            x_in_cam = torch.linalg.inv(gt_poses).bmm(x3d_hom.transpose(1, 2))  

            # Normalize and extract depth
            x_in_cam = x_in_cam / (x_in_cam[:, 3:4, :] + 1e-8)  
            depth = x_in_cam[:, 2, :]     

            return depth.view(B, H, W)     

        else:
            raise ValueError("No depth map found in gt2, cur dataset does not support depth loss")            