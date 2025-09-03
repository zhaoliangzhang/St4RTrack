# --------------------------------------------------------
# Dummy optimizer for visualizing pairs
# --------------------------------------------------------
import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
import cv2
import tqdm
import sys
sys.path.append('.')
from tqdm import tqdm
from glob import glob
from dust3r.cloud_opt.base_opt import c2w_to_tumpose
from dust3r.utils.geometry import inv, geotrf, depthmap_to_absolute_camera_coordinates
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.cloud_opt.cost_fun import AdaptiveHuberPnPCost
from dust3r.cloud_opt.camera import PerspectiveCamera
from dust3r.cloud_opt.levenberg_marquardt import LMSolver, RSLMSolver
from dust3r.cloud_opt.epropnp import EProPnP6DoF
from dust3r.cloud_opt.common import evaluate_pnp, pnp_normalize, pnp_denormalize
from functools import partial
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt

def pose7d_to_matrix(pose):
    """
    Convert a 7D pose (translation + quaternion) to a 4x4 transformation matrix.
    
    Args:
        pose (torch.Tensor): Shape (..., 7), where the last dimension contains
                             [x, y, z, w, i, j, k].
                             
    Returns:
        torch.Tensor: Shape (..., 4, 4), the transformation matrix.
    """
    assert pose.shape[-1] == 7, "Input pose must have 7 dimensions: [x, y, z, w, i, j, k]."
    
    # Extract translation and quaternion
    translation = pose[..., :3]  # (..., 3)
    quaternion = pose[..., 3:]   # (..., 4)
    
    # Normalize the quaternion to ensure it represents a valid rotation
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    
    # Extract quaternion components
    w, i, j, k = quaternion.unbind(dim=-1)  # Scalars
    
    # Compute rotation matrix from quaternion
    rotation_matrix = torch.stack([
        1 - 2 * (j**2 + k**2), 2 * (i*j - k*w),     2 * (i*k + j*w),
        2 * (i*j + k*w),     1 - 2 * (i**2 + k**2), 2 * (j*k - i*w),
        2 * (i*k - j*w),     2 * (j*k + i*w),     1 - 2 * (i**2 + j**2)
    ], dim=-1).reshape(*quaternion.shape[:-1], 3, 3)  # (..., 3, 3)
    
    # Construct 4x4 transformation matrix
    transformation_matrix = torch.eye(4, dtype=pose.dtype, device=pose.device)
    # Set rotation matrix and translation
    num_batches = rotation_matrix.shape[:-2]
    transformation_matrix = transformation_matrix.repeat(*num_batches, 1, 1)
    transformation_matrix[..., :3, :3] = rotation_matrix
    transformation_matrix[..., :3, 3] = translation
    
    return transformation_matrix

def monte_carlo_pose_loss(pose_sample_logweights, cost_target):
    """
    Args:
        pose_sample_logweights: Shape (mc_samples, num_obj)
        cost_target: Shape (num_obj, )

    Returns:
        Tensor: Shape (num_obj, )
    """
    loss_tgt = cost_target
    loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

    loss_pose = loss_tgt + loss_pred  # (num_obj, )
    loss_pose[loss_pose.isnan()] = 0
    return loss_pose

def build_cam(K, x3d, H, W):
    # Calculate image boundaries
    allowed_border = 30
    wh_begin = torch.tensor([0, 0], dtype=torch.float32, device=x3d.device)
    out_res = torch.tensor([H, W], dtype=torch.float32, device=x3d.device)
    lb = wh_begin - allowed_border
    ub = wh_begin + (out_res - 1) + allowed_border

    # Build camera model
    bs = x3d.size(0)
    cam_intrinsic = torch.tensor(K, dtype=torch.float32, device=x3d.device)
    camera = PerspectiveCamera(
        cam_mats=cam_intrinsic[None].expand(bs, -1, -1),
        z_min=0.01,
        lb=lb,
        ub=ub
    )
    return camera

def lm_solver(x3d, x2d, w2d, camera, pose_init=None, force_init_solve=False, **kwargs):
    # todo: camera
    solver = LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=10,
                num_iter=20))
    cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
    cost_fun.set_param(x2d, w2d)
    evaluate_fun = partial(
            evaluate_pnp,
            x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
    cost_init = evaluate_fun(pose=pose_init)[1] if pose_init is not None else None
    pose_opt, pose_cov, cost, pose_opt_plus  = solver(
            x3d, x2d, w2d, camera, cost_fun,
            pose_init=pose_init, cost_init=cost_init,
            with_pose_cov=True, force_init_solve=force_init_solve,
            normalize_override=False, **kwargs)
    return pose_opt_plus


class OneRefViewer:
    def __init__(self,
                pts3d_i,        # (B, H, W, 3)
                pts3d_j,        # (B, H, W, 3)
                conf_i = None,  # (B, H, W)
                conf_j = None,  # (B, H, W)
                share_focal = True,
                focal_pair1 = False,   # use the focal length of the first pair (1,1) for all pairs
                min_conf_thr = 1.5,
                epro_pnp=False,
                *args, **kwargs):
        
        self.pts3d_i = pts3d_i
        self.pts3d_j = pts3d_j
        self.conf_i = torch.zeros_like(pts3d_i[..., 0]) if conf_i is None else conf_i
        self.conf_j = torch.zeros_like(pts3d_j[..., 0]) if conf_j is None else conf_j
        self.share_focal = share_focal
        self.focal_pair1 = focal_pair1
        self.min_conf_thr = min_conf_thr
        self.epro_pnp = epro_pnp
        self.device = pts3d_i.device
        self.n_imgs = pts3d_i.shape[0]

        self.focals = []
        self.pp = []
        self.im_poses = []

        self.intrinsics_solver()
        if self.epro_pnp:
            # 初始化EProPnP求解器
            self.epropnp = EProPnP6DoF(
                mc_samples=512,
                num_iter=4,
                solver=LMSolver(
                    dof=6,
                    num_iter=5,
                    init_solver=RSLMSolver(
                        dof=6,
                        num_points=16,
                        num_proposals=10,
                        num_iter=20))
            ).to(self.device)
        for i in tqdm(range(self.n_imgs)):
            self.pair_solver(i) if not self.epro_pnp else self.epro_pnp_solver(i)

    def intrinsics_solver(self):
        for i in range(self.n_imgs):
            pts3d_i = self.pts3d_i[i]
            H, W = pts3d_i.shape[:2]
            pp = torch.tensor((W/2, H/2)).to(self.device)
            # assume the focal length is the same for all images
            focal = float(estimate_focal_knowing_depth(pts3d_i[None], pp, focal_mode='weiszfeld'))
            self.focals.append(focal)
            self.pp.append(pp)
        if self.share_focal:
            self.focals = [np.mean(self.focals)] * self.n_imgs
        if self.focal_pair1:
            self.focals = [self.focals[0]] * self.n_imgs

    def pair_solver(self, i):
        pts3d_i = self.pts3d_i[i]
        pts3d_j = self.pts3d_j[i]
        conf_i = self.conf_i[i]
        conf_j = self.conf_j[i]
        H, W = pts3d_i.shape[:2]
        pp = self.pp[i]
        focal = self.focals[i]

        # estimate the pose of pts1 in image 2
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        pts3d_j = pts3d_j.cpu().numpy()
        assert pts3d_j.shape[:2] == (H, W)
        msk = conf_j > self.min_conf_thr if conf_j.sum() > 0 else torch.ones_like(conf_j).bool()
        msk2 = pts3d_j[..., 2] > 1e-5   # depth > 0
        msk = msk & torch.from_numpy(msk2).to(msk.device)
        pp = pp.cpu().numpy()
        msk = msk.cpu().numpy()
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        try:
            res = cv2.solvePnPRansac(pts3d_j[msk], pixels[msk], K, None,
                                        iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
            success, R, T, inliers = res
            assert success

            R = cv2.Rodrigues(R)[0]  # world to cam
            pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
        except Exception as e:
            print(e, f'Failed to solve PnP, using identity pose for image {i}')
            pose = np.eye(4)

        self.im_poses.append(torch.from_numpy(pose.astype(np.float32)))

    def get_focals(self):
        return self.focals
    
    def get_intrinsics(self):
        focals = self.get_focals()
        pps = self.get_principal_points()
        K = torch.zeros((len(focals), 3, 3), device=self.device)
        for i in range(len(focals)):
            K[i, 0, 0] = K[i, 1, 1] = focals[i]
            K[i, :2, 2] = pps[i]
            K[i, 2, 2] = 1
        return K.cpu()
    
    def get_principal_points(self):
        return self.pp
    
    def get_im_poses(self):
        return self.im_poses
    
    def get_tum_poses(self):
        poses = self.get_im_poses()
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]

    def save_tum_poses(self, path):
        traj = self.get_tum_poses()
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses

    def save_depth(self, output_dir: str):
        """
        Saves the depth map for each camera view.
        The depth is calculated by transforming self.pts3d_j[i] (points in a model space)
        into the coordinate system of camera i, using self.im_poses[i].
        Saves raw depth as .npy and a colored .png visualization.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(self.n_imgs):
            pose_C_i_to_M_tensor = self.im_poses[i] 
            pose_M_to_C_i_tensor = inv(pose_C_i_to_M_tensor)
            model_pts_tensor = self.pts3d_j[i] 
            
            pose_M_to_C_i_tensor = pose_M_to_C_i_tensor.to(device=model_pts_tensor.device, dtype=model_pts_tensor.dtype)
            pts_in_C_i_tensor = geotrf(pose_M_to_C_i_tensor, model_pts_tensor)
            depth_map_i_tensor = pts_in_C_i_tensor[..., 2]
            
            depth_map_i_numpy = to_numpy(depth_map_i_tensor)

            # --- Save as .npy file (raw depth values) ---
            npy_filename = os.path.join(output_dir, f"frame_{i:04d}.npy")
            np.save(npy_filename, depth_map_i_numpy)

            # --- Prepare for .png visualization ---
            # Clip depth for visualization (e.g., remove negative depths)
            depth_map_i_numpy_viz = np.clip(depth_map_i_numpy, 0, None) 

            min_val = np.min(depth_map_i_numpy_viz)
            max_val = np.max(depth_map_i_numpy_viz)

            if max_val > min_val: # Avoid division by zero if all depths are the same
                depth_map_normalized = (depth_map_i_numpy_viz - min_val) / (max_val - min_val)
            else:
                # Handle cases with no depth variation (e.g., all points clipped to 0 or single depth value)
                depth_map_normalized = np.zeros_like(depth_map_i_numpy_viz)
            
            # Scale to [0, 255] and convert to uint8
            depth_map_u8 = (depth_map_normalized * 255).astype(np.uint8)
            
            # Apply color map
            depth_map_colored = cv2.applyColorMap(depth_map_u8, cv2.COLORMAP_JET)

            # --- Save as .png file (visualized depth) ---
            png_filename = os.path.join(output_dir, f"depth_{i:04d}.png")
            try:
                cv2.imwrite(png_filename, depth_map_colored)
            except Exception as e_cv:
                print(f"Warning: OpenCV cv2.imwrite failed for {png_filename}. Error: {e_cv}")
            
            print(f"Saved depth for image {i}: {npy_filename}, {png_filename}")

    def epro_pnp_solver(self, i, pose_gt=None):
        # Get 3D points and confidences for the i-th image from the current batch
        pts3d_i = self.pts3d_i[i]
        pts3d_j = self.pts3d_j[i]
        conf_i = self.conf_i[i]
        conf_j = self.conf_j[i]

        H, W = pts3d_i.shape[:2]
        pp = self.pp[i]
        focal = self.focals[i]

        # Convert pixel coordinates to a grid
        pixels = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)
        pts3d_j_np = pts3d_j.cpu().numpy()
        
        # Filter out valid points based on confidence
        if conf_j.sum() > 0:
            msk = (conf_j > self.min_conf_thr).cpu().numpy() 
        else:
            msk = np.ones_like(conf_j.cpu().numpy()) > 0
            conf_j = torch.ones_like(conf_j)
        msk2 = pts3d_j_np[..., 2] > 1e-5
        msk = msk & msk2
        pp_np = pp.cpu().numpy()
        K = np.float32([
            [focal, 0,    pp_np[0]],
            [0,     focal,pp_np[1]],
            [0,     0,    1      ]
        ])

        # Prepare input tensors
        x3d = torch.from_numpy(pts3d_j_np[msk]).float().to(self.device).unsqueeze(0)
        x2d = torch.from_numpy(pixels[msk]).float().to(self.device).unsqueeze(0)
        w2d = conf_j[msk].float().to(self.device).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 2)

        # Calculate image boundaries
        allowed_border = 30
        wh_begin = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
        out_res = torch.tensor([H, W], dtype=torch.float32, device=self.device)
        lb = wh_begin - allowed_border
        ub = wh_begin + (out_res - 1) + allowed_border

        # Build camera model
        bs = x3d.size(0)
        cam_intrinsic = torch.tensor(K, dtype=torch.float32, device=self.device)
        camera = PerspectiveCamera(
            cam_mats=cam_intrinsic[None].expand(bs, -1, -1),
            z_min=0.01,
            lb=lb,
            ub=ub
        )

        # Define cost function
        cost_fun = AdaptiveHuberPnPCost(relative_delta=0.1)
        cost_fun.set_param(x2d, w2d)

        # Call monte_carlo_forward for optimization
        pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
            x3d, x2d, w2d, camera, cost_fun,
            pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)

        # Calculate monte carlo pose loss
        if pose_gt is not None:
            loss_mc = monte_carlo_pose_loss(pose_sample_logweights, cost_tgt)
        
        monte_carlo_matrix = True
        if monte_carlo_matrix:
            weights = torch.softmax(pose_sample_logweights, dim=0)  # (mc_samples, num_obj)
            # For single object case:
            weights = weights[:, 0]  # (mc_samples,)
            pose_samples = pose_samples[:, 0]  # (mc_samples, 7)

            # Expected translation: (mc_samples, 3)
            translation_samples = pose_samples[..., :3]
            mean_translation = (weights[..., None] * translation_samples).sum(dim=0)
            # use the one with the highest weight
            # max_idx = torch.argmax(weights)
            # mean_translation = translation_samples[max_idx]

            # Expected rotation is more complex, we need to average the quaternions while ensuring the result remains on the unit sphere
            rot_samples = pose_samples[..., 3:]  # (mc_samples, 4)
            rot_samples_normalized = rot_samples / rot_samples.norm(dim=-1, keepdim=True)
            # Use weighted average to calculate the mean quaternion, then normalize it:
            weighted_rot = (weights[..., None] * rot_samples_normalized).sum(dim=0)
            weighted_rot = weighted_rot / weighted_rot.norm()

            # Final expected pose in 7D
            expected_pose = torch.cat([mean_translation, weighted_rot], dim=-1)
            expected_pose_matrix = pose7d_to_matrix(expected_pose)
            expected_pose_matrix = inv(expected_pose_matrix)
            self.im_poses.append(expected_pose_matrix)
        
        else:
            # Convert 7D pose to matrix
            pose_opt_plus_matrix = pose7d_to_matrix(pose_opt)[0] # squeeze the batch dimension
            # Invert the matrix
            pose_opt_plus_matrix = inv(pose_opt_plus_matrix)

            self.im_poses.append(pose_opt_plus_matrix)

        # Return the optimization result
        return pose_opt_plus, pose_sample_logweights, cost_tgt


def oneref_viewer_wrapper(folder_path: str, share_focal: bool=False, focal_pair1: bool=False, min_conf_thr: float=1.5, epro_pnp: bool=False, save_depth: bool=False) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the data
    ptsi_3d_paths = sorted(glob(folder_path + '/pts3d1_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
    ptsj_3d_paths = sorted(glob(folder_path + '/pts3d2_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
    confi_paths = sorted(glob(folder_path + '/conf1_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
    confj_paths = sorted(glob(folder_path + '/conf2_p*.npy'), key=lambda x: int(x.split('_p')[-1].split('.')[0]))
    ptsi_3d = torch.stack([torch.from_numpy(np.load(p)) for p in ptsi_3d_paths], dim=0)[..., :3].to(device)
    ptsj_3d = torch.stack([torch.from_numpy(np.load(p)) for p in ptsj_3d_paths], dim=0)[..., :3].to(device)
    # ptsi_3d = torch.from_numpy(np.load(folder_path + '/pts3d1.npy'))[..., :3].to(device)
    # ptsj_3d = torch.from_numpy(np.load(folder_path + '/pts3d2.npy'))[..., :3].to(device)
    confi = torch.stack([torch.from_numpy(np.load(p)) for p in confi_paths], dim=0).to(device) if len(confi_paths) > 0 else None
    confj = torch.stack([torch.from_numpy(np.load(p)) for p in confj_paths], dim=0).to(device) if len(confj_paths) > 0 else None
    viewer = OneRefViewer(ptsi_3d, ptsj_3d, confi, confj,
                          share_focal=share_focal,
                          focal_pair1=focal_pair1,
                          min_conf_thr=min_conf_thr, epro_pnp=epro_pnp)
    # save the intrinsic, and pose
    intrinsics = viewer.get_intrinsics()
    if epro_pnp:
        intrinsics_save_path = folder_path + '/pred_intrinsics_epro.txt'
    else:
        intrinsics_save_path = folder_path + '/pred_intrinsics.txt'
    np.savetxt(intrinsics_save_path, intrinsics.numpy().reshape(viewer.n_imgs, -1))
    poses = viewer.get_im_poses()
    if epro_pnp:
        poses_save_path = folder_path + '/pred_traj_matrix_epro.txt'
    else:
        poses_save_path = folder_path + '/pred_traj_matrix.txt'
    np.savetxt(poses_save_path, torch.stack(poses).cpu().numpy().reshape(viewer.n_imgs, -1))
    if epro_pnp:
        tum_pose_save_path = folder_path + '/pred_traj_epro.txt'
    else:
        tum_pose_save_path = folder_path + '/pred_traj.txt'
    tum_poses = viewer.save_tum_poses(tum_pose_save_path)

    if save_depth:
        depth_output_dir = os.path.join(folder_path) # Changed dir name slightly
        viewer.save_depth(depth_output_dir)

    return intrinsics_save_path, poses_save_path, tum_pose_save_path, tum_pose_save_path

if __name__ == '__main__':
    import tyro
    tyro.cli(oneref_viewer_wrapper)
