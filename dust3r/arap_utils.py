import torch
import torch.nn.functional as F

def arap_loss_patch(pred_pts3d, neighbor_offsets=None):
    """
    Compute a vectorized ARAP loss using local patches (neighbor windows) across 
    consecutive frames. This loss encourages each local patch to be well–approximated 
    by a rigid (rotational) transformation between frames.
    
    Args:
        pred_pts3d: Tensor of shape [B, T, 3, H, W] containing the 3D coordinates.
                    pred_pts3d[b, t, :, h, w] is the (x,y,z) coordinate of pixel (h,w)
                    at time t.
        neighbor_offsets: List of (dh, dw) offsets defining the neighborhood.  
                          For example, a 4-neighborhood can be given as:
                          [(1,0), (-1,0), (0,1), (0,-1)].
                          The center pixel (0,0) is automatically included if not provided.
                          
    Returns:
        total_loss: A scalar tensor with the ARAP loss.
    """
    if neighbor_offsets is None:
        neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    # Always include the center pixel.
    if (0, 0) not in neighbor_offsets:
        neighbor_offsets = [(0, 0)] + neighbor_offsets

    B, T, C, H, W = pred_pts3d.shape
    assert C == 3, "Expected channel dimension of size 3 for XYZ coordinates."

    # We compute the loss between consecutive frame pairs.
    X_t  = pred_pts3d[:, :-1]  # shape: [B, T-1, 3, H, W]
    X_t1 = pred_pts3d[:, 1:]   # shape: [B, T-1, 3, H, W]

    # For each neighbor offset, we shift the frames and extract a patch.
    patches_t_list  = []
    patches_t1_list = []
    for (dh, dw) in neighbor_offsets:
        # Determine the required padding (only for H, W).
        pad_left   = max(dw, 0)
        pad_right  = max(-dw, 0)
        pad_top    = max(dh, 0)
        pad_bottom = max(-dh, 0)
        # Pad X_t and X_t1 along the spatial dimensions.
        # F.pad accepts a 4-tuple (left, right, top, bottom) and applies it to the last two dims.
        X_t_padded  = F.pad(X_t,  (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        X_t1_padded = F.pad(X_t1, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        
        # Compute starting indices for the crop.
        start_h = pad_top - dh
        start_w = pad_left - dw
        # Crop to obtain patches of shape [B, T-1, 3, H, W].
        patch_t  = X_t_padded[:, :, :, start_h:start_h+H, start_w:start_w+W]
        patch_t1 = X_t1_padded[:, :, :, start_h:start_h+H, start_w:start_w+W]
        
        # Add a new dimension for the neighbor index.
        patches_t_list.append(patch_t.unsqueeze(2))   # Now shape: [B, T-1, 1, 3, H, W]
        patches_t1_list.append(patch_t1.unsqueeze(2))
    
    # Stack all neighbor patches along a new dimension.
    # Final shape: [B, T-1, n, 3, H, W], where n is the number of neighbor offsets.
    patches_t  = torch.cat(patches_t_list, dim=2)
    patches_t1 = torch.cat(patches_t1_list, dim=2)
    
    # Compute the local centroids over the neighbor dimension.
    centroid_t  = patches_t.mean(dim=2, keepdim=True)   # [B, T-1, 1, 3, H, W]
    centroid_t1 = patches_t1.mean(dim=2, keepdim=True)   # [B, T-1, 1, 3, H, W]
    
    # Center the patches.
    patches_t_centered  = patches_t  - centroid_t    # [B, T-1, n, 3, H, W]
    patches_t1_centered = patches_t1 - centroid_t1   # [B, T-1, n, 3, H, W]
    
    # Rearrange dimensions so that the neighbor and channel dimensions come last.
    # New shape: [B, T-1, H, W, n, 3]
    p = patches_t_centered.permute(0, 1, 4, 5, 2, 3)
    q = patches_t1_centered.permute(0, 1, 4, 5, 2, 3)
    
    # Compute the 3x3 cross-covariance matrix per pixel (over the neighbor dimension).
    # Using Einstein summation: for each pixel, cov = sum_i (p_i^T * q_i)
    cov = torch.einsum('bthwni,bthwnj->bthwij', p, q)  # Shape: [B, T-1, H, W, 3, 3]
    
    # Perform batched SVD on the covariance matrices.
    U, S, Vh = torch.linalg.svd(cov)
    # Compute rotation matrices: R = Vh^T @ U^T.
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, T-1, H, W, 3, 3]
    
    # Correct for improper rotations (ensure det(R) == +1).
    det_R = torch.det(R)  # Shape: [B, T-1, H, W]
    sign = torch.sign(det_R).unsqueeze(-1).unsqueeze(-1)  # [B, T-1, H, W, 1, 1]
    Vh_corrected = Vh.clone()
    # Multiply the last row of Vh by the sign for pixels with negative determinant.
    Vh_corrected[..., -1, :] = Vh_corrected[..., -1, :] * sign.squeeze(-1).squeeze(-1)
    R = torch.matmul(Vh_corrected.transpose(-2, -1), U.transpose(-2, -1))
    
    # Apply the computed rotation to align p to q.
    # p has shape [B, T-1, H, W, n, 3] and R has shape [B, T-1, H, W, 3, 3].
    p_aligned = torch.matmul(p, R.transpose(-2, -1))  # [B, T-1, H, W, n, 3]
    diff = q - p_aligned  # Alignment error, shape: [B, T-1, H, W, n, 3]
    
    # Compute squared error over the coordinate dimension.
    error = (diff ** 2).sum(dim=-1)  # Shape: [B, T-1, H, W, n]
    # Average error over the neighbor dimension.
    error = error.mean(dim=-1)  # Shape: [B, T-1, H, W]
    
    # Finally, average over batch, time, and spatial dimensions.
    total_loss = error.mean()
    return total_loss



def arap_loss_patch_masked(pred_pts3d, mask, neighbor_offsets=None, eps=1e-6):
    """
    Compute a vectorized, patch–based ARAP loss for 3D point cloud sequences with masking.
    Only points within the human (valid) region are used to compute the loss.
    
    Args:
        pred_pts3d: Tensor of shape [B, T, 3, H, W] containing the 3D coordinates.
        mask: Binary mask indicating the human region.
              Shape can be [B, T, 1, H, W] or [B, T, H, W] (1: valid/human, 0: background).
        neighbor_offsets: List of (dh, dw) offsets defining the neighborhood.
                          For example, a 4-neighborhood: [(1,0), (-1,0), (0,1), (0,-1)].
                          The center (0,0) is automatically included if not provided.
        eps: A small constant to avoid division by zero.
    
    Returns:
        total_loss: A scalar tensor representing the ARAP loss averaged over valid patches.
    """
    # Default neighbor offsets: 4-neighborhood.
    if neighbor_offsets is None:
        neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if (0, 0) not in neighbor_offsets:
        neighbor_offsets = [(0, 0)] + neighbor_offsets

    B, T, C, H, W = pred_pts3d.shape
    assert C == 3, "pred_pts3d channel must be 3 (XYZ coordinates)"
    
    # Ensure mask has shape [B, T, 1, H, W]
    if mask.ndim == 4:
        mask = mask.unsqueeze(2)
    
    # We compute the loss over consecutive frame pairs.
    # For time t and t+1:
    X_t  = pred_pts3d[:, :-1]  # [B, T-1, 3, H, W]
    X_t1 = pred_pts3d[:, 1:]   # [B, T-1, 3, H, W]
    mask_t  = mask[:, :-1]     # [B, T-1, 1, H, W]
    mask_t1 = mask[:, 1:]      # [B, T-1, 1, H, W]

    # Lists to store neighbor patches for the point clouds and masks.
    patches_t_list = []
    patches_t1_list = []
    mask_patches_t_list = []
    mask_patches_t1_list = []
    
    # For each neighbor offset, shift and crop the tensors.
    for (dh, dw) in neighbor_offsets:
        # Determine how much to pad in each spatial direction.
        pad_left   = max(dw, 0)
        pad_right  = max(-dw, 0)
        pad_top    = max(dh, 0)
        pad_bottom = max(-dh, 0)
        
        # For the point clouds, we use replicate padding.
        X_t_padded  = F.pad(X_t,  (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        X_t1_padded = F.pad(X_t1, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        # For the mask, we pad with a constant 0 (so that out–of–bounds regions are considered background).
        mask_t_padded  = F.pad(mask_t,  (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        mask_t1_padded = F.pad(mask_t1, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Compute the crop indices.
        start_h = pad_top - dh
        start_w = pad_left - dw
        
        # Crop to get patches of the original spatial size.
        patch_t  = X_t_padded[:, :, :, start_h:start_h+H, start_w:start_w+W]   # [B, T-1, 3, H, W]
        patch_t1 = X_t1_padded[:, :, :, start_h:start_h+H, start_w:start_w+W]  # [B, T-1, 3, H, W]
        m_patch_t  = mask_t_padded[:, :, :, start_h:start_h+H, start_w:start_w+W]  # [B, T-1, 1, H, W]
        m_patch_t1 = mask_t1_padded[:, :, :, start_h:start_h+H, start_w:start_w+W] # [B, T-1, 1, H, W]
        
        # Add a new neighbor dimension.
        patches_t_list.append(patch_t.unsqueeze(2))       # Now: [B, T-1, 1, 3, H, W]
        patches_t1_list.append(patch_t1.unsqueeze(2))
        mask_patches_t_list.append(m_patch_t.unsqueeze(2))  # Now: [B, T-1, 1, 1, H, W]
        mask_patches_t1_list.append(m_patch_t1.unsqueeze(2))
    
    # Stack along the neighbor dimension.
    patches_t  = torch.cat(patches_t_list, dim=2)   # [B, T-1, n, 3, H, W]
    patches_t1 = torch.cat(patches_t1_list, dim=2)   # [B, T-1, n, 3, H, W]
    mask_patches_t  = torch.cat(mask_patches_t_list, dim=2)   # [B, T-1, n, 1, H, W]
    mask_patches_t1 = torch.cat(mask_patches_t1_list, dim=2)   # [B, T-1, n, 1, H, W]
    
    # Only consider pixels that are valid in BOTH frames.
    valid_mask = mask_patches_t * mask_patches_t1  # [B, T-1, n, 1, H, W]
    
    # Compute weighted (valid–only) centroids for each patch over the neighbor dimension.
    sum_valid = torch.sum(valid_mask, dim=2, keepdim=True)  # [B, T-1, 1, 1, H, W]
    centroid_t  = torch.sum(patches_t * valid_mask, dim=2, keepdim=True) / (sum_valid + eps)  # [B, T-1, 1, 3, H, W]
    centroid_t1 = torch.sum(patches_t1 * valid_mask, dim=2, keepdim=True) / (sum_valid + eps)  # [B, T-1, 1, 3, H, W]
    
    # Center the patches (only valid points contribute).
    patches_t_centered  = (patches_t - centroid_t) * valid_mask  # [B, T-1, n, 3, H, W]
    patches_t1_centered = (patches_t1 - centroid_t1) * valid_mask  # [B, T-1, n, 3, H, W]
    
    # Rearrange dimensions to bring the neighbor dimension to the end:
    # New shape: [B, T-1, H, W, n, 3]
    p = patches_t_centered.permute(0, 1, 4, 5, 2, 3)
    q = patches_t1_centered.permute(0, 1, 4, 5, 2, 3)
    valid_mask_perm = valid_mask.permute(0, 1, 4, 5, 2, 3)  # [B, T-1, H, W, n, 1]
    
    # Compute the 3x3 covariance matrix per spatial location by summing over neighbors.
    cov = torch.einsum('bthwni,bthwnj->bthwij', p, q)  # [B, T-1, H, W, 3, 3]
    
    # Compute SVD on the covariance matrices.
    U, S, Vh = torch.linalg.svd(cov)
    # Compute the candidate rotation: R = Vh^T @ U^T.
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))  # [B, T-1, H, W, 3, 3]
    
    # Correct any improper rotations (ensure det(R) == +1).
    det_R = torch.det(R)  # [B, T-1, H, W]
    sign = torch.sign(det_R).unsqueeze(-1).unsqueeze(-1)  # [B, T-1, H, W, 1, 1]
    Vh_corrected = Vh.clone()
    Vh_corrected[..., -1, :] = Vh_corrected[..., -1, :] * sign.squeeze(-1).squeeze(-1)
    R = torch.matmul(Vh_corrected.transpose(-2, -1), U.transpose(-2, -1))
    
    # Rotate the centered patch p to align with q.
    p_aligned = torch.matmul(p, R.transpose(-2, -1))  # [B, T-1, H, W, n, 3]
    diff = q - p_aligned  # [B, T-1, H, W, n, 3]
    
    # Compute squared error per neighbor.
    sq_error = (diff ** 2).sum(dim=-1)  # [B, T-1, H, W, n]
    # Squeeze the valid mask (from shape [B, T-1, H, W, n, 1] to [B, T-1, H, W, n]).
    valid_mask_squeezed = valid_mask_perm.squeeze(-1)
    # Sum the error over the neighbor dimension and normalize by the number of valid neighbors.
    error_patch = torch.sum(sq_error, dim=-1) / (torch.sum(valid_mask_squeezed, dim=-1) + eps)  # [B, T-1, H, W]
    
    # Create an indicator for patches that have at least 2 valid neighbors.
    valid_count = torch.sum(valid_mask_squeezed, dim=-1)  # [B, T-1, H, W]
    valid_indicator = (valid_count >= 2).float()  # [B, T-1, H, W]
    # Zero out error in patches with too few valid points.
    error_patch = error_patch * valid_indicator
    
    # Average the loss over only the valid patches.
    total_valid = torch.sum(valid_indicator) + eps
    total_loss = torch.sum(error_patch) / total_valid
    
    return total_loss

def distance_preservation_loss_3d(pred_pts3d, k_neighbors=8):
    """
    Optimized version of distance preservation loss using K-nearest neighbors in 3D space.
    Vectorized across time dimension to avoid for-loop.
    
    Args:
        pred_pts3d: Tensor of shape [B, T, 3, H, W] representing 3D point clouds
                   with temporal correspondence.
        k_neighbors: Number of nearest neighbors to consider for each point.
                    
    Returns:
        dist_loss: A scalar tensor representing the loss.
    """
    B, T, C, H, W = pred_pts3d.shape
    assert C == 3, "Expected 3D points"
    N = H * W
    
    # Reshape points to [B, T, N, 3]
    pts = pred_pts3d.flatten(3).permute(0, 1, 3, 2)
    
    # Compute KNN only on the first frame
    pts_first = pts[:, 0]  # [B, N, 3]
    
    # Compute pairwise distances
    diff = pts_first.unsqueeze(2) - pts_first.unsqueeze(1)  # [B, N, N, 3]
    dist_matrix = torch.norm(diff, dim=-1)  # [B, N, N]
    
    # Get k+1 nearest neighbors (including self) and remove self
    neighbor_idx = torch.topk(dist_matrix, k=k_neighbors+1, dim=-1, largest=False)[1][..., 1:]
    neighbor_idx = neighbor_idx.to(torch.int32)  # Memory optimization # shape [B, N, k]
    
    # Pre-compute batch indices for gathering
    batch_idx = torch.arange(B, device=pred_pts3d.device)[:, None, None]
    batch_idx = batch_idx.expand(-1, N, k_neighbors)
    flat_idx = (batch_idx * N).reshape(-1, k_neighbors) + neighbor_idx.reshape(-1, k_neighbors)  # [B*N, k]
    
    # Get points with half sequence gap
    pts_t = pts[:, :-T//2]    # [B, T-T//2, N, 3]
    pts_t1 = pts[:, T//2:]    # [B, T-T//2, N, 3]
    
    # Reshape for efficient gathering
    pts_t_flat = pts_t.reshape(-1, 3)    # [B*(T-T//2)*N, 3]
    pts_t1_flat = pts_t1.reshape(-1, 3)  # [B*(T-T//2)*N, 3]
    
    # Expand flat_idx for all timesteps
    time_idx = torch.arange(T-T//2, device=pred_pts3d.device)[:, None, None]  # [T-T//2, 1, 1]
    time_offset = time_idx * (B * N)  # [T-T//2, 1, 1]
    flat_idx_expanded = flat_idx.unsqueeze(0) + time_offset  # [T-T//2, B*N, k]
    flat_idx_expanded = flat_idx_expanded.reshape(-1, k_neighbors)  # [(T-T//2)*B*N, k]
    
    # Gather neighbors for all timesteps at once
    neighbors_t = torch.index_select(pts_t_flat, 0, flat_idx_expanded.reshape(-1))
    neighbors_t = neighbors_t.reshape(B, T-T//2, N, k_neighbors, 3)
    
    neighbors_t1 = torch.index_select(pts_t1_flat, 0, flat_idx_expanded.reshape(-1))
    neighbors_t1 = neighbors_t1.reshape(B, T-T//2, N, k_neighbors, 3)
    
    # Compute Euclidean distances for all timesteps at once
    dist_t = torch.norm(neighbors_t - pts_t.unsqueeze(3), dim=-1)    # [B, T-T//2, N, k]
    dist_t1 = torch.norm(neighbors_t1 - pts_t1.unsqueeze(3), dim=-1) # [B, T-T//2, N, k]

    # Compute loss across all dimensions at once
    loss = torch.sqrt((dist_t1 - dist_t).pow(2).mean())
    
    return loss
    
# ------------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Example: a batch of human body motion sequences.
    B, T, H, W = 1, 16, 512, 288  # 8 sequences, 30 frames each, 64x64 resolution.
    # Simulated predicted 3D points.
    pred_pts3d = torch.randn(B, T, 3, H, W, requires_grad=True).cuda()

    
    # Simulated binary mask (here, randomly generated for illustration;
    # in practice, use your actual human-region mask where 1=human, 0=background).
    mask = (torch.rand(B, T, H, W) > 0.5).float()  # shape: [B, T, H, W]

    # Subsample points with random shift for variation
    rand_offset = torch.randint(0, 4, (1,), device=pred_pts3d.device)[0]
    pred_pts3d = pred_pts3d[..., rand_offset::4, rand_offset::4]
    loss = distance_preservation_loss_3d(pred_pts3d)
    # loss = arap_loss_patch_masked(pred_pts3d, mask)
    loss.backward()
    
    print("Masked ARAP patch-based loss =", loss.item())