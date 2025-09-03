import torch

def build_knn_adjacency(points, k=16, device="cpu"):
    """
    Build K-NN adjacency matrix with batch support.

    Args:
      points (torch.Tensor): (B, N, 3)
      k (int): Number of neighbors per point (excluding self)
      device (str): 'cpu' or 'cuda'

    Returns:
      adjacency (torch.Tensor): (B, N, k) neighbor indices for each point
    """
    points = points.to(device)  # (B, N, 3)
    B, N, _ = points.shape

    # Compute pairwise distances within each batch
    dist_matrix = torch.norm(points[:, :, None, :] - points[:, None, :, :], dim=-1)

    # Get k+1 nearest neighbors (including self)
    knn_indices = dist_matrix.topk(k=k+1, largest=False, dim=2).indices  # (B, N, k+1)

    # Remove self index
    adjacency = knn_indices[:, :, 1:]
    return adjacency


def compute_normals_from_adjacency_vec(points, adjacency):
    """
    Vectorized normal estimation using shared adjacency structure.

    Args:
      points (torch.Tensor): (B, N, 3)
      adjacency (torch.Tensor): (B, N, k) - neighbor indices for each point

    Returns:
      normals (torch.Tensor): (B, N, 3) normalized local patch normals
    """
    device = points.device
    B, N, _ = points.shape
    _, _, k = adjacency.shape

    if k < 2:
        return torch.zeros_like(points)

    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    neighbors = points[batch_indices, adjacency]  # (B, N, k, 3)

    center_expanded = points.unsqueeze(2)  # (B, N, 1, 3)
    edges = neighbors - center_expanded

    v0 = edges
    v1 = torch.roll(edges, shifts=-1, dims=2)
    cross_vals = torch.cross(v0, v1, dim=3)     # (B, N, k, 3)

    patch_normal = cross_vals.mean(dim=2)

    norm_len = patch_normal.norm(dim=2, keepdim=True)  # (B, N, 1)
    eps = 1e-12
    normals = patch_normal / (norm_len + eps)

    zero_mask = (norm_len < eps).squeeze(-1)  # (B, N)
    normals[zero_mask] = 0

    return normals


def normal_consistency_loss(pred_normals, gt_normals):
    """
    Compute normal consistency loss:
      Mean of 1 - |dot(n_pred, n_gt)| for each point.

    Args:
      pred_normals (torch.Tensor): (B, N, 3)
      gt_normals   (torch.Tensor): (B, N, 3)

    Returns:
      Loss scalar: 0 when normals are parallel or anti-parallel
    """
    dot = torch.sum(pred_normals * gt_normals, dim=2)  # (B, N)
    dot_clamped = torch.clamp(dot, -1.0, 1.0)
    return torch.mean(1.0 - torch.abs(dot_clamped))


def get_normal_loss(
    pred_points,
    gt_points,
    k=8,
    device="cpu"
):
    """
    Compute geometric loss:
      1) Build adjacency matrix based on GT point cloud
      2) Estimate normals for both predicted and ground truth points using the same adjacency
      3) Return normal consistency loss

    Args:
      pred_points (torch.Tensor): (B, N, 3) predicted point cloud
      gt_points   (torch.Tensor): (B, N, 3) ground truth point cloud
      k (int): number of neighbors
      device (str): 'cpu' or 'cuda'

    Returns:
      Normal consistency loss (scalar)
    """
    pred_points = pred_points.to(device)
    gt_points   = gt_points.to(device)
    
    adjacency = build_knn_adjacency(gt_points, k=k, device=device)  # (B, N, k)

    pred_normals = compute_normals_from_adjacency_vec(pred_points, adjacency)  # (B, N, 3)
    gt_normals   = compute_normals_from_adjacency_vec(gt_points, adjacency)    # (B, N, 3)

    nor_l = normal_consistency_loss(pred_normals, gt_normals)
    return nor_l
