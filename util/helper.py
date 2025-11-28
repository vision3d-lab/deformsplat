import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import DBSCAN

from jhutil.algorithm import knn
from jhutil import cache_output
from .loss import arap_loss, drag_loss
from .mini_pytorch3d import (
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)

import torch_fpsample


def load_points_and_anchor(ckpt_path):
    ckpt = torch.load(ckpt_path)
    if "clustered" not in ckpt:
        ckpt["splats"] = cluster_largest(ckpt["splats"])
        ckpt["clustered"] = True
        torch.save(ckpt, ckpt_path)

    points = ckpt["splats"]["means"]
    # anchor = voxelize_pointcloud_and_get_means(points, 0.04)
    anchor, _ = torch_fpsample.sample(points.detach().cpu(), 500)

    return points, anchor


def save_points_and_anchor(points, anchor, ckpt_path):
    ckpt = {"splats": {"means": points, "anchor": anchor}}
    torch.save(ckpt, ckpt_path)


def make_simple_goal(anchor, random_rotation=False):
    if random_rotation:
        # random indice
        indices_goal = torch.randint(0, len(anchor), (30,))
        anchor_goal = anchor[indices_goal].clone()
        # rdom rotation
        q = torch.randn(4, device=anchor.device)
        q = F.normalize(q, dim=-1)
        R = quaternion_to_matrix(q)
        anchor_goal = anchor_goal @ R.T
    else:
        indices_goal = torch.concat(
            [torch.tensor([0, len(anchor) - 1]), torch.arange(100, 250)]
        )
        anchor_goal = anchor[indices_goal].clone()
        anchor_goal[0:2] = torch.tensor([[-0.05, -0.15, -0.15], [0.15, -0.155, 0.25]])

    return indices_goal, anchor_goal


def rbf_weight(distances, gamma):
    k = torch.exp(-gamma * distances)
    return k / (k.sum(dim=1, keepdim=True) + 1e-8)


def deform_point_cloud_arap(
    anchor,  # (N, 3) 원본 point cloud
    control_indices,  # 드래그를 적용할 point의 인덱스 (예: 고정점이나, 일부 point)
    anchor_goal,  # 드래그 목표 위치 (drag_indices에 해당하는 점들의 목표)
    num_iterations=500,
    coef_arap=1,
    coef_drag=5e1,
    lr=3e-2,
):
    """
    ARAP + drag 손실을 동시에 최소화하여 x_origin을 deform하는 예시 코드.
    R: (N, 3, 3), t: (N, 3)를 학습 변수로 둔다.
    """

    with torch.no_grad():
        distances, indices_knn = knn(anchor, anchor, k=5)
        weight = rbf_weight(distances, gamma=30)

    device = anchor.device
    N = anchor.shape[0]

    # -----------------------------
    # 1. 학습 파라미터 정의
    # -----------------------------
    # R은 처음에 항등행렬로 초기화(각 포인트별)
    # t는 0 벡터로 초기화
    q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
    q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
    t_init = torch.zeros((N, 3), device=device)

    q = nn.Parameter(q_init)  # (N, 3, 3)
    t = nn.Parameter(t_init)  # (N, 3)

    # 최적화 알고리즘(Adam)
    optimizer = optim.Adam([
        {'params': q, 'lr': lr},        # Learning rate for R
        {'params': t, 'lr': lr * 0.1}  # Learning rate for t (e.g., 10 times smaller)
    ])

    # -----------------------------
    # 2. 최적화 루프
    # -----------------------------
    for step in range(num_iterations):
        # 2-1. 현재 변형된 x_current 계산
        #    x_current[i] = R[i] x_origin[i] + t[i]
        #    (N, 3, 3) x (N, 3) -> broadcasting에 주의
        R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
        anchor_translated = anchor + t  # (N, 3)

        loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)
        anchor_control = anchor_translated[control_indices]  # 실제 드래그가 필요한 점들
        loss_drag = drag_loss(anchor_control, anchor_goal)
        loss = coef_arap * loss_arap + coef_drag * loss_drag

        # 2-3. Backprop 및 optimizer.step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{num_iterations}] - Arap loss: {loss_arap.item():.6f}   Drag loss: {loss_drag.item():.6f}")

    return R.detach(), t.detach()


def voxelize_pointcloud_and_get_means(points: torch.Tensor, voxel_size: float):
    """
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        voxel_size (float): Size of each voxel along each dimension.

    Returns:
        voxel_means (torch.Tensor): Mean positions per voxel, shape = (M, 3),
                                    where M is the number of occupied voxels.
        voxel_coords (torch.LongTensor): The integer voxel indices (i, j, k)
                                         for each mean in `voxel_means`.
                                         Shape = (M, 3).
    """
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)

    # 2. Compute voxel indices [i, j, k] for each point
    #    i, j, k = floor((x - min_x) / voxel_size)
    voxel_indices = torch.floor((points - min_coords) / voxel_size).long()
    # voxel_indices: shape (N, 3), each row is (i, j, k)

    # 3. Convert voxel indices into a single flattened index for grouping
    grid_size = torch.ceil((max_coords - min_coords) / voxel_size).long()
    Nx, Ny, Nz = grid_size.tolist()

    # Make sure the computed indices do not exceed Nx-1, Ny-1, Nz-1
    # (In case a point lies exactly on max_coords boundary)
    voxel_indices[:, 0] = torch.clamp(voxel_indices[:, 0], 0, Nx - 1)
    voxel_indices[:, 1] = torch.clamp(voxel_indices[:, 1], 0, Ny - 1)
    voxel_indices[:, 2] = torch.clamp(voxel_indices[:, 2], 0, Nz - 1)

    # Flattened voxel index: i + j*Nx + k*Nx*Ny
    flattened_indices = (
        voxel_indices[:, 0] + voxel_indices[:, 1] * Nx + voxel_indices[:, 2] * Nx * Ny
    )

    # 4. Accumulate sums per voxel and counts per voxel
    #    We'll do it manually here; for large-scale data you can use torch_scatter.
    num_points = points.shape[0]
    device = points.device

    # Number of possible voxels = Nx * Ny * Nz
    total_voxels = Nx * Ny * Nz

    sums = torch.zeros((total_voxels, 3), device=device, dtype=points.dtype)
    counts = torch.zeros(total_voxels, device=device, dtype=points.dtype)

    # Accumulate sums & counts
    sums.index_add_(0, flattened_indices, points)
    counts.index_add_(
        0, flattened_indices, torch.ones(num_points, device=device, dtype=points.dtype)
    )

    # Avoid division by zero by masking out empty voxels
    non_empty_mask = counts > 0

    # 5. Compute mean per voxel
    voxel_means_all = torch.zeros_like(sums)
    voxel_means_all[non_empty_mask] = sums[non_empty_mask] / counts[
        non_empty_mask
    ].unsqueeze(-1)

    # We only return occupied voxels (those with non-empty_mask)
    occupied_indices = torch.nonzero(non_empty_mask, as_tuple=True)[0]  # shape (M,)
    voxel_means = voxel_means_all[occupied_indices]  # shape (M, 3)

    # Convert flattened voxel idx -> (i, j, k) for the occupied voxels
    k = occupied_indices // (Nx * Ny)
    rem = occupied_indices % (Nx * Ny)
    j = rem // Nx
    i = rem % Nx
    # voxel_coords = torch.stack((i, j, k), dim=1)  # shape (M, 3)
    # voxel_means = torch.load("/tmp/.cache/new_anchor.pt")

    return voxel_means


def linear_blend_skinning_knn(
    points: torch.Tensor,  # (N, 3)  모든 Point의 원본 위치 (mu_j)
    anchor_positions: torch.Tensor,  # (M, 3)  Anchor들의 위치 (p_k)
    R: torch.Tensor,  # (M, 3, 3)  각 Anchor의 회전행렬 (R_k^t)
    t: torch.Tensor,  # (M, 3)  각 Anchor의 평행 이동 (T_k^t)
    k: int = 5,
    gamma: float = 30.0,
):
    """
    KNN + Linear Blend Skinning
      1) Point LBS:  mu_j^t = ∑ w_jk [ R_k^t(mu_j - p_k) + p_k + T_k^t ]
      2) Quaternion LBS: r_j^t = normalize( ∑ w_jk * q_k^t )
         단, q_k^t는 3x3 R_k^t로부터 matrix_to_quaternion()을 통해 on-the-fly로 변환
    """
    # --------------------------------------------------
    # [1] KNN으로 Anchor 중 가장 가까운 k개의 인덱스, 거리 구하기
    # [2] RBF 가중치 계산
    # --------------------------------------------------
    with torch.no_grad():
        # TODO: replace this with djasta distance
        # distances, indices_knn = knn_djastra(points, anchor_positions, k=k)
        distances, indices_knn = knn(points, anchor_positions, k=k)

        weights = rbf_weight(distances, gamma=gamma)  # (N, k)
        w_expanded = weights.unsqueeze(-1)  # (N, k, 1)

    # --------------------------------------------------
    # [3] LBS (Point)
    #     mu_j^t = sum_k w_jk * [ R_k^t(mu_j - p_k) + p_k + T_k^t ]
    # --------------------------------------------------
    anchor_closest = anchor_positions[indices_knn]  # (N, k, 3)
    R_closest = R[indices_knn]  # (N, k, 3, 3)
    t_closest = t[indices_knn]  # (N, k, 3)

    # (mu_j - p_k)
    points_expanded = points.unsqueeze(1)  # (N, 1, 3)
    centered = points_expanded - anchor_closest  # (N, k, 3)

    # R_k^t(mu_j - p_k)
    N, k_ = R_closest.shape[:2]  # N, k
    centered_reshaped = centered.reshape(N * k_, 3, 1)
    R_reshaped = R_closest.reshape(N * k_, 3, 3)

    rotated_flat = torch.bmm(R_reshaped, centered_reshaped)  # (N*k,3,1)
    rotated = rotated_flat.reshape(N, k_, 3)  # (N, k, 3)

    # R_k^t(...) + p_k + T_k^t
    transformed = rotated + anchor_closest + t_closest  # (N, k, 3)

    # 가중치 blend
    skinned_points = (transformed * w_expanded).sum(dim=1)  # (N, 3)

    # --------------------------------------------------
    # [4] LBS (Quaternion)
    #     r_j^t = normalize( ∑ w_jk * q_k^t ),
    # --------------------------------------------------
    # 4-1) 먼저, R_k^t → Quaternion
    quaternion_reshaped = matrix_to_quaternion(R_reshaped)  # (N*k, 4)
    quaternion_closest = quaternion_reshaped.reshape(N, k_, 4)  # (N, k, 4)

    # 4-2) 가중치 곱 + 합
    blended_quats = (quaternion_closest * w_expanded).sum(dim=1)  # (N, 4)

    # 4-3) L2 노름으로 정규화
    blended_quats = blended_quats / (
        blended_quats.norm(dim=1, keepdim=True) + 1e-8
    )  # (N, 4)

    return skinned_points, blended_quats


def cluster_largest(splat, eps=0.05, min_samples=5):

    point_cloud = splat["means"]
    points_np = point_cloud.cpu().numpy()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    labels = dbscan.fit_predict(points_np)

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = unique_labels >= 0
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]

    largest_cluster_label = valid_labels[np.argmax(valid_counts)]
    largest_cluster_mask = labels == largest_cluster_label
    # assert largest_cluster_mask.sum() > 0.9 * len(point_cloud)
    print(f"{len(point_cloud) - largest_cluster_mask.sum()} points removed")

    for key in splat:
        splat[key] = splat[key][largest_cluster_mask]

    return splat


@torch.no_grad()
def get_target_indices_drag(sparse_2d, dense_2d, depth):
    assert sparse_2d.dim() == 2
    assert dense_2d.dim() == 2
    assert depth.dim() == 1

    distances, knn_indices = knn(sparse_2d, dense_2d, k=5)

    candidate_depths = depth[knn_indices]

    min_depth_indices = candidate_depths.argmin(dim=1)

    num_queries = knn_indices.shape[0]
    nearest_indices = knn_indices[torch.arange(num_queries), min_depth_indices]

    return nearest_indices


@torch.no_grad()
def get_visible_mask_by_depth(points_2d, depth, H, W):

    visible_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    points_2d_visible = points_2d[visible_mask]
    depth_visible = depth[visible_mask]

    front_mask = get_front_mask_by_depth(points_2d_visible, depth_visible)
    visible_mask[visible_mask == True] = front_mask

    return visible_mask


@torch.no_grad()
def get_drag_mask(points_2d, points_mask, drag_source, filter_distance, one_to_one=False):

    # points_2d : (N, 2)
    # points_mask : (N, )
    # drag_source : (M, 2)
    filtered_points_2d = points_2d[points_mask]  # (n, 2)
    distances, nearest_indices = knn(
        filtered_points_2d.detach(), drag_source.detach(), k=1
    )  # (n, 1), (n, 1)

    distances = distances.squeeze()  # (n,)
    nearest_indices = nearest_indices.squeeze()  # (n,)

    if one_to_one:
        for i in range(0, nearest_indices.max() + 1):
            mask = nearest_indices == i
            if mask.sum() > 1:
                min_idx = mask.nonzero(as_tuple=True)[0][distances[mask].argmin()]
                distances[mask] = 100
                distances[min_idx] = 0

    points_mask[points_mask == True] = distances < filter_distance
    drag_indice = nearest_indices[distances < filter_distance]   # (m, )

    return points_mask, drag_indice

@torch.no_grad()
def get_front_mask_by_depth(points_2d, depth):

    w = points_2d[:, 0].max().int().item() + 1
    h = points_2d[:, 1].max().int().item() + 1

    max_pixel_depth = torch.full((h * w,), -float("inf"), device=depth.device)
    min_pixel_depth = torch.full((h * w,), float("inf"), device=depth.device)

    linear_idx = points_2d[:, 1].long() * w + points_2d[:, 0].long()

    max_pixel_depth.scatter_reduce_(0, linear_idx, depth, reduce="amax")
    min_pixel_depth.scatter_reduce_(0, linear_idx, depth, reduce="amin")

    max_pixel_depth = max_pixel_depth.view(h, w)
    min_pixel_depth = min_pixel_depth.view(h, w)

    kernel_size = 21
    padding = 10

    local_max = torch.nn.functional.max_pool2d(
        max_pixel_depth.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ).squeeze(0).squeeze(0)

    local_min = -torch.nn.functional.max_pool2d(
        -min_pixel_depth.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    ).squeeze(0).squeeze(0)

    x = points_2d[:, 0].long()
    y = points_2d[:, 1].long()

    point_local_max = local_max[y, x]
    point_local_min = local_min[y, x]

    front_mask = (point_local_max - depth) >= (depth - point_local_min)

    return front_mask


def project_pointcloud_to_2d(pointcloud, camtoworld, K):
    ones = torch.ones((pointcloud.shape[0], 1), device=pointcloud.device)
    pointcloud_h = torch.cat([pointcloud, ones], dim=-1)  # (N, 4)

    # Transform pointcloud from world to camera coordinates
    worldtocam = torch.linalg.inv(camtoworld)
    pointcloud_cam = (worldtocam @ pointcloud_h.T).T  # (N, 4)

    # Keep only the 3D points (drop the homogeneous coordinate)
    pointcloud_cam = pointcloud_cam[:, :3]

    # Project the 3D points to 2D
    pointcloud_proj = (K @ pointcloud_cam.T).T  # (N, 3)

    # Normalize by the depth to get 2D image coordinates
    pointcloud_2d = pointcloud_proj[:, :2] / pointcloud_proj[:, 2:3]
    depth = pointcloud_cam[:, 2]

    pointcloud_2d = pointcloud_2d.squeeze(-1)
    depth = depth.squeeze(-1)

    return pointcloud_2d, depth


def deform_point_cloud_arap_2d(
    points,  # (N, 3) 원본 point cloud
    drag_from,
    drag_to,  # 드래그 목표 위치 (drag_indices에 해당하는 점들의 목표)
    camtoworld,
    K,
    num_iterations=500,
    coef_arap=3e4,
    coef_drag=1,
    lr=1e-2,
):
    """
    ARAP + drag 손실을 동시에 최소화하여 x_origin을 deform하는 예시 코드.
    R: (N, 3, 3), t: (N, 3)를 학습 변수로 둔다.
    """
    device = points.device
    anchor = voxelize_pointcloud_and_get_means(points, 0.05)

    with torch.no_grad():
        means2d, depth = project_pointcloud_to_2d(points, camtoworld, K)
        target_indices = get_target_indices_drag(drag_from, means2d, depth)
        points_target = points[target_indices].clone()

    # -----------------------------
    # 1. 학습 파라미터 정의
    # -----------------------------
    # R은 처음에 항등행렬로 초기화(각 포인트별)
    # t는 0 벡터로 초기화
    N = anchor.shape[0]
    q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)
    q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
    t_init = torch.zeros((N, 3), dtype=torch.float32, device=device)

    q = nn.Parameter(q_init)  # (N, 3, 3)
    t = nn.Parameter(t_init)  # (N, 3)

    # 최적화 알고리즘(Adam)
    optimizer = optim.Adam([
        {'params': q, 'lr': lr},        # Learning rate for R
        {'params': t, 'lr': lr}  # Learning rate for t (e.g., 10 times smaller)
    ])

    # -----------------------------
    # 2. 최적화 루프
    # -----------------------------
    with torch.no_grad():
        distances, indices_knn = knn(anchor, anchor, k=5)
        weight = rbf_weight(distances, gamma=30)

    for step in range(num_iterations):
        # 2-1. 현재 변형된 x_current 계산
        #    x_current[i] = R[i] x_origin[i] + t[i]
        #    (N, 3, 3) x (N, 3) -> broadcasting에 주의
        R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
        anchor_translated = anchor + t  # (N, 3)

        loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)

        points_deformed, _ = linear_blend_skinning_knn(points_target, anchor, R, t)
        means2d_deformed, _ = project_pointcloud_to_2d(points_deformed, camtoworld, K)
        loss_drag = drag_loss(means2d_deformed, drag_to)

        loss = coef_arap * loss_arap + coef_drag * loss_drag

        # 2-3. Backprop 및 optimizer.step
        optimizer.zero_grad()
        # TODO: 왜 retain_graph=True가 필요한지 이해하기
        loss.backward(retain_graph=True)
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{num_iterations}] - Arap loss: {loss_arap.item():.6f}   Drag loss: {loss_drag.item():.6f}")

    with torch.no_grad():
        points_deformed, blended_quats = linear_blend_skinning_knn(points, anchor, R, t)

    return points_deformed, blended_quats, target_indices


@cache_output(func_name="knn_djastra", verbose=False)
@torch.no_grad()
def knn_djastra(
    points: torch.Tensor,
    anchor: torch.Tensor,
    k: int,
    src_type: str = "points",
    dst_type: str = "anchor",
    max_iters: int = None,
    batch_size: int = 1024,
):
    device = points.device
    N = points.shape[0]
    n = anchor.shape[0]
    M = N + n  # 전체 노드 수

    # 1. points와 anchor를 concat (points가 앞쪽, anchor가 뒤쪽)
    points_all = torch.cat([points, anchor], dim=0)  # (M, 3)

    # 2. 제공된 코드로 knn graph 구성 (자기 자신 제거)
    knn_dist, knn_idx = knn(points_all, points_all, k=k + 1, is_sklearn=True)
    knn_dist = knn_dist[:, 1:]
    knn_idx = knn_idx[:, 1:]

    # 2.1. Edge list 구성 및 symmetrize
    row_idx = torch.arange(M, device=device).unsqueeze(1).repeat(1, k)  # (M, k)
    src_edges = row_idx.reshape(-1)  # (M*k,)
    tgt_edges = knn_idx.reshape(-1)  # (M*k,)
    weights = knn_dist.reshape(-1)  # (M*k,)

    # 양방향 edge 추가
    src_all = torch.cat([src_edges, tgt_edges], dim=0)  # (2*M*k,)
    tgt_all = torch.cat([tgt_edges, src_edges], dim=0)  # (2*M*k,)
    weight_all = torch.cat([weights, weights], dim=0)  # (2*M*k,)
    num_edges = src_all.shape[0]

    # 3. source와 destination의 global index 결정 (points: 0~N-1, anchor: N~M-1)
    if src_type == "points":
        source_indices = torch.arange(0, N, device=device)
    elif src_type == "anchor":
        source_indices = torch.arange(N, M, device=device)
    else:
        raise ValueError("src_type must be either 'points' or 'anchor'")

    if dst_type == "points":
        dest_indices = torch.arange(0, N, device=device)
    elif dst_type == "anchor":
        dest_indices = torch.arange(N, M, device=device)
    else:
        raise ValueError("dst_type must be either 'points' or 'anchor'")

    num_sources = source_indices.shape[0]
    # 배치 사이즈가 주어지지 않으면 전체 source를 한 번에 처리
    if batch_size is None:
        batch_size = num_sources

    # 4. 배치별로 multi-source 최단 경로 계산 (벡터화된 Bellman-Ford)
    # 각 배치마다 D: (batch_size_current, M)
    D_batches = []
    for i in range(0, num_sources, batch_size):
        batch_src = source_indices[i : i + batch_size]
        current_batch_size = batch_src.shape[0]
        # 배치에 대한 D 초기화: 각 행 i는 batch_src[i]에서 모든 노드까지의 거리
        D_batch = torch.full((current_batch_size, M), float("inf"), device=device)
        D_batch[torch.arange(current_batch_size, device=device), batch_src] = 0.0

        # 배치에 맞게 tgt_expand 생성 (shape: (current_batch_size, num_edges))
        local_tgt_expand = tgt_all.unsqueeze(0).expand(current_batch_size, num_edges)
        # max_iters 기본값: M (노드 개수)
        _max_iters = max_iters if max_iters is not None else M

        for _ in range(_max_iters):
            # 각 edge의 source 노드에서의 거리를 모으고, weight를 더함.
            D_src = D_batch[:, src_all]  # (current_batch_size, num_edges)
            candidate = D_src + weight_all.unsqueeze(
                0
            )  # (current_batch_size, num_edges)
            new_D_batch = D_batch.clone()
            new_D_batch.scatter_reduce_(
                dim=1,
                index=local_tgt_expand,
                src=candidate,
                reduce="amin",
                include_self=True,
            )
            # 수렴 검사
            if torch.allclose(new_D_batch, D_batch, atol=1e-6):
                D_batch = new_D_batch
                break
            D_batch = new_D_batch

        D_batches.append(D_batch.cpu())
    # 모든 배치를 concat하여 전체 D: (num_sources, M)
    D_full = torch.cat(D_batches, dim=0)

    # 5. 각 source에 대해 destination set 내에서 k개의 최단 경로 선택
    D_dest = D_full[:, dest_indices.cpu()]  # (num_sources, num_destinations)
    D_dest = D_dest.to(device)
    knn_distances, knn_indices_in_dest = torch.topk(D_dest, k=k, largest=False, dim=1)
    # global index로 복원
    global_dest_idx = dest_indices[knn_indices_in_dest]
    # destination이 anchor라면 원래 anchor 내 인덱스로 변환 (anchor는 points_all에서 뒤쪽 부분)
    if dst_type == "anchor":
        final_indices = global_dest_idx - N
    else:
        final_indices = global_dest_idx

    return knn_distances, final_indices


def count_covered_patches(
    pixels: torch.Tensor,   # (N, 2)  [x, y]
    patch_size: int,
) -> int:
    if pixels.numel() == 0:
        return 0

    x = pixels[:, 0].floor().long()
    y = pixels[:, 1].floor().long()

    row_idx = y // patch_size  # (N,)
    col_idx = x // patch_size  # (N,)

    patch_coords = torch.stack([row_idx, col_idx], dim=1)  # (N, 2)
    covered_count = torch.unique(patch_coords, dim=0).size(0)

    return int(covered_count)