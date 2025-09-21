import torch
import numpy as np
import random
import cv2
from jhutil.algorithm import knn, ball_query
from jhutil import cache_output
from .loss import arap_loss


def get_inlier_with_pnp_ransac(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    camera_matrix: torch.Tensor = None,
    dist_coeffs: torch.Tensor = None,
    reprojection_error: float = 3.0,
    confidence: float = 0.99,
    iterations_count: int = 100,
):
    if camera_matrix is None:
        camera_matrix = torch.eye(3, dtype=torch.float32)
    if dist_coeffs is None:
        dist_coeffs = torch.zeros((5,), dtype=torch.float32)

    p3d = points_3d.detach().cpu().numpy().astype(np.float32)
    p2d = points_2d.detach().cpu().numpy().astype(np.float32)
    cam_mat = camera_matrix.detach().cpu().numpy().astype(np.float32)
    dist_coe = dist_coeffs.detach().cpu().numpy().astype(np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=p3d,
        imagePoints=p2d,
        cameraMatrix=cam_mat,
        distCoeffs=dist_coe,
        reprojectionError=reprojection_error,
        confidence=confidence,
        iterationsCount=iterations_count,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None:
        inlier_mask = torch.ones((p3d.shape[0],), dtype=torch.bool)
        inlier_errors = torch.zeros((p3d.shape[0],), dtype=torch.float32)
        return inlier_mask, inlier_errors

    inliers_idx = inliers.flatten()
    num_points = p3d.shape[0]
    inlier_mask_np = np.zeros(num_points, dtype=bool)
    inlier_mask_np[inliers_idx] = True

    projected, _ = cv2.projectPoints(p3d, rvec, tvec, cam_mat, dist_coe)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(p2d - projected, axis=1)

    inlier_errors_np = np.zeros(num_points, dtype=np.float32)
    inlier_errors_np[inlier_mask_np] = errors[inlier_mask_np]

    inlier_mask = torch.from_numpy(inlier_mask_np)
    inlier_errors = torch.from_numpy(inlier_errors_np)

    return inlier_mask, inlier_errors


def naive_rigid_grouping(points_3d, drag_target, reprojection_error, camera_matrix):
    groups = []
    remaining_idx = torch.arange(points_3d.shape[0])
    for i in range(5):
        if points_3d.shape[0] < 6:
            break

        inlier_mask, inlier_error = get_inlier_with_pnp_ransac(
            points_3d,
            drag_target,
            camera_matrix,
            dist_coeffs=None,
            reprojection_error=reprojection_error,
            confidence=0.99,
        )
        if inlier_mask.sum() < 6:
            continue

        groups.append(remaining_idx[inlier_mask])
        remaining_idx = remaining_idx[torch.logical_not(inlier_mask)]
        points_3d = points_3d[torch.logical_not(inlier_mask)]
        drag_target = drag_target[torch.logical_not(inlier_mask)]

    return groups


@cache_output(func_name="local_rigid_grouping")
def local_rigid_grouping(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    k: int = 100,
    min_inlier_ratio: float = 0.5,
    min_group_size: int = 20,
    max_expansion_iterations: int = 10,
    reprojection_error: float = 3.0,
    confidence: float = 0.99,
    iterations_count: int = 100,
    camera_matrix: torch.Tensor = None,
    dist_coeffs: torch.Tensor = None,
):
    """
    DBSCAN 스타일의 locality grouping.

    알고리즘 순서:
      1. ungrouped set에서 임의의 seed point 선택
      2. seed point의 kNN로 초기 그룹 G 구성 후 PnP-ransac 적용
         - inlier ratio가 min_inlier_ratio보다 낮으면 해당 seed는 discard
      3. 그룹 G 내의 각 inlier point에 대해 knn search로 새로운 후보 포인트를 찾고 G에 추가
      4. 추가된 G에 대해 다시 PnP-ransac 적용 (transformation 재추정)
      5. 그룹 크기가 더 이상 증가하지 않을 때까지 3-4 반복
      6. G의 크기가 min_group_size 이상이면 그룹으로 확정하고, 그렇지 않으면 outlier 처리
      7. 남은 ungrouped 포인트에 대해 반복

    취약점 및 보완점:
      - 그룹 확장 시 잘못된 outlier가 포함될 위험이 있으므로, 매 반복마다 inlier ratio와 PnP 결과의 안정성을 확인함.
      - transformation 간 큰 차이가 발생하면 그룹이 섞일 위험이 있으나, 여기서는 간단히 inlier ratio로 판단 (추후 rvec, tvec의 변화 체크 등 추가 고려 가능)
    """
    points_3d = points_3d.detach().cpu()
    distances, indices = knn(points_3d, points_3d, k)
    distances = distances.cpu().numpy()
    indices = indices.cpu().numpy()

    N = points_3d.shape[0]
    ungrouped = set(range(N))
    groups = []
    group_transformations = {}

    while ungrouped:
        seed = random.choice(list(ungrouped))
        candidate_idxs = set(indices[seed][:k])
        candidate_idxs = candidate_idxs.intersection(ungrouped)

        if len(candidate_idxs) < 6:
            ungrouped.discard(seed)
            continue

        current_group = set(candidate_idxs)

        group_list = list(current_group)
        group_points3d = points_3d[group_list]
        group_points2d = points_2d[group_list]
        inlier_mask, inlier_errors = get_inlier_with_pnp_ransac(
            group_points3d,
            group_points2d,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            reprojection_error=reprojection_error,
            confidence=confidence,
            iterations_count=iterations_count,
        )
        inlier_idxs = torch.tensor(group_list)[inlier_mask]

        if len(inlier_idxs) / len(current_group) < min_inlier_ratio:
            ungrouped.discard(seed)
            continue

        expansion_iter = 0
        group_changed = True
        while group_changed and expansion_iter < max_expansion_iterations:
            group_changed = False
            new_candidates = set()
            for idx in inlier_idxs:
                neighbors = set(indices[idx][:k])
                new_candidates.update(neighbors.intersection(ungrouped))

            if new_candidates - current_group:
                candidate_group = current_group.union(new_candidates)
                candidate_list = list(candidate_group)
                cand_points3d = points_3d[candidate_list]
                cand_points2d = points_2d[candidate_list]
                new_inlier_mask, new_inlier_errors = get_inlier_with_pnp_ransac(
                    cand_points3d,
                    cand_points2d,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    reprojection_error=reprojection_error,
                    confidence=confidence,
                    iterations_count=iterations_count,
                )
                new_inlier_idxs = torch.tensor(candidate_list)[new_inlier_mask]
                if len(candidate_group) > len(current_group):
                    current_group = candidate_group
                    inlier_idxs = new_inlier_idxs
                    group_changed = True
            expansion_iter += 1

        if len(new_inlier_idxs) >= min_group_size:
            groups.append(new_inlier_idxs)
            ungrouped = ungrouped - current_group
            group_transformations[len(groups) - 1] = None
        else:
            ungrouped = ungrouped - current_group

    outliers = set(range(N))
    for group in groups:
        outliers = outliers - set(group.tolist())
    outliers = torch.tensor(list(outliers))

    return groups, outliers, group_transformations


@torch.no_grad()
def refine_rigid_group(
    points_3d, points_lbs, group_id_all, R, radius=0.05, outlier_threhold=0.03
):
    device = points_3d.device

    indices = torch.randperm(group_id_all.max() + 1)
    loss_all = outlier_threhold * torch.ones_like(group_id_all, dtype=torch.float32)

    for i in indices:

        group_mask = group_id_all == i
        group_indice = group_mask.nonzero(as_tuple=True)[0]
        group = points_lbs[group_indice]
        if len(group) == 0:
            continue

        # 1. Enlarge group with nearest neighbors
        enlarged_indice = ball_query(group, points_lbs, radius, concat=True)
        enlarged_mask = torch.zeros_like(group_mask, dtype=torch.bool)
        enlarged_mask[enlarged_indice] = True

        sub_mask = group_mask[enlarged_indice]
        sub_indice = sub_mask.nonzero(as_tuple=True)[0]

        # 2. Clean with ARAP loss
        n = len(sub_indice)
        n_sample = 300
        if n > n_sample:
            indices = sub_indice[torch.randperm(n)[:n_sample]]
            weight = torch.ones(n_sample, device=device) / n_sample
        else:
            indices = sub_indice
            weight = torch.ones(n, device=device) / n

        with torch.no_grad():
            loss = arap_loss(
                points_3d[enlarged_indice],
                points_lbs[enlarged_indice],
                R[enlarged_indice],
                weight=weight,
                indices=indices,
                return_list=True,
                loss_type="L1",
            )

        loss_prev = loss_all[enlarged_indice]
        update_mask = loss_prev > loss
        loss_prev = torch.min(loss_prev, loss)
        loss_all[enlarged_indice] = loss_prev

        # updare group
        group_id_all[enlarged_indice] = torch.where(
            update_mask, i, group_id_all[enlarged_indice]
        )
        # remove outlier
        group_id_all[group_mask] = torch.where(loss[sub_mask] > outlier_threhold, -1, i)

    return group_id_all
