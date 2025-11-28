import torch
import torch.nn.functional as F
from util.drot_loss import PointLossFunction
from jhutil import (
    get_img_diff,
    crop_two_image_with_background,
    crop_two_image_with_alpha,
)


def arap_loss_grouped(points, points_translated, R, group_ids, max_group_size=3000):
    device = points.device
    loss = 0.
    n_group = int(group_ids.max().item())

    for i in range(0, n_group + 1):    
        target_idx = group_ids == i

        N = target_idx.sum()
        if N < 2:
            continue
        if N > max_group_size:
            target_idx = prune_bool_idx(target_idx, max_group_size)
            N = target_idx.sum()
        
        weight = torch.ones(N, N, device=device) / N
        indices = torch.arange(0, N, device=device).repeat(N, 1)
        loss += arap_loss(points[target_idx], points_translated[target_idx], R[target_idx], weight, indices)

    return loss


def prune_bool_idx(idx_bool, N):
    """
    Prune idx to a maximum of N elements.
    """
    device = idx_bool.device
    
    idx_int = torch.arange(0, len(idx_bool), device=device)[idx_bool]
    prune_idx = torch.randperm(idx_int.size(0))[:N]
    idx_int_sparse = idx_int[prune_idx]
    idx_bool_sparse = torch.zeros_like(idx_bool, dtype=torch.bool, device=device)
    idx_bool_sparse[idx_int_sparse] = True

    return idx_bool_sparse


def arap_loss(points, points_translated, R, weight=None, indices=None, return_list=False, loss_type="L2"):
    """
    x_origin: (N, 3)
    x_current: (N, 3)
    R: (N, 3, 3)
    weight: (N, k)
    indices: (N, k)
    """

    if weight is None and indices is None:
        device = points.device
        N = points.shape[0]
        weight = torch.ones(N, N, device=device) / N
        indices = torch.arange(0, N, device=device).repeat(N, 1)
    
    # (N, 1, 3) - (N, k, 3) => (N, k, 3)
    e_origin = points[:, None, :] - points[indices]
    e_translated = points_translated[:, None, :] - points_translated[indices]

    batched_R = R[:, None, :, :]  # (N, 1, 3, 3)
    e_origin_reshaped = e_origin.unsqueeze(-1)  # (N, k, 3, 1)

    # (N, k, 1, 3) x (N, 1, 3, 3) => (N, k, 1, 3)
    rotated = torch.matmul(batched_R, e_origin_reshaped)  # (N, k, 3, 1)
    rotated = rotated.squeeze(-1)  # (N, k, 3)

    diff = rotated - e_translated
    if loss_type == "L2":
        loss = weight * diff.pow(2).sum(dim=2)
    elif loss_type == "L1":
        loss = weight * diff.abs().sum(dim=2)
    
    if return_list:
        return loss.sum(dim=1)
    else:
        return loss.sum()


def arap_loss_rot(q, weight, indices):

    q_normlaized = F.normalize(q, dim=-1).squeeze()
    # Eq.13 in "3D Gaussian Editing with A Single Image"
    diff = q_normlaized[:, None, :] - q_normlaized[indices]  # (N, k, 4)
    loss = weight * diff.pow(2).sum(dim=2)  # (N, k)

    return loss.sum(dim=1).mean()


def arap_loss_dist(points, points_translated, weight, indices):
    """
    x_origin: (N, 3)
    x_current: (N, 3)
    R: (N, 3, 3)
    weight: (N, k)
    indices: (N, k)
    """

    if weight is None and indices is None:
        device = points.device
        N = points.shape[0]
        weight = torch.ones(N, N, device=device) / N
        indices = torch.arange(0, N, device=device).repeat(N, 1)
    
    # (N, 1, 3) - (N, k, 3) => (N, k, 3)
    diff_origin = (points[:, None, :] - points[indices])
    diff_translated = points_translated[:, None, :] - points_translated[indices]

    loss = weight * (diff_origin.norm(dim=-1) - diff_translated.norm(dim=-1)).abs()

    return loss.sum(dim=1).mean()



def drag_loss(x_target, x_goal):
    """
    x_current: (n, 3)
    x_goal: (n, 3)
    """
    loss = torch.mean(((x_target - x_goal) ** 2))
    return loss


def downsample_by_mean(img, downsample):
    """
    img: torch.Tensor of shape (H, W, 3)
    returns: downsampled image of shape (H//2, W//2, 3)
    """
    # (H, W, 3) → (1, 3, H, W)
    img = img.permute(0, 3, 1, 2)

    # 평균 풀링을 이용한 다운샘플링
    img_down = F.avg_pool2d(img, kernel_size=downsample, stride=downsample)
    img_down = img_down.permute(0, 2, 3, 1)

    return img_down


def drot_loss(pred_image, gt_image, uv, return_matching=False, downsample=1):
    """_summary_

    Args:
        pred_image (_type_):
        gt_image (_type_):
        means2d (_type_): (N, 2) tensor

    Returns:
        _type_:
    """
    assert pred_image.shape == gt_image.shape
    
    if pred_image.dim() == 3:
        pred_image = pred_image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)

    if pred_image.shape[1] == 4:
        pred_image, mask = pred_image[:, :3], pred_image[:, 3]
        gt_image = gt_image[:, :3]

    if pred_image.shape[1] == 3:
        pred_image = pred_image.permute(0, 2, 3, 1)
        gt_image = gt_image.permute(0, 2, 3, 1)

    union_bbox, pred_image_cropped, gt_image_cropped = crop_two_image_with_background(
        pred_image[0], gt_image[0], shape="hwc",
    )
    pred_image_cropped = pred_image_cropped.unsqueeze(0)
    gt_image_cropped = gt_image_cropped.unsqueeze(0)
    
    left, top, right, bottom = union_bbox
    uv_cropped = uv[top:bottom, left:right]

    if downsample > 1:
        pred_image_cropped = downsample_by_mean(pred_image_cropped, downsample)
        gt_image_cropped = downsample_by_mean(gt_image_cropped, downsample)
        uv_cropped = downsample_by_mean(uv_cropped.unsqueeze(0), downsample)[0]

    h, w = pred_image_cropped.shape[1:3]
    loss_func = PointLossFunction(resolution=(h, w))

    uv_cropped = uv_cropped - uv_cropped[0:1, 0:1]
    uv_cropped = uv_cropped / uv_cropped.max()

    uv_cropped = (2 * uv_cropped) - 1
    uv_cropped = uv_cropped.unsqueeze(0)

    mask = torch.ones_like(uv_cropped[..., 0]).bool()

    render_result = {
        "msk": mask.cuda(),
        "pos": uv_cropped.cuda(),
        "images": pred_image_cropped.cuda(),
    }
    loss = loss_func.get_loss(render_result, gt_image_cropped, 0, return_matching)
    return loss * downsample**2



def drot_loss_with_means2d(pred_image, gt_image, means2d, return_matching=False):
    """_summary_

    Args:
        pred_image (_type_):
        gt_image (_type_):
        means2d (_type_): (N, 2) tensor

    Returns:
        _type_:
    """
    device = pred_image.device
    h, w = pred_image.shape[1:3]

    x = torch.arange(0, w, device=device, dtype=torch.float32)
    y = torch.arange(0, h, device=device, dtype=torch.float32)

    y_grid, x_grid = torch.meshgrid(x, y, indexing="xy")  # (2, r, r)
    uv = torch.stack([y_grid, x_grid], dim=-1)

    coords = torch.round(means2d).long()
    coords[:, 1] = torch.clamp(coords[:, 1], 0, h - 1)
    coords[:, 0] = torch.clamp(coords[:, 0], 0, w - 1)
    uv[coords[:, 1], coords[:, 0]] = means2d

    return drot_loss(pred_image, gt_image, uv, return_matching)
