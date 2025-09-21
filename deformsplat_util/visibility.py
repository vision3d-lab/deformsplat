from gsplat.cuda._torch_impl import *
from gsplat.cuda._torch_impl import (
    _world_to_cam,
    _ortho_proj,
    _fisheye_proj,
    _persp_proj,
    _quat_to_rotmat,
)
from jhutil import cache_output


def _fully_fused_projection2(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(v1))  # (...,)
    # v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    # radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0

    return means2d, depths, covars2d


def _compute_visibility(means2d, cov2d, opacities, depths):
    N = means2d.shape[0]
    # opacities가 (N, 1)인 경우 squeeze
    if opacities.dim() > 1:
        opacities = opacities.squeeze(-1)

    # depth 기준으로 정렬 (앞쪽부터)
    sorted_idx = torch.argsort(depths)
    sorted_means = means2d[sorted_idx]  # (N, 2)
    sorted_cov2d = cov2d[sorted_idx]  # (N, 2, 2)
    sorted_opacities = opacities[sorted_idx]  # (N,)

    sorted_cov2d_inv = torch.inverse(sorted_cov2d)  # (N, 2, 2)

    sorted_visibility = torch.empty_like(sorted_opacities)

    for k in range(N):
        if k == 0:
            prod_term = 1.0
        else:
            dx = sorted_means[k] - sorted_means[:k]  # (k, 2)
            exponents = -0.5 * (
                dx[:, 0] ** 2 * sorted_cov2d_inv[:k, 0, 0]
                + dx[:, 1] ** 2 * sorted_cov2d_inv[:k, 1, 1]
                + 2.0 * dx[:, 0] * dx[:, 1] * sorted_cov2d_inv[:k, 0, 1]
            )
            G = torch.exp(exponents)  # (k,)
            alphas = sorted_opacities[:k].clamp(max=0.99)
            prod_term = torch.prod(1 - alphas * G)

        sorted_visibility[k] = prod_term

    visibility = torch.empty_like(sorted_visibility)
    visibility[sorted_idx] = sorted_visibility
    return visibility


@torch.no_grad()
@cache_output(func_name="compute_visibility")
def compute_visibility(
    width, height, means3d, quats, opacities, scales, Ks, camtoworlds
):

    viewmats = torch.linalg.inv(camtoworlds)
    opacities = torch.sigmoid(opacities)
    scales = torch.exp(scales)

    R = _quat_to_rotmat(quats)  # (..., 3, 3)

    M = R * scales[..., None, :]  # (..., 3, 3)
    covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)

    means2d, depths, cov2d = _fully_fused_projection2(means3d, covars, viewmats, Ks, width, height)
    means2d, cov2d, opacities, depths = [x.squeeze() for x in [means2d, cov2d, opacities, depths]]
    opacities = opacities[:, None]

    visibility = _compute_visibility(means2d, cov2d, opacities, depths)
    
    visibility[depths < 0.01] = 0.0

    return visibility, means2d

