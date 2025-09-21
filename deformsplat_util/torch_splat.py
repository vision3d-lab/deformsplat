# Most code is from https://github.com/hbb1/torch-splatting
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result



def build_color(means3D, shs, cam_origin, sh_degree=0):
    rays_o = cam_origin
    rays_d = means3D - rays_o

    color = eval_sh(sh_degree, shs.permute(0, 2, 1), rays_d)
    color = (color + 0.5).clip(min=0.0)
    return color


def rgb_to_sh(rgb):
    assert rgb.shape[-1] == 3
    sh0 = torch.zeros((rgb.shape[0], 1, 3), device=rgb.device, dtype=rgb.dtype)
    sh0[:, 0] = (rgb - 0.5) / C0

    return sh0

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


def render(width, height, means2D, cov2d, color, opacity, depths, white_bkgd=False):
    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=width, height=height)
    
    pix_coord = torch.stack(
        torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'),
        dim=-1
    ).to('cuda')  # shape: (height, width, 2)
    render_color = torch.zeros(*pix_coord.shape[:2], 3, device='cuda')
    render_depth = torch.zeros(*pix_coord.shape[:2], 1, device='cuda')
    render_alpha = torch.zeros(*pix_coord.shape[:2], 1, device='cuda')

    TILE_SIZE = 64
    for h in range(0, height, TILE_SIZE):
        h_end = min(h + TILE_SIZE, height)
        for w in range(0, width, TILE_SIZE):
            w_end = min(w + TILE_SIZE, width)

            over_tl = (
                rect[0][..., 0].clip(min=w),
                rect[0][..., 1].clip(min=h)
            )
            over_br = (
                rect[1][..., 0].clip(max=w_end - 1),
                rect[1][..., 1].clip(max=h_end - 1)
            )
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
            if not in_mask.any():
                continue

            P = in_mask.sum()
            tile_coord = pix_coord[h:h_end, w:w_end].flatten(0, -2)  # (B, 2), B = (h_end-h)*(w_end-w)

            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask][index]
            sorted_cov2d = cov2d[in_mask][index]
            sorted_conic = torch.inverse(sorted_cov2d)  # (P,2,2)
            sorted_opacity = opacity[in_mask][index]
            sorted_color = color[in_mask][index]

            dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # (B, P, 2)
            gauss_weight = torch.exp(-0.5 * (
                dx[..., 0]**2 * sorted_conic[:, 0, 0] +
                dx[..., 1]**2 * sorted_conic[:, 1, 1] +
                dx[..., 0]*dx[..., 1] * (sorted_conic[:, 0, 1] + sorted_conic[:, 1, 0])
            ))

            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clamp(max=0.99)  # (B, P, 1)
            T = torch.cat(
                [torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]],
                dim=1
            ).cumprod(dim=1)  # (B, P, 1)
            acc_alpha = (alpha * T).sum(dim=1)  # (B, 1)
            tile_color = (
                (T * alpha * sorted_color[None]).sum(dim=1)
                + (1 - acc_alpha) * (1 if white_bkgd else 0)
            )  # (B, 3)
            tile_depth = (
                (T * alpha * sorted_depths[None, :, None]).sum(dim=1)
            )  # (B, 1)

            h_size = h_end - h
            w_size = w_end - w
            render_color[h:h_end, w:w_end] = tile_color.view(h_size, w_size, -1)
            render_depth[h:h_end, w:w_end] = tile_depth.view(h_size, w_size, -1)
            render_alpha[h:h_end, w:w_end] = acc_alpha.view(h_size, w_size, -1)

    return {
        "render": render_color,
        "depth": render_depth,
        "alpha": render_alpha,
        "visiility_filter": radii > 0,
        "radii": radii
    }



def render_uv_coordinate(width, height, means2D, cov2d, opacity, depths, white_bkgd=False):
    radii = get_radius(cov2d)
    rect = get_rect(means2D, radii, width=width, height=height)
    
    pix_coord = torch.stack(torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy'), dim=-1).to('cuda')

    uv = pix_coord.clone().float()

    TILE_SIZE = 64
    for h in range(0, height, TILE_SIZE):
        h_end = min(h + TILE_SIZE, height)
        for w in range(0, width, TILE_SIZE):
            w_end = min(w + TILE_SIZE, width)

            # check if the rectangle penetrate the tile
            over_tl = (
                rect[0][..., 0].clip(min=w),
                rect[0][..., 1].clip(min=h)
            )
            over_br = (
                rect[1][..., 0].clip(max=w_end - 1),
                rect[1][..., 1].clip(max=h_end - 1)
            )
            in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
            if not in_mask.any():
                continue

            tile_coord = pix_coord[h:h_end, w:w_end].flatten(0, -2)  # (B, 2)
            
            sorted_depths, index = torch.sort(depths[in_mask])
            sorted_means2D = means2D[in_mask][index]
            sorted_cov2d = cov2d[in_mask][index]
            sorted_conic = torch.inverse(sorted_cov2d)
            sorted_opacity = opacity[in_mask][index]

            dx = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # (B, P, 2)
            gauss_weight = torch.exp(-0.5 * (
                dx[..., 0]**2 * sorted_conic[:, 0, 0] +
                dx[..., 1]**2 * sorted_conic[:, 1, 1] +
                dx[..., 0]*dx[..., 1] * (sorted_conic[:, 0, 1] + sorted_conic[:, 1, 0])
            ))

            alpha = (gauss_weight[..., None] * sorted_opacity[None]).clamp(max=0.99)  # (B, P, 1)
            T = torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], dim=1).cumprod(dim=1)
            acc_alpha = (alpha * T).sum(dim=1)  # (B, 1)

            sorted_std = torch.stack([sorted_cov2d[:, 0, 0], sorted_cov2d[:, 1, 1]], dim=-1).sqrt()

            # Eq.8 in "3D Gaussian Editing with A Single Image"
            with torch.no_grad():
                epsilon = dx / sorted_std[None, :, :]
            intersection_point = sorted_means2D[None, :, :] + sorted_std[None, :, :] * epsilon

            # Eq.9 in "3D Gaussian Editing with A Single Image"
            T_det, alpha_det, acc_alpha_det = T.detach(), alpha.detach(), acc_alpha.detach()
            uv_recovered = (T_det * alpha_det * intersection_point).sum(dim=1) + \
                           (1 - acc_alpha_det) * uv[h:h_end, w:w_end].reshape(-1, 2)

            h_size, w_size = h_end - h, w_end - w
            uv[h:h_end, w:w_end] = uv_recovered.view(h_size, w_size, 2)

    return uv
