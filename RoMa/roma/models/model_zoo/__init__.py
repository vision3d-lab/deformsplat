from typing import Union
import torch
from .roma_models import roma_model
from PIL import Image


def get_roma(device, 
             roma_path="/kaggle/input/roma/pytorch/1/1/roma_outdoor.pth", 
             dino_path="/kaggle/input/dinov2-vit-pretrain/dinov2_vitl14_pretrain.pth"):
    weights = torch.load(roma_path, map_location=device, weights_only=False)
    dinov2_weights = torch.load(dino_path, map_location=device, weights_only=False)
    roma_model = roma_outdoor(device=device, weights=weights, dinov2_weights=dinov2_weights,
                              coarse_res=560, upsample_res=(864, 1152))
    H, W = roma_model.get_output_resolution()
    return roma_model, H, W


def sample(
    matches,
    certainty,
    sample_thresh=0.05,
):

    matches, certainty = (
        matches.reshape(-1, 4),
        certainty.reshape(-1),
    )

    good_samples = certainty > sample_thresh
    return matches[good_samples], certainty[good_samples]


def get_matches_roma(roma_model, im1_path, im2_path, device):

    warp, certainty_all = roma_model.match(im1_path, im2_path, device=device)
    matches, certainty = sample(warp, certainty_all)
    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    return matches, certainty, kpts1, kpts2


def get_alike_from_roma_coord(alike_coord1, alike_coord2, roma_coord1, roma_coord2, threshold=1):
    from jhutil.algorithm import knn
    min_dist1, indices1 = knn(roma_coord1, alike_coord1)
    # distance1 = torch.norm(alike_coord1[None, :] - roma_coord1[:, None, :], dim=-1)
    # min_dist1, indices1 = distance1.min(dim=-1)
    is_close1 = min_dist1 < threshold

    min_dist2, indices2 = knn(roma_coord2, alike_coord2)
    # distance2 = torch.norm(alike_coord2[None, :] - roma_coord2[:, None, :], dim=-1)
    # min_dist2, indices2 = distance2.min(dim=-1)
    is_close2 = min_dist2 < threshold

    is_close = is_close1 & is_close2
    to_alike_indices1 = indices1[is_close]
    to_alike_indices2 = indices2[is_close]
    indices = torch.stack((to_alike_indices1, to_alike_indices2), dim=-1)
    return torch.unique(indices, dim=0)


weight_urls = {
    "roma": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # hopefully this doesnt change :D
}


def roma_outdoor(device, weights=None, dinov2_weights=None, coarse_res: Union[int, tuple[int, int]] = 560, upsample_res: Union[int, tuple[int, int]] = 864, amp_dtype: torch.dtype = torch.float16):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["outdoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                            map_location=device)
    model = roma_model(resolution=coarse_res, upsample_preds=True,
                       weights=weights, dinov2_weights=dinov2_weights, device=device, amp_dtype=amp_dtype)
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model


def roma_indoor(device, weights=None, dinov2_weights=None, coarse_res: Union[int, tuple[int, int]] = 560, upsample_res: Union[int, tuple[int, int]] = 864, amp_dtype: torch.dtype = torch.float16):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["indoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                            map_location=device)
    model = roma_model(resolution=coarse_res, upsample_preds=True,
                       weights=weights, dinov2_weights=dinov2_weights, device=device, amp_dtype=amp_dtype)
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model
