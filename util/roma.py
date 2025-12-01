import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from roma import roma_outdoor
from jhutil import save_img, crop_two_image_with_alpha
from jhutil import to_cuda
from datetime import datetime
import time

roma_model = None


def roma_match(im1_path_crop, im2_path_crop, device) -> None:
    global roma_model
    if roma_model is None:
        roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))

    warp, certainty = roma_model.match(
        im1_path_crop, im2_path_crop, device=device
    )  # (H, 2 * W, 4), (H, 2 * W)
    return warp, certainty


def get_drag_roma(
    im1,
    im2,
    cycle_threshold=10,
    device="cuda",
):

    if im1.dim() == 4:
        if im1.shape[0] != 1:
            raise NotImplementedError("This function only takes single image")
        im1 = im1.squeeze()
        im2 = im2.squeeze()
    if im1.shape[2] == 4:
        im1 = im1.permute(2, 0, 1)
        im2 = im2.permute(2, 0, 1)

    ##############################################################################
    ################################ 1. Run ROMA #################################
    ##############################################################################
    union_bbox, cropped_img1, cropped_img2 = crop_two_image_with_alpha(im1, im2)

    # mkdir .cache
    os.makedirs(".cache", exist_ok=True)
    im1_path_crop = f".cache/{cropped_img1.sum().item()}_im1.png"
    im2_path_crop = f".cache/{cropped_img2.sum().item()}_im2.png"

    save_img(cropped_img1, im1_path_crop)
    save_img(cropped_img2, im2_path_crop)
    
    warp, certainty = roma_match(im1_path_crop, im2_path_crop, device=device)

    ##############################################################################
    ############################ 2. Get Empty Region #############################
    ##############################################################################
    H_net, W_net = 864, 1152
    W_bbox = union_bbox[2] - union_bbox[0]
    H_bbox = union_bbox[3] - union_bbox[1]

    with Image.open(im1_path_crop) as img:
        im1_crop = img.copy().resize((W_net, H_net))
    with Image.open(im2_path_crop) as img:
        im2_crop = img.copy().resize((W_net, H_net))

    # os.remove(im1_path_crop)
    # os.remove(im2_path_crop)

    im1_crop = (torch.tensor(np.array(im1_crop)) / 255).to(device).permute(2, 0, 1)
    im2_crop = (torch.tensor(np.array(im2_crop)) / 255).to(device).permute(2, 0, 1)

    not_empty_from = im1_crop[3] != 0
    not_empty_to = im2_crop[3] != 0

    warp_first_img = warp[:, :, 2:]

    drag_from = warp_first_img[:, W_net:][not_empty_from]
    drag_to = warp_first_img[:, :W_net][not_empty_from]
    cert_from = certainty[:, W_net:][not_empty_from]

    ##############################################################################
    ####################### 3. Scale Drag to origin scale ########################
    ##############################################################################
    drag_from[:, 0] = (1 + drag_from[:, 0]) / 2 * (W_bbox - 1) + union_bbox[0]
    drag_from[:, 1] = (1 + drag_from[:, 1]) / 2 * (H_bbox - 1) + union_bbox[1]

    drag_to[:, 0] = (1 + drag_to[:, 0]) / 2 * (W_bbox - 1) + union_bbox[0]
    drag_to[:, 1] = (1 + drag_to[:, 1]) / 2 * (H_bbox - 1) + union_bbox[1]

    ##############################################################################
    ############################ 4. Filter duplicate #############################
    ##############################################################################
    drag_from = drag_from.round()
    drag_to = drag_to.round()

    drag_from_np = drag_from.cpu().numpy()
    _, unique_indices = np.unique(drag_from_np, axis=0, return_index=True)
    drag_from = drag_from[unique_indices]
    cert_from = cert_from[unique_indices]
    drag_to = drag_to[unique_indices]

    ##############################################################################
    ############################ 5. Filter by cycle ##############################
    ##############################################################################
    if cycle_threshold > 0:
        warp_second_img = warp[:, :, :2]
        drag_from_second = warp_second_img[:, :W_net][not_empty_to]
        drag_to_second = warp_second_img[:, W_net:][not_empty_to]

        drag_to_second[:, 0] = (1 + drag_to_second[:, 0]) / 2 * (W_bbox - 1) + union_bbox[0]
        drag_to_second[:, 1] = (1 + drag_to_second[:, 1]) / 2 * (H_bbox - 1) + union_bbox[1]

        drag_from_second[:, 0] = (1 + drag_from_second[:, 0]) / 2 * (W_bbox - 1) + union_bbox[0]
        drag_from_second[:, 1] = (1 + drag_from_second[:, 1]) / 2 * (H_bbox - 1) + union_bbox[1]
        
        drag_to_second = drag_to_second.round()
        drag_from_second = drag_from_second.round()

        drag_from_second_np = drag_from_second.cpu().numpy()
        _, unique_indices = np.unique(drag_from_second_np, axis=0, return_index=True)

        drag_to_second = drag_to_second[unique_indices]
        drag_from_second = drag_from_second[unique_indices]

        dist = cycle_distances(drag_from, drag_to, drag_from_second, drag_to_second)
        drag_from = drag_from[dist < cycle_threshold]
        drag_to = drag_to[dist < cycle_threshold]
        cert_from = cert_from[dist < cycle_threshold]
        
    # 6. filter by alpha
    drag_from, drag_to, cert_from = filter_outlier(im1.to(device), im2.to(device), drag_from, drag_to, cert_from)
    
    while len(drag_from) > 10e6:
        drag_from = drag_from[::2]
        drag_to = drag_to[::2]
    
    return drag_from, drag_to, union_bbox


def get_inlier_indice(points, alpha):
    points = points.long()
    inlier_indice = alpha[points[:, 1], points[:, 0]] != 0
    return inlier_indice


def filter_outlier(image_from, image_to, drag_from, drag_to, cert_from):
    image_from, image_to, drag_from, drag_to = to_cuda([image_from, image_to, drag_from, drag_to])
    inlier_indice = get_inlier_indice(drag_from, image_from[3])
    drag_from = drag_from[inlier_indice]
    drag_to = drag_to[inlier_indice]
    cert_from = cert_from[inlier_indice]

    inlier_indice = get_inlier_indice(drag_to, image_to[3])
    drag_from = drag_from[inlier_indice]
    drag_to = drag_to[inlier_indice]
    cert_from = cert_from[inlier_indice]
    
    return drag_from, drag_to, cert_from


def cycle_distances(
    drag_from, drag_to, drag_from_second, drag_to_second, chunk_size=1024
):
    device = drag_from.device

    N = drag_from.shape[0]
    M = drag_from_second.shape[0]

    best_dists = torch.full((N,), float("inf"), device=device)
    best_j = torch.zeros((N,), dtype=torch.long, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = drag_from_second[start:end]

        diff = drag_to.unsqueeze(1) - chunk.unsqueeze(0)
        dist_sq = diff.pow(2).sum(dim=-1)

        chunk_min_dist_sq, chunk_min_idx = dist_sq.min(dim=1)

        mask = chunk_min_dist_sq < best_dists
        best_dists[mask] = chunk_min_dist_sq[mask]
        best_j[mask] = chunk_min_idx[mask] + start

    x_prime = drag_to_second[best_j]
    cycle_dist = (drag_from - x_prime).norm(dim=-1)

    return cycle_dist


def match_two_image_roma(im1, im2, vis_certainty=False, device="cuda"):
    
    global roma_model
    if roma_model is None:
        roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    H_net, W_net = roma_model.get_output_resolution()

    union_bbox, cropped_img1, cropped_img2 = crop_two_image_with_alpha(im1, im2)
    
    random_str = str(datetime.now().timestamp())
    im1_path_crop = f"{random_str}_im1.png"
    im2_path_crop = f"{random_str}_im2.png"

    save_img(cropped_img1, im1_path_crop)
    save_img(cropped_img2, im2_path_crop)

    im1 = Image.open(im1_path_crop)
    im2 = Image.open(im2_path_crop)

    im1 = im1.resize((W_net, H_net))
    im2 = im2.resize((W_net, H_net))

    # Match
    warp, certainty = roma_model.match(
        im1_path_crop, im2_path_crop, device=device
    )  # (H, 2 * W, 4), (H, 2 * W)
    os.remove(im1_path_crop)
    os.remove(im2_path_crop)
    
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
        x2[None], warp[:, :W_net, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1[None], warp[:, W_net:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    origin_im = torch.cat((x1, x2), dim=2)
    white_im = torch.ones((H_net, 2 * W_net), device=device)
    vis_im = warp_im
    if vis_certainty:
        vis_im = certainty * warp_im + (1 - certainty) * white_im
    else:
        vis_im[3] = origin_im[3]
    vis_im = torch.cat((origin_im, vis_im), dim=1)

    return vis_im, warp
