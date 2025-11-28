# This script is for "hourglass" in diva360
import argparse
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import os


def load_img(path, downsample=1):
    img = ToTensor()(Image.open(path))
    if downsample > 1:
        img = img[:, ::downsample, ::downsample]
    return img


def save_img(tensor, path):
    tensor = tensor.squeeze().detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in [3, 4]:
        tensor = tensor.permute(1, 2, 0)

    tensor = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    
    args = parser.parse_args()
    
    img_path_list = os.listdir(args.folder_path)

    for img_path in img_path_list:
        img_path = os.path.join(args.folder_path, img_path)
        img = load_img(img_path)
        new_img = img * img[3:]
        save_img(new_img, img_path)
