import json
from collections import OrderedDict
from pycolmap.scene_manager import Quaternion, Image, Camera
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from PIL import Image as PILImage


def dfa_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    # change deformation
    c2w = c2w[[1, 0, 2, 3], :]
    c2w = c2w[:, [1, 2, 0, 3]]

    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]

    tvec = w2c[:3, -1]

    rotation = R.from_matrix(rot)
    qvec = rotation.as_quat()  # Returns [x, y, z, w]
    qvec = np.array(qvec)[[3, 0, 1, 2]]

    return qvec, tvec


class SceneManagerDFA:
    def __init__(self, data_dir="./gsplat/data/DFA_processed/beagle_dog(s1)/0"):
        self.data_dir = data_dir
        self.cameras = OrderedDict()
        self.images = OrderedDict()
        self.name_to_image_id = {}
        self.last_image_id = 0

        self.width, self.height = self._get_image_resolution()
        self.intrinsics = self._load_intrinsics()
        self.n_cameras = len(self.intrinsics)

    def _get_image_resolution(self):
        img_dir = os.path.join(self.data_dir, "images")
        rgb_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if not rgb_files:
            raise Exception("No RGB image files found in the first frame folder.")
        first_rgb_path = os.path.join(img_dir, rgb_files[0])
        with PILImage.open(first_rgb_path) as im:
            width, height = im.size
        return width, height

    def _load_intrinsics(self):
        intrinsics_path = os.path.join(self.data_dir, "Intrinsic.inf")
        intrinsics = {}
        with open(intrinsics_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != ""]

        i = 0
        while i < len(lines):
            cam_index = int(lines[i])
            row1 = [float(x) for x in lines[i + 1].split()]
            row2 = [float(x) for x in lines[i + 2].split()]
            row3 = [float(x) for x in lines[i + 3].split()]
            
            # Intrinsics matrix:
            # [ fx    0   cx ]
            # [  0   fy   cy ]
            # [  0    0    1 ]
            fx = row1[0]
            cx = row1[2]
            fy = row2[1]
            cy = row2[2]
            intrinsics[cam_index] = (fx, fy, cx, cy)
            i += 4
        return intrinsics

    def load_cameras(self):
        for cam_index in sorted(self.intrinsics.keys()):
            fx, fy, cx, cy = self.intrinsics[cam_index]
            new_cam_id = cam_index + 1
            camera = Camera(1, self.width, self.height, [fx, fy, cx, cy])
            self.cameras[new_cam_id] = camera

    def _load_extrinsics(self):
        extrinsics_path = os.path.join(self.data_dir, "Campose.inf")
        extrinsics_list = []
        with open(extrinsics_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip() != ""]
        for line in lines:
            parts = line.split()
            if len(parts) != 12:
                raise Exception(f"Line in CamPose.inf does not contain 12 numbers: {line}")
            nums = [float(x) for x in parts]
            mat_4x3 = np.array(nums).reshape(4, 3)
            mat_4x4 = np.zeros((4, 4))
            mat_4x4[:3, :3] = mat_4x3[:3, :3].T
            mat_4x4[:3, 3] = mat_4x3[3, :]
            mat_4x4[3, :] = np.array([0, 0, 0, 1])
            extrinsics_list.append(mat_4x4)
        
        return extrinsics_list

    def load_extrinsics(self):
        self.extrinsics_list = self._load_extrinsics()
        assert len(self.extrinsics_list) == self.n_cameras
        
        for view in range(self.n_cameras):
            file_name = f"img_{view:04d}_rgba.png"
            image_path = os.path.join(self.data_dir, "images", file_name)
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} does not exist.")
                continue
            transform = self.extrinsics_list[view]
            if transform.shape == (4, 4):
                qvec, tvec = dfa_to_colmap(transform)
                q = Quaternion(qvec)
                image = Image(file_name, view + 1, q, tvec)
                self.last_image_id += 1
                self.images[self.last_image_id] = image
                self.name_to_image_id[image_path] = self.last_image_id

    def genrate_random_points3D(self, n=10000):
        translation_list = np.empty((len(self.images), 3))
        for camera_id, image in self.images.items():
            translation = image.tvec
            translation_list[camera_id - 1] = translation

        self.bbox = np.array(
            [np.min(translation_list, axis=0), np.max(translation_list, axis=0)]
        )
        self.points3D = np.random.uniform(-0.5, 0.5, (n, 3))
        self.point3D_colors = np.ones((n, 3), dtype=np.uint8) * 50
        self.point3D_errors = np.random.rand(n)
        self.point3D_id_to_images = {}


# Example usage
if __name__ == "__main__":
    manager = SceneManagerDFA()
    manager.load_cameras()
    manager.load_extrinsics()
    manager.genrate_random_points3D()

    for image_id, image in manager.images.items():
        camera = manager.cameras[image.camera_id]
