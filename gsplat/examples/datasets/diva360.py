import json
from collections import OrderedDict
from pycolmap.scene_manager import Quaternion, Image, Camera
import numpy as np
from scipy.spatial.transform import Rotation as R


def diva360_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[0:3, 1] *= -1  # flip the y and z axis
    c2w[0:3, 2] *= -1

    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]
    tvec = w2c[:3, -1]

    rotation = R.from_matrix(rot)
    qvec = rotation.as_quat()  # Returns [x, y, z, w]
    qvec = np.array(qvec)[[3, 0, 1, 2]]

    return qvec, tvec


class SceneManagerDiva360:
    def __init__(self, json_path):
        self.json_path = json_path
        self.cameras = OrderedDict()
        self.images = OrderedDict()
        self.name_to_image_id = {}
        self.last_image_id = 0

    def load_cameras_from_json(self):
        with open(self.json_path, "r") as file:
            data = json.load(file)

        frames = data.get("frames", [])

        for camera_id, frame in enumerate(frames, 1):
            width = frame.get("w", 0)
            height = frame.get("h", 0)
            fl_x = frame.get("fl_x", 0.0)
            fl_y = frame.get("fl_y", 0.0)
            cx = frame.get("cx", 0.0)
            cy = frame.get("cy", 0.0)

            # Create a Camera object
            camera = Camera(1, width, height, [fl_x, fl_y, cx, cy])

            self.cameras[camera_id] = camera

    def load_extrinsics_from_json(self):
        with open(self.json_path, "r") as file:
            data = json.load(file)

        frames = data.get("frames", [])

        for camera_id, frame in enumerate(frames, 1):
            file_name = frame.get("file_path", "")
            transform = np.array(frame.get("transform_matrix", []))

            if transform.shape == (4, 4):
                qvec, tvec = diva360_to_colmap(transform)
                q = Quaternion(qvec)

                # Create an Image object
                image = Image(file_name, camera_id, q, tvec)

                self.images[camera_id] = image
                self.name_to_image_id[file_name] = camera_id

                self.last_image_id = max(self.last_image_id, camera_id)

    def genrate_random_points3D(self, n=5000):
        translation_list = np.empty((len(self.images), 3))
        for camera_id, image in self.images.items():
            translation = image.tvec
            translation_list[camera_id - 1] = translation

        self.bbox = np.array(
            [np.min(translation_list, axis=0), np.max(translation_list, axis=0)]
        )
        self.points3D = np.random.uniform(-1, 1, (n, 3))
        self.point3D_colors = np.ones((n, 3), dtype=np.uint8) * 255
        self.point3D_errors = np.random.rand(n)
        self.point3D_id_to_images = {}


# Example usage
if __name__ == "__main__":
    json_path = "./gsplat/data/diva360_processed/penguin_0217/cameras.json"
    manager = SceneManagerDiva360(json_path)
    manager.load_cameras_from_json()
    manager.load_extrinsics_from_json()
    manager.genrate_random_points3D()

    for image_id, image in manager.images.items():
        camera = manager.cameras[image.camera_id]
