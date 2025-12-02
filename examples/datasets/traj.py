"""
Code borrowed from

https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py
"""

import numpy as np
import scipy
from scipy.spatial.transform import Rotation



def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses,
    bounds,
    n_frames=120,
    n_rots=2,
    zrate=0.5,
    spiral_scale_f=1.0,
    spiral_scale_r=1.0,
    focus_distance=0.75,
):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min()
    far_bound = bounds.max()
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (((1 - focus_distance) / near_bound + focus_distance / far_bound))
    focal = focal * spiral_scale_f

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = radii * spiral_scale_r
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], height])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


def generate_ellipse_path_y(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at y=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], height, center[2]])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    y_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    y_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-z.
        # Optionally also interpolate in y to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                variation
                * (
                    y_low[1]
                    + (y_high - y_low)[1]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
                low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def viewmatrix(z: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Constructs a 3x4 camera-to-world pose matrix.
    
    This function creates a pose matrix that defines the camera's orientation
    and position in the world.

    Args:
      z: The look-at direction vector (often world -Z).
      y: The up direction vector (often world +Y).
      p: The camera's position in world coordinates.

    Returns:
      A (3, 4) numpy array representing the camera pose.
    """
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    y_new = np.cross(z, x)
    y_new = y_new / np.linalg.norm(y_new)
    
    # Create the 3x4 pose matrix
    pose = np.stack([x, y_new, z, p], axis=1)
    return pose


def generate_interpolated_path_circle(
    poses: np.ndarray,
    n_interp: int,
):
    """
    입력된 카메라 포즈들을 기반으로 원형 경로를 생성합니다. (정합성 개선 버전)

    이 함수는 생성된 궤적의 첫 번째 프레임이 수학적으로 poses[0]과
    정확히 일치하도록 방정식을 수정하여, 자연스러운 시작을 보장합니다.

    1. poses의 평균 위치(`average_center`)를 계산합니다.
    2. `average_center`를 `poses[0]`의 위치와 방향으로 정의된 회전 평면에 투영하여,
       원의 정확한 중심(`circle_center`)을 찾습니다.
    3. 이 `circle_center`를 기준으로 `poses[0]`를 통과하는 원형 경로를 생성합니다.
    4. 카메라의 방향은 `poses[0]`의 초기 방향을 기준으로 함께 회전합니다.

    Args:
      poses: (n, 3, 4) 형태의 입력 포즈 키프레임 배열.
      n_interp: 반환될 경로에 포함될 전체 포즈의 수.

    Returns:
      (n_interp, 3, 4) 형태의 새로운 카메라 포즈 배열.
    """
    # === 1. 초기값 설정 ===
    start_pos = poses[0, :3, 3]
    # 회전축은 poses[0]의 'up' 벡터로, 회전 평면의 법선 벡터가 됩니다.
    pivot_axis = poses[0, :3, 1] / np.linalg.norm(poses[0, :3, 1])
    initial_rotation = poses[0, :3, :3]

    # === 2. 원의 정확한 중심 계산 ===
    # 가. 모든 포즈의 위치 평균을 구합니다.
    average_center = np.mean(poses[:, :3, 3], axis=0)
    
    # 나. 평균 중심을 poses[0]가 정의하는 회전 평면으로 투영(projection)합니다.
    #    이는 average_center와 start_pos를 잇는 벡터를 pivot_axis에 투영하여
    #    수직 성분을 제거하는 것과 같습니다.
    vec_to_avg_center = average_center - start_pos
    dist_from_plane = np.dot(vec_to_avg_center, pivot_axis)
    projection_vec = dist_from_plane * pivot_axis
    
    # 다. 투영된 점이 바로 원의 정확한 중심이 됩니다.
    circle_center = average_center - projection_vec
    
    # === 3. 원 궤적 파라미터 정의 ===
    radius = np.linalg.norm(start_pos - circle_center)
    if radius < 1e-6:
        # 중심과 시작점이 거의 같은 경우 (예: 포즈가 하나만 있음)
        # 카메라의 앞쪽을 기준으로 가상의 원을 만듭니다.
        radius = 1.0
        vec_to_start = -poses[0, :3, 2] # forward vector
        circle_center = start_pos - vec_to_start * radius
    
    # 원의 시작 방향(x축)과 오른쪽 방향(y축) 벡터를 정의합니다.
    x_axis = (start_pos - circle_center) / radius
    y_axis = np.cross(pivot_axis, x_axis)
    
    # === 4. 경로 생성 ===
    new_poses = []
    thetas = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)

    for theta in thetas:
        # 가. 원 위의 새로운 '위치' 계산
        current_pos = circle_center + radius * (x_axis * np.cos(theta) + y_axis * np.sin(theta))
        
        # 나. 초기 방향을 기준으로 회전된 새로운 '방향' 계산
        pivot_rotation = Rotation.from_rotvec(theta * pivot_axis).as_matrix()
        current_rotation = pivot_rotation @ initial_rotation
        
        # 다. 위치와 방향을 합쳐 새로운 3x4 포즈 행렬 생성
        pose = np.eye(4)[:3, :]
        pose[:3, :3] = current_rotation
        pose[:3, 3] = current_pos
        new_poses.append(pose)

        
    new_poses[0] = poses[0][:3]

    return np.array(new_poses)


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)
