import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from typing_extensions import Literal, assert_never

# Types for strategy
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from easydict import EasyDict



@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None

    render_traj_all: bool = False

    render_traj_simple: bool = False

    video_path: str = None
    # Render trajectory path
    render_traj_path: str = "diva360_spiral"  # "interp", "ellipse", "spiral"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 1e4
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]
    )
    # Steps to save the model
    save_steps: List[int] = field(
        default_factory=lambda: [3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]
    )

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1e3
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 0
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 1e2
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "vgg"

    data_name: str = "diva360"

    single_finetune: bool = False

    cam_idx: int = 0

    wandb: bool = False

    object_name: str = None

    wandb_group: str = None

    wandb_sweep: bool = False
    
    without_group: bool = False

    without_group_refine: bool = False

    naive_group: bool = False
    
    motion_video: bool = False

    video_name: str = "video"
    
    skip_eval: bool = False

    simple_video: bool = False

    finetune_with_only_rgb: bool = False


    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)



hyperparam = EasyDict({
    "DFA": {
        "anchor_k": 9,
        "coef_arap_drag": 2e3,
        "coef_drag": 0.5,
        "coef_drag_3d": 3e3,
        "coef_group_arap": 5e2,
        "coef_rgb": 5e3,
        "confidence": 0.99,
        "cycle_threshold": 10,
        "lr_motion": 3e-4,
        "lr_q": 0.03,
        "lr_t": 0.e23,
        "min_inlier_ratio": 0.7,
        "n_anchor": 300,
        "rbf_gamma": 50,
        "reprojection_error": 6,
        "rigidity_k": 50,
        "vis_threshold": 0.5,
        "refine_radius": 0.05,
        "refine_threhold": 0.01,
        "voxel_size": 0.02,
        "filter_distance": 1,
        "min_group_size": 1e2,
    },
    "diva360": {
        "anchor_k": 10,
        "coef_arap_drag": 1e4,
        "coef_drag": 0.5,
        "coef_drag_3d": 3e3,
        "coef_group_arap": 1e3,
        "coef_rgb": 5e4,
        "confidence": 0.97,
        "cycle_threshold": 10,
        "lr_motion": 1e-3,
        "lr_q": 0.05,
        "lr_t": 0.01,
        "n_anchor": 300,
        "min_inlier_ratio": 0.7,
        "rbf_gamma": 50,
        "refine_radius": 0.1,
        "refine_threhold": 0.01,
        "reprojection_error": 8,
        "rigidity_k": 25,
        "vis_threshold": 0.3,
        "voxel_size": 0.06,
        "filter_distance": 1,
        "min_group_size": 25,
    }
})