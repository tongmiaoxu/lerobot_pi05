"""
Eval pipeline for xArm7 + Gaussian Splatting compositing.

Adapted from eval_pipeline_advait_new.py (ALOHA version):
  - MuJoCo model: xarm7/scene.xml (instead of aloha/robolab_setup.xml)
  - Robot geom detection: xArm body names
  - Observation state: 8-DOF (7 joints + gripper) instead of 14-DOF
  - Camera intrinsics/extrinsics: loaded from configs/ JSON files
  - Camera pose: computed from calibration (camera_config.py)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HAS_DISPLAY = os.environ.get("DISPLAY") is not None
if not _HAS_DISPLAY:
    os.environ["MUJOCO_GL"] = "egl"

import time
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d

import mujoco
from mujoco import MjData, MjModel

from camera_config import load_camera_config, set_mujoco_camera_from_config

# ===== ICP alignment transform (Gaussian Splatting <-> MuJoCo) =====
ICP_TRANSFORM_PATH = "pointclouds/icp_transform.npy"
if os.path.exists(ICP_TRANSFORM_PATH):
    T_splat2mj = np.load(ICP_TRANSFORM_PATH)
else:
    raise ValueError("ICP transform not found")

# ===== Load camera calibration from config =====
_stationary_cfg = load_camera_config("stationary_cam")

CAM_POS_MJ = _stationary_cfg["cam_pos_mj"]
CAM_XMAT_MJ = _stationary_cfg["cam_xmat_mj"]
REALSENSE_INTRINSICS_640x480 = _stationary_cfg["intrinsics"]

# Expose intermediate transforms for backwards compatibility
R_cam2board = _stationary_cfg["R_cam2board"]
t_cam2board = _stationary_cfg["t_cam2board"]
R_world2base = _stationary_cfg["R_world2base"]
t_world2base = _stationary_cfg["t_world2base"]
R_cam2base = _stationary_cfg["R_cam2base"]
t_cam_in_base = _stationary_cfg["t_cam_in_base"]


def set_camera_from_calibration(data, model, cam_name):
    """
    Override the MuJoCo camera pose with the real camera calibration.
    """
    return set_mujoco_camera_from_config(data, model, cam_name, _stationary_cfg)



if _HAS_DISPLAY:
    import mujoco.viewer

try:
    from splatam.utils.recon_helpers import setup_camera
except ImportError:
    print("Warning: splatam.utils.recon_helpers.setup_camera not found. Using basic implementation.")
    class CameraParams:
        def __init__(self, image_height, image_width, tanfovx, tanfovy, scale_modifier,
                     viewmatrix, projmatrix, sh_degree, campos, prefiltered):
            self.image_height = image_height
            self.image_width = image_width
            self.tanfovx = tanfovx
            self.tanfovy = tanfovy
            self.scale_modifier = scale_modifier
            self.viewmatrix = viewmatrix
            self.projmatrix = projmatrix
            self.sh_degree = sh_degree
            self.campos = campos
            self.prefiltered = prefiltered

    def setup_camera(w, h, k, w2c, near=0.01, far=100):
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w2c = torch.tensor(w2c).cuda().float()
        cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = CameraParams(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False
        )
        return cam



# ===== FOREGROUND FUNCTIONS (MuJoCo) =====

def get_robot_geom_ids(model, include_list=None):
    """
    Returns geom IDs belonging to the xArm robot (all mesh geoms, base cylinder).
    Optionally include mug, sticker, table based on include_list.
    Args:
        model: MuJoCo model
        include_list: [mug, sticker, table] (1=include, 0=exclude)
    """
    # Default: exclude all
    if include_list is None:
        include_list = [1, 1, 1]
    mug_flag, sticker_flag, table_flag = include_list
    # Build exclusion set
    EXCLUDE_GEOMS = set()
    if not mug_flag:
        EXCLUDE_GEOMS.add("mug")
    if not sticker_flag:
        EXCLUDE_GEOMS.add("sticker")
    if not table_flag:
        EXCLUDE_GEOMS.add("table")

    robot_geom_ids = set()
    # Hide excluded geoms (mesh or box) by setting alpha to 0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name in EXCLUDE_GEOMS:
            model.geom_rgba[geom_id, 3] = 0.0  # fully transparent
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            robot_geom_ids.add(geom_id)
    # Also check for box geoms (e.g. sticker)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name in EXCLUDE_GEOMS:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            robot_geom_ids.add(geom_id)
    # Include robot base cylinder (white cylinder at origin)
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not geom_name:
                robot_geom_ids.add(geom_id)
    # Optionally include table as foreground object
    if table_flag:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        if geom_id != -1:
            robot_geom_ids.add(geom_id)
    return robot_geom_ids


def get_mujoco_camera_pose(model, data, cam_name):
    """Get camera pose (4x4 c2w matrix) from MuJoCo model and data."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{cam_name}' not found")
    camera_xpos = data.cam_xpos[cam_id]
    camera_xmat = data.cam_xmat[cam_id].reshape(3, 3)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_xmat
    camera_pose[:3, 3] = camera_xpos
    return camera_pose


def mj_pose_to_gaussian_w2c(camera_pose, T_splat2mj):
    """Convert MuJoCo camera pose (c2w) to Gaussian Splatting w2c matrix."""
    T_mj2splat = np.linalg.inv(T_splat2mj)
    P_gs_cam = T_mj2splat @ camera_pose
    transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    w2c = P_gs_cam @ transform_matrix
    w2c = np.linalg.inv(w2c)
    return w2c


def shift_for_principal_point(image, intrinsics, seg=False):
    """Shift a MuJoCo render to compensate for off-center principal point.

    MuJoCo always renders with the principal point at the image center
    (w/2, h/2).  The real camera has (cx, cy) which may differ by a few
    pixels.  This function translates the rendered image by
    (cx - w/2, cy - h/2) so that it aligns with a renderer that uses the
    true intrinsics (e.g. Gaussian Splatting).

    Args:
        image: H×W or H×W×C numpy array (uint8 or int32).
        intrinsics: 3×3 camera matrix with fx, fy, cx, cy.
        seg: If True, use nearest-neighbour interpolation (for
             segmentation labels); otherwise bilinear (for RGB).

    Returns:
        Shifted image of the same shape and dtype.
    """
    h, w = image.shape[:2]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    tx = cx - w / 2.0
    ty = cy - h / 2.0
    # Skip if the shift is negligible (<0.05 px)
    if abs(tx) < 0.05 and abs(ty) < 0.05:
        return image
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    flags = cv2.INTER_NEAREST if seg else cv2.INTER_LINEAR
    border = cv2.BORDER_REPLICATE if not seg else cv2.BORDER_CONSTANT
    shifted = cv2.warpAffine(image, M, (w, h), flags=flags,
                             borderMode=border, borderValue=0)
    return shifted


# xArm: single stationary camera
xarm_cam_name = "stationary_cam"


def create_observation(model, prompt, data):
    """
    Creates an observation dict for xArm7.
    State: 8-DOF (7 arm joints + 1 gripper driver joint).
    """
    state = np.zeros((8,))
    state[:7] = data.qpos[:7]       # 7 arm joints (radians)
    state[7] = data.qpos[7]         # left_driver_joint (gripper, 0=open, 0.85=closed)

    images = {}
    renderer.update_scene(data, camera=xarm_cam_name)
    image = renderer.render()
    image = np.transpose(image, (2, 0, 1))  # (C, H, W)
    images["stationary_cam"] = image

    observation = {"state": state, "images": images, "prompt": prompt}
    return observation


# ===== BACKGROUND FUNCTIONS (3DGS: Gaussian Splatting) =====
# These are identical to the ALOHA version.

def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    org_width = all_params['org_width']
    org_height = all_params['org_height']
    w2c = all_params['w2c']
    intrinsics = all_params['intrinsics']
    k = intrinsics[:3, :3].copy()
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    for k in all_params.keys():
        all_params[k] = torch.tensor(all_params[k]).cuda().float()
    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    def build_rotation(quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)]),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)]),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        ])
        return R

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = params['cam_unnorm_rots'][..., t_i]
        cam_rot = F.normalize(cam_rot, dim=0)
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    if params['log_scales'].shape[-1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }

    def get_depth_and_silhouette(means3D, w2c):
        ones = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
        points_homogeneous = torch.cat([means3D, ones], dim=1)
        cam_points = (w2c @ points_homogeneous.T).T
        depth = cam_points[:, 2]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = torch.zeros_like(depth)
        colors = depth_norm.unsqueeze(1).repeat(1, 3)
        return colors

    depth_rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': None,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar['colors_precomp'] = get_depth_and_silhouette(params['means3D'], first_frame_w2c)
    return rendervar, depth_rendervar, all_w2cs


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    try:
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    except ImportError as e:
        raise ImportError(
            f"diff_gaussian_rasterization is not installed: {e}\n\n"
            "To install it, you need:\n"
            "  1. CUDA toolkit (conda install -c nvidia cuda-toolkit)\n"
            "  2. Build tools (ninja, gcc)\n"
            "  3. Then run: pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git\n"
        )
    cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
    white_bg_cam = Camera(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
        scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
        campos=cam.campos, prefiltered=cam.prefiltered, debug=False
    )
    black_bg_cam = Camera(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
        projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
        campos=cam.campos, prefiltered=cam.prefiltered, debug=False
    )
    im, depth = Renderer(raster_settings=white_bg_cam)(**timestep_data)
    depth_sil, _ = Renderer(raster_settings=black_bg_cam)(**timestep_depth_data)
    return im, depth, depth_sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=float(torch.max(cols).item()))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


# ===== INTERACTIVE VIEWER =====

class InteractiveCompositeViewer:
    """
    Interactive viewer for composite MuJoCo + Gaussian Splatting rendering.

    Uses **matplotlib** with three independent subplots so each panel can be
    zoomed / panned individually via the mouse scroll-wheel.

    Controls
    --------
    * **Left-drag** on any panel        → rotate the 3-D camera  (all panels update)
    * **Middle-drag** on any panel      → pan the 3-D camera     (all panels update, like Open3D)
    * **Right-drag** on any panel       → pan the 3-D camera     (all panels update)
    * **Scroll-wheel**                  → zoom 3-D camera forward/backward (all panels update)
    * **r**                            → reset camera + reset all zoom levels
    * **s**                            → save current views to ``images/``
    * **q**                            → quit
    """

    def __init__(self, model, data, renderer, seg_renderer, scene_data, scene_depth_data,
                 viz_cfg, k, camera_intrinsics, target_geom_ids, cam_name, T_splat2mj):
        self.model = model
        self.data = data
        self.renderer = renderer
        self.seg_renderer = seg_renderer
        self.scene_data = scene_data
        self.scene_depth_data = scene_depth_data
        self.viz_cfg = viz_cfg
        self.k = k
        self.camera_intrinsics = camera_intrinsics
        self.target_geom_ids = target_geom_ids
        self.cam_name = cam_name
        self.T_splat2mj = T_splat2mj

        self.camera_pose = self._get_initial_camera_pose()

        # Image dimensions (for zoom clamping)
        self._img_w = viz_cfg['viz_w']
        self._img_h = viz_cfg['viz_h']

        # Mouse-drag state
        self._dragging = False
        self._drag_button = None
        self._last_xy = None

        # ---- Disable default matplotlib key bindings that clash with ours ----
        for key_list_name in ('keymap.save', 'keymap.quit', 'keymap.quit_all',
                              'keymap.fullscreen', 'keymap.home', 'keymap.back',
                              'keymap.forward', 'keymap.pan', 'keymap.zoom'):
            try:
                plt.rcParams[key_list_name] = []
            except KeyError:
                pass

        # ---- Create figure with three subplots ----
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.canvas.manager.set_window_title(
            "Interactive Composite Viewer  |  Drag=rotate  Right-drag=pan  Scroll=zoom-panel  r=reset  +/-=alpha  q=quit")
        titles = ["MuJoCo Foreground", "Gaussian Background", "Alpha Blend (α=0.50)"]
        for ax, title in zip(self.axes, titles):
            ax.set_title(title)
            ax.axis('off')

        # Alpha-blend factor for the third panel (0=background only, 1=robot only)
        self._blend_alpha = 0.5

        # Handles for imshow images (set on first render)
        self._im_handles = [None, None, None]

        # ---- Connect matplotlib events ----
        self._cids = [
            self.fig.canvas.mpl_connect('button_press_event',  self._on_press),
            self.fig.canvas.mpl_connect('button_release_event', self._on_release),
            self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion),
            self.fig.canvas.mpl_connect('scroll_event',        self._on_scroll),
            self.fig.canvas.mpl_connect('key_press_event',     self._on_key),
        ]

        # Initial render
        self._render_and_display()
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------
    def _get_initial_camera_pose(self):
        """Get camera pose from calibration (already set on data)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        camera_xpos = self.data.cam_xpos[cam_id]
        camera_xmat = self.data.cam_xmat[cam_id].reshape(3, 3)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_xmat
        camera_pose[:3, 3] = camera_xpos
        return camera_pose

    # ------------------------------------------------------------------
    # Matplotlib event callbacks
    # ------------------------------------------------------------------
    def _on_press(self, event):
        if event.inaxes is None:
            return
        if event.button in (1, 2, 3):       # left / middle / right
            self._dragging = True
            self._drag_button = event.button
            self._last_xy = (event.x, event.y)   # display-pixel coords

    def _on_release(self, event):
        self._dragging = False
        self._drag_button = None
        self._last_xy = None

    def _on_motion(self, event):
        if not self._dragging or self._last_xy is None:
            return
        dx = event.x - self._last_xy[0]
        dy = event.y - self._last_xy[1]
        self._last_xy = (event.x, event.y)

        if self._drag_button == 1:              # left = rotate 3-D camera
            self._rotate_camera(dx, dy)
            self._render_and_display()
        elif self._drag_button in (2, 3):       # middle or right = pan 3-D camera
            self._pan_camera(dx, dy)
            self._render_and_display()

    def _on_scroll(self, event):
        """Zoom by moving the 3-D camera along its viewing direction.

        Scroll-up  → move camera forward  (zoom in).
        Scroll-down → move camera backward (zoom out).
        Same convention as Open3D / most 3-D viewers.
        All three panels update together.
        """
        if event.button == 'up':
            zoom_factor = 1.1   # move toward scene
        else:
            zoom_factor = 0.9   # move away from scene

        forward = -self.camera_pose[:3, 2]          # camera looks along -Z
        zoom_amount = (zoom_factor - 1.0)            # +0.1 or -0.1
        self.camera_pose[:3, 3] += forward * zoom_amount
        self._render_and_display()

    def _on_key(self, event):
        if event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'r':
            self.camera_pose = self._get_initial_camera_pose()
            # Reset zoom on all subplots to full image
            for ax in self.axes:
                ax.set_xlim(-0.5, self._img_w - 0.5)
                ax.set_ylim(self._img_h - 0.5, -0.5)
            self._render_and_display()
        elif event.key == 's':
            self._save_views()
        elif event.key in ('+', '='):   # increase robot opacity
            self._blend_alpha = min(1.0, self._blend_alpha + 0.05)
            self._update_blend_title()
            self._render_and_display()
        elif event.key == '-':          # decrease robot opacity
            self._blend_alpha = max(0.0, self._blend_alpha - 0.05)
            self._update_blend_title()
            self._render_and_display()

    def _update_blend_title(self):
        self.axes[2].set_title(f"Alpha Blend (α={self._blend_alpha:.2f})")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 3-D camera manipulation  (affects all three panels)
    # ------------------------------------------------------------------
    def _rotate_camera(self, dx, dy):
        rot_speed = 0.01
        angle_x = -dy * rot_speed
        angle_y = -dx * rot_speed
        pos = self.camera_pose[:3, 3]
        rot = self.camera_pose[:3, :3]
        rot_x = R.from_euler('x', angle_x, degrees=False).as_matrix()
        rot_y = R.from_euler('y', angle_y, degrees=False).as_matrix()
        new_rot = rot @ rot_x @ rot_y
        self.camera_pose[:3, :3] = new_rot
        self.camera_pose[:3, 3] = pos

    def _pan_camera(self, dx, dy):
        pan_speed = 0.001
        right = self.camera_pose[:3, 0]
        up = self.camera_pose[:3, 1]
        translation = (right * dx * pan_speed) - (up * dy * pan_speed)
        self.camera_pose[:3, 3] += translation

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_mujoco_rgb_and_geomseg_from_pose(self, camera_pose_c2w):
        cam_pos = camera_pose_c2w[:3, 3]
        cam_rot_c2w = camera_pose_c2w[:3, :3]

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if cam_id == -1:
            raise ValueError(f"Camera '{self.cam_name}' not found in the model.")

        original_pos = self.data.cam_xpos[cam_id].copy()
        original_mat = self.data.cam_xmat[cam_id].copy()

        self.data.cam_xpos[cam_id] = cam_pos
        self.data.cam_xmat[cam_id] = cam_rot_c2w.flatten()

        self.renderer.update_scene(self.data, camera=cam_id)
        rgb = self.renderer.render()
        fg_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        self.seg_renderer.update_scene(self.data, camera=cam_id)
        seg = self.seg_renderer.render()
        seg_labels = seg[:, :, 0].astype(np.int32)
        # Note: background pixels have seg label -1; do NOT remap to 0
        # because geom ID 0 is the robot base cylinder.

        # Compensate for off-center principal point (MuJoCo assumes centered)
        fg_bgr = shift_for_principal_point(fg_bgr, self.camera_intrinsics)
        seg_labels = shift_for_principal_point(seg_labels, self.camera_intrinsics, seg=True)

        self.data.cam_xpos[cam_id] = original_pos
        self.data.cam_xmat[cam_id] = original_mat

        return fg_bgr, seg_labels

    def _compute_views(self):
        """Render all three views and return them as RGB numpy arrays."""
        fg_image_bgr, seg_labels = self._render_mujoco_rgb_and_geomseg_from_pose(self.camera_pose)

        robot_mask_binary = np.isin(seg_labels, list(self.target_geom_ids))
        mask_uint8 = (robot_mask_binary.astype(np.uint8)) * 255
        foreground_masked = cv2.bitwise_and(fg_image_bgr, fg_image_bgr, mask=mask_uint8)

        P_mj_cam = self.camera_pose.copy()
        T_mj2splat = np.linalg.inv(self.T_splat2mj)
        P_gs_cam = T_mj2splat @ P_mj_cam
        w2c_bg = P_gs_cam

        transform_matrix = np.array([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]])
        w2c_bg = w2c_bg @ transform_matrix
        w2c_bg = np.linalg.inv(w2c_bg)

        bg_im, depth, sil = render(w2c_bg, self.k, self.scene_data,
                                   self.scene_depth_data, self.viz_cfg)

        bg_im_np = bg_im.permute(1, 2, 0).cpu().numpy()
        bg_im_np = (bg_im_np * 255).astype(np.uint8)

        # Alpha blend (work in BGR, then convert everything to RGB for matplotlib)
        bg_bgr = cv2.cvtColor(bg_im_np, cv2.COLOR_RGB2BGR)
        alpha = self._blend_alpha
        mask_bool = mask_uint8 > 0
        composite_bgr = bg_bgr.copy()
        composite_bgr[mask_bool] = (
            alpha * foreground_masked[mask_bool].astype(np.float32)
            + (1.0 - alpha) * bg_bgr[mask_bool].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        fg_rgb = cv2.cvtColor(fg_image_bgr, cv2.COLOR_BGR2RGB)
        bg_rgb = bg_im_np  # already RGB
        composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

        return fg_rgb, bg_rgb, composite_rgb, fg_image_bgr, bg_bgr, composite_bgr

    def _render_and_display(self):
        fg_rgb, bg_rgb, composite_rgb, _, _, _ = self._compute_views()
        images = [fg_rgb, bg_rgb, composite_rgb]

        for i, (ax, img) in enumerate(zip(self.axes, images)):
            if self._im_handles[i] is None:
                self._im_handles[i] = ax.imshow(img)
            else:
                self._im_handles[i].set_data(img)

        self.fig.canvas.draw_idle()

    def _save_views(self):
        _, _, _, fg_bgr, bg_bgr, composite_bgr = self._compute_views()
        os.makedirs("images", exist_ok=True)
        timestamp = int(time.time())
        cv2.imwrite(f"images/xarm_fg_{timestamp}.png", fg_bgr)
        cv2.imwrite(f"images/xarm_bg_{timestamp}.png", bg_bgr)
        cv2.imwrite(f"images/xarm_composite_{timestamp}.png", composite_bgr)
        print(f"Saved: xarm_fg_{timestamp}.png, xarm_bg_{timestamp}.png, "
              f"xarm_composite_{timestamp}.png")

    # ------------------------------------------------------------------
    def run(self):
        print("\n=== Interactive Composite Viewer (xArm) ===")
        print("Controls:")
        print("  - Left mouse drag  : Rotate 3-D camera (all panels update)")
        print("  - Middle-click drag: Pan 3-D camera    (all panels update, like Open3D)")
        print("  - Right mouse drag : Pan 3-D camera    (all panels update)")
        print("  - Scroll wheel     : Move camera forward/backward (all panels update)")
        print("  - Press 'r'        : Reset camera & zoom")
        print("  - Press 's'        : Save current views")
        print("  - Press '+'/'-'    : Increase/decrease robot alpha in blend panel")
        print("  - Press 'q'        : Quit")
        print("=============================================\n")

        plt.show()   # enters matplotlib event loop; returns when window is closed


# ===== MAIN =====

def main():
    # ----- Setup MuJoCo (xArm7) -----
    model_path = "./xarm7/scene.xml"
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)

    im_height = 480
    im_width = 640

    # Override MuJoCo fovy to match the real camera intrinsics exactly.
    # MuJoCo uses a symmetric projection so we can only match fy, not cx/cy offset.
    fy = REALSENSE_INTRINSICS_640x480[1, 1]
    correct_fovy_deg = float(2.0 * np.degrees(np.arctan(im_height / (2.0 * fy))))
    cam_id_tmp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, xarm_cam_name)
    old_fovy = model.cam_fovy[cam_id_tmp]
    model.cam_fovy[cam_id_tmp] = correct_fovy_deg
    print(f"Corrected MuJoCo fovy: {old_fovy:.2f}° → {correct_fovy_deg:.2f}° "
          f"(from fy={fy:.3f}, h={im_height})")

    global renderer
    renderer = mujoco.Renderer(model, height=im_height, width=im_width)
    seg_renderer = mujoco.Renderer(model, height=im_height, width=im_width)
    seg_renderer.enable_segmentation_rendering()

    # Override camera pose from calibration (after mj_step so data is populated)
    cam_id = set_camera_from_calibration(data, model, xarm_cam_name)
    print(f"Camera '{xarm_cam_name}' (id={cam_id}) pose set from calibration:")
    print(f"  Position: {CAM_POS_MJ}")
    print(f"  Rotation:\n{CAM_XMAT_MJ}")

    # Robot geom IDs (all mesh geoms = xArm links + gripper)
    robot_geom_ids = get_robot_geom_ids(model)
    target_geom_ids = robot_geom_ids.copy()
    print(f"xArm robot geom IDs: {sorted(target_geom_ids)}")

    prompt = "Pick up the cube"

    # ----- Setup background (3DGS) -----
    scene_path = "pointclouds/xarm7_black.npz"
    cam_name = xarm_cam_name

    viz_cfg = {
        'viz_w': im_width,
        'viz_h': im_height,
        'viz_near': 0.1,
        'viz_far': 10.0,
        'view_scale': 1.0,
        'sh_degree': 0,
        'offset_first_viz_cam': False,
        'render_mode': 'rgb',
        'show_sil': False
    }

    # Use intrinsics from config (stationary_cam.json)
    k = REALSENSE_INTRINSICS_640x480.copy()
    serial = _stationary_cfg.get("serial_number", "N/A")
    print(f"Using stationary_cam intrinsics (serial {serial}):")
    print(f"  fx={k[0,0]:.3f}, fy={k[1,1]:.3f}, cx={k[0,2]:.3f}, cy={k[1,2]:.3f}")

    # Use calibrated camera pose for w2c_init (not from npz, which may have old camera baked in)
    init_pose = get_mujoco_camera_pose(model, data, xarm_cam_name)
    w2c_bg_init = mj_pose_to_gaussian_w2c(init_pose, T_splat2mj)

    scene_data, scene_depth_data, all_w2cs = load_scene_data(scene_path, w2c_bg_init, k)
    scene_radius = torch.norm(scene_data['means3D'], dim=1).median()
    print("Scene radius:", scene_radius.item())

    # Camera pose (from calibration)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = CAM_XMAT_MJ
    camera_pose[:3, 3] = CAM_POS_MJ

    start_time = time.time()

    # ----- Main simulation loop -----
    def run_simulation(viewer=None):
        count = 0
        while count < 1000:
            # Re-apply calibrated camera pose (mj_step may reset it)
            set_camera_from_calibration(data, model, xarm_cam_name)

            # (1) Get foreground observation from MuJoCo
            obs = create_observation(model, prompt, data)
            renderer.update_scene(data, camera=cam_name)
            fg_image = renderer.render()
            fg_image_bgr = cv2.cvtColor(fg_image, cv2.COLOR_RGB2BGR)

            # Compensate for off-center principal point (MuJoCo assumes centered)
            fg_image_bgr = shift_for_principal_point(fg_image_bgr, k)

            # (2) Get segmentation mask and extract foreground (robot)
            seg_renderer.update_scene(data, camera=cam_name)
            seg_mask = seg_renderer.render()
            seg_labels = seg_mask[:, :, 0]
            # Note: background pixels have seg label -1; do NOT remap to 0
            # because geom ID 0 is the robot base cylinder.
            seg_labels = shift_for_principal_point(seg_labels.astype(np.int32), k, seg=True)
            robot_mask_binary = np.isin(seg_labels, list(target_geom_ids))
            mask_uint8 = (robot_mask_binary.astype(np.uint8)) * 255
            foreground_masked = cv2.bitwise_and(fg_image_bgr, fg_image_bgr, mask=mask_uint8)

            # (3) Get background image from 3DGS
            P_mj_cam = camera_pose.copy()
            T_mj2splat = np.linalg.inv(T_splat2mj)
            P_gs_cam = T_mj2splat @ P_mj_cam
            w2c_bg = P_gs_cam

            transform_matrix = np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 0, 0, 1]])
            w2c_bg = w2c_bg @ transform_matrix
            w2c_bg = np.linalg.inv(w2c_bg)

            bg_im, depth, sil = render(w2c_bg, k, scene_data, scene_depth_data, viz_cfg)

            bg_im_np = bg_im.permute(1, 2, 0).cpu().numpy()
            bg_im_np = (bg_im_np * 255).astype(np.uint8)
            bg_im_np = cv2.cvtColor(bg_im_np, cv2.COLOR_RGB2BGR)

            # (4) Composite
            composite = bg_im_np.copy()
            composite[mask_uint8 > 0] = foreground_masked[mask_uint8 > 0]


            # (6) Display
            if _HAS_DISPLAY:
                if count == 0:
                    interactive_viewer = InteractiveCompositeViewer(
                        model, data, renderer, seg_renderer, scene_data, scene_depth_data,
                        viz_cfg, k, k, target_geom_ids, cam_name, T_splat2mj
                    )
                    interactive_viewer.run()
                    break
                else:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.title("MuJoCo (xArm)")
                    plt.imshow(cv2.cvtColor(fg_image_bgr, cv2.COLOR_BGR2RGB))
                    plt.axis("off")
                    plt.subplot(1, 3, 2)
                    plt.title("Background")
                    plt.imshow(cv2.cvtColor(bg_im_np, cv2.COLOR_BGR2RGB))
                    plt.axis("off")
                    plt.subplot(1, 3, 3)
                    plt.title("Composite")
                    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()

            # (7) Step simulation with random action
            # xArm: 7 joint actuators + 1 gripper actuator = 8
            action = np.random.uniform(-1, 1, size=data.ctrl.shape)
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            count += 1
            if viewer is not None:
                viewer.sync()

    if _HAS_DISPLAY:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_simulation(viewer)
    else:
        print("[INFO] Running in headless mode (no viewer)")
        run_simulation(None)


if __name__ == "__main__":
    main()
