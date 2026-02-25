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
T_splat2mj = np.load(ICP_TRANSFORM_PATH)

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

def get_robot_geom_ids(model):
    """
    Returns geom IDs belonging to the xArm robot (all mesh geoms) and the cube.
    Excludes floor and primitive geoms (cylinder base, etc.).
    """
    robot_geom_ids = set()
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            robot_geom_ids.add(geom_id)
    # Also include the cube as foreground
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    if cube_id != -1:
        robot_geom_ids.add(cube_id)
    return robot_geom_ids


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
    Allows mouse drag/rotate to change camera view and re-renders in real-time.
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
        self.mouse_dragging = False
        self.last_mouse_pos = None

        self.window_name = "Interactive Composite View (Drag to rotate, Right-drag to pan, Scroll to zoom)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._render_and_display()

    def _get_initial_camera_pose(self):
        """Get camera pose from calibration (already set on data)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        camera_xpos = self.data.cam_xpos[cam_id]
        camera_xmat = self.data.cam_xmat[cam_id].reshape(3, 3)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_xmat
        camera_pose[:3, 3] = camera_xpos
        return camera_pose

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_dragging = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
            self.last_mouse_pos = None
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging:
            if self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self._rotate_camera(dx, dy)
                self.last_mouse_pos = (x, y)
                self._render_and_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_dragging = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.mouse_dragging = False
            self.last_mouse_pos = None
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging and flags & cv2.EVENT_FLAG_RBUTTON:
            if self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self._pan_camera(dx, dy)
                self.last_mouse_pos = (x, y)
                self._render_and_display()
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16)
            if delta > 32767:
                delta -= 65536
            zoom_factor = 1.0 + np.clip((delta / 120.0) * 0.02, -0.1, 0.1)
            self._zoom_camera(zoom_factor)
            self._render_and_display()

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

    def _zoom_camera(self, zoom_factor):
        forward = -self.camera_pose[:3, 2]
        zoom_speed = 0.05
        translation = forward * (zoom_factor - 1.0) * zoom_speed
        self.camera_pose[:3, 3] += translation

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
        seg_labels[seg_labels == -1] = 0

        self.data.cam_xpos[cam_id] = original_pos
        self.data.cam_xmat[cam_id] = original_mat

        return fg_bgr, seg_labels

    def _render_and_display(self):
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

        bg_im, depth, sil = render(w2c_bg, self.k, self.scene_data, self.scene_depth_data, self.viz_cfg)

        bg_im_np = bg_im.permute(1, 2, 0).cpu().numpy()
        bg_im_np = (bg_im_np * 255).astype(np.uint8)
        bg_im_np = cv2.cvtColor(bg_im_np, cv2.COLOR_RGB2BGR)

        composite = bg_im_np.copy()
        composite[mask_uint8 > 0] = foreground_masked[mask_uint8 > 0]

        h, w = composite.shape[:2]
        display_height = 480
        display_width = int(w * display_height / h)

        fg_resized = cv2.resize(fg_image_bgr, (display_width, display_height))
        bg_resized = cv2.resize(bg_im_np, (display_width, display_height))
        comp_resized = cv2.resize(composite, (display_width, display_height))

        side_by_side = np.hstack([fg_resized, bg_resized, comp_resized])
        cv2.putText(side_by_side, "MuJoCo Foreground", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Gaussian Background", (display_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Composite", (2 * display_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(self.window_name, side_by_side)

    def run(self):
        print("\n=== Interactive Composite Viewer (xArm) ===")
        print("Controls:")
        print("  - Left mouse drag: Rotate camera")
        print("  - Right mouse drag: Pan camera")
        print("  - Mouse wheel: Zoom in/out")
        print("  - Press 'q' to quit")
        print("  - Press 'r' to reset camera")
        print("  - Press 's' to save current view")
        print("=============================================\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.camera_pose = self._get_initial_camera_pose()
                self._render_and_display()
            elif key == ord('s'):
                fg_image_bgr, seg_labels = self._render_mujoco_rgb_and_geomseg_from_pose(self.camera_pose)
                robot_mask_binary = np.isin(seg_labels, list(self.target_geom_ids))
                mask_uint8 = (robot_mask_binary.astype(np.uint8)) * 255
                foreground_masked = cv2.bitwise_and(fg_image_bgr, fg_image_bgr, mask=mask_uint8)

                P_mj_cam = self.camera_pose.copy()
                T_mj2splat = np.linalg.inv(self.T_splat2mj)
                P_gs_cam = T_mj2splat @ P_mj_cam
                w2c_bg = P_gs_cam
                transform_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                w2c_bg = w2c_bg @ transform_matrix
                w2c_bg = np.linalg.inv(w2c_bg)
                bg_im, depth, sil = render(w2c_bg, self.k, self.scene_data, self.scene_depth_data, self.viz_cfg)
                bg_im_np = bg_im.permute(1, 2, 0).cpu().numpy()
                bg_im_np = (bg_im_np * 255).astype(np.uint8)
                bg_im_np = cv2.cvtColor(bg_im_np, cv2.COLOR_RGB2BGR)
                composite = bg_im_np.copy()
                composite[mask_uint8 > 0] = foreground_masked[mask_uint8 > 0]

                os.makedirs("images", exist_ok=True)
                timestamp = int(time.time())
                cv2.imwrite(f"images/xarm_fg_{timestamp}.png", fg_image_bgr)
                cv2.imwrite(f"images/xarm_bg_{timestamp}.png", bg_im_np)
                cv2.imwrite(f"images/xarm_composite_{timestamp}.png", composite)
                print(f"Saved: xarm_fg_{timestamp}.png, xarm_bg_{timestamp}.png, xarm_composite_{timestamp}.png")

        cv2.destroyAllWindows()


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
    scene_path = "pointclouds/xarm7.npz"
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

    # Use real RealSense D455 intrinsics (serial 246322303954)
    k = REALSENSE_INTRINSICS_640x480.copy()
    print(f"Using RealSense D455 intrinsics (serial 246322303954):")
    print(f"  fx={k[0,0]:.3f}, fy={k[1,1]:.3f}, cx={k[0,2]:.3f}, cy={k[1,2]:.3f}")

    w2c_bg_init, _ = load_camera(viz_cfg, scene_path)

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

            # (2) Get segmentation mask and extract foreground (robot)
            seg_renderer.update_scene(data, camera=cam_name)
            seg_mask = seg_renderer.render()
            seg_labels = seg_mask[:, :, 0]
            seg_labels[seg_labels == -1] = 0
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
