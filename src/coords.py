"""坐标与预处理约定（EXECUTION_PLAN.md 第 3 节“坐标和预处理约定”）。

统一规则：
  - 位姿为 T_world_from_camera，满足齐次坐标关系
        X_world = T_world_from_camera @ X_camera
  - 世界坐标系由序列起始帧固定。
  - 点图、掩码与图像必须具有明确的像素对应关系。
  - 内参随 resize / crop / padding 同步调整。
"""
from __future__ import annotations

import numpy as np


def make_identity_pose() -> np.ndarray:
    """起始帧位姿：单位阵（世界系 = 第 0 帧相机系）。"""
    return np.eye(4, dtype=np.float64)


def transform_points(pose_w_from_c: np.ndarray, points_camera: np.ndarray) -> np.ndarray:
    """将相机系点变换到世界系。

    Args:
        pose_w_from_c: (4, 4) T_world_from_camera。
        points_camera: (N, 3)。
    Returns:
        (N, 3) 世界系点。
    """
    assert pose_w_from_c.shape == (4, 4), "pose must be 4x4"
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (pose_w_from_c @ homogeneous.T).T[:, :3]


def invert_pose(pose_w_from_c: np.ndarray) -> np.ndarray:
    """SE(3) 逆：返回 T_camera_from_world。"""
    R = pose_w_from_c[:3, :3]
    t = pose_w_from_c[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose_poses(pose_a_from_b: np.ndarray, pose_b_from_c: np.ndarray) -> np.ndarray:
    """位姿复合：T_a_from_c = T_a_from_b @ T_b_from_c。"""
    return pose_a_from_b @ pose_b_from_c


def camera_center(pose_w_from_c: np.ndarray) -> np.ndarray:
    """相机光心在世界系中的坐标。"""
    return pose_w_from_c[:3, 3].copy()


def adjust_intrinsics_for_resize(
    intrinsics: np.ndarray, old_size: tuple, new_size: tuple
) -> np.ndarray:
    """resize 变换下的内参同步调整。

    Args:
        intrinsics: (3, 3)。
        old_size: (width, height)。
        new_size: (width, height)。
    """
    ow, oh = old_size
    nw, nh = new_size
    K = np.array(intrinsics, dtype=np.float64, copy=True)
    K[0, :] *= nw / ow
    K[1, :] *= nh / oh
    return K


def adjust_intrinsics_for_crop(intrinsics: np.ndarray, crop_xy: tuple) -> np.ndarray:
    """crop 变换下的内参同步调整（左上角偏移）。"""
    K = np.array(intrinsics, dtype=np.float64, copy=True)
    K[0, 2] -= crop_xy[0]
    K[1, 2] -= crop_xy[1]
    return K


def adjust_intrinsics_for_padding(intrinsics: np.ndarray, pad_xy: tuple) -> np.ndarray:
    """padding 变换下的内参同步调整（内容整体偏移）。"""
    K = np.array(intrinsics, dtype=np.float64, copy=True)
    K[0, 2] += pad_xy[0]
    K[1, 2] += pad_xy[1]
    return K


def check_mask_image_alignment(mask: np.ndarray, image: np.ndarray) -> bool:
    """掩码与图像必须具有明确的像素对应关系（空间形状一致）。"""
    return mask.shape[:2] == image.shape[:2]
