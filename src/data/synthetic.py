"""合成场景数据生成（阶段 2 数据）。

覆盖计划 6 中的五类场景：
  1. 平稳运动与持续可见；
  2. 离开视野后的再观测；
  3. 显著视角变化；
  4. 多个同类相似物体；
  5. 相机回环。

真值独立存放：生成函数返回的 ground_truth 与 frames 分离，
系统关联逻辑不得访问真值 ID、真值位姿或完整物体中心（计划 6）。

当前为合成数据（可控仿真），真实数据接入时保持同样的返回结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import FrameInput, SequenceMetadata


@dataclass
class SyntheticScene:
    """合成场景：帧序列 + 独立真值 + 元数据。"""

    frames: list                     # list[FrameInput]，extra 仅供 mock 适配器
    ground_truth: dict               # 真值独立存放，关联逻辑禁止访问
    metadata: SequenceMetadata
    text_queries: list = field(default_factory=list)


def _pose_from_yaw_radius(yaw: float, radius: float, height: float = 1.5) -> np.ndarray:
    """相机沿圆周轨迹、看向圆心的位姿（T_world_from_camera）。"""
    cam = np.array([radius * np.cos(yaw), radius * np.sin(yaw), height])
    target = np.array([0.0, 0.0, 1.0])
    forward = target - cam
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=1)  # 相机系: x右, y下, z前
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cam
    return T


def _render_object_instance(
    h: int,
    w: int,
    center_px: tuple,
    size_px: tuple,
    color: tuple,
) -> np.ndarray:
    """渲染一个矩形物体掩码（合成）。"""
    cx, cy = center_px
    sw, sh = size_px
    mask = np.zeros((h, w), dtype=bool)
    y0, y1 = max(0, cy - sh // 2), min(h, cy + sh // 2)
    x0, x1 = max(0, cx - sw // 2), min(w, cx + sw // 2)
    mask[y0:y1, x0:x1] = True
    return mask


def generate_orbit_scene(
    num_frames: int = 40,
    image_size: tuple = (96, 128),
    num_chairs: int = 4,
    leave_and_reenter: bool = True,
    seed: int = 0,
) -> SyntheticScene:
    """环视场景：相机绕原点环视，多个同类物体（椅子）分布周围。

    覆盖场景 1/3/4/5：平稳运动、视角变化、多同类物体、回环
    （轨迹闭合一周）。
    leave_and_reenter=True 时物体在部分帧离开视野（场景 2）。
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    text_queries = ["chair"]

    # 物体世界位置（完整物体中心，仅存于真值，系统不可访问）
    object_centers_world = []
    for i in range(num_chairs):
        ang = 2 * np.pi * i / num_chairs + 0.3
        r = 3.0 + 0.5 * rng.random()
        object_centers_world.append(
            np.array([r * np.cos(ang), r * np.sin(ang), 0.6])
        )

    intrinsics = np.array(
        [[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]]
    )

    frames = []
    gt = {
        "object_centers": np.stack(object_centers_world),
        "visible_ids": [],           # 每帧可见的真值物体 ID
        "poses": [],
        "pointmaps": [],
        "masks": [],                 # 每帧每个可见物体的掩码（按真值ID顺序）
    }

    for t in range(num_frames):
        yaw = 2 * np.pi * t / num_frames          # 一周闭合 -> 回环
        pose = _pose_from_yaw_radius(yaw, radius=5.0)
        gt["poses"].append(pose)

        image = np.full((h, w, 3), 40, dtype=np.uint8)
        pointmap = np.zeros((h, w, 3), dtype=np.float32)
        valid = np.zeros((h, w), dtype=bool)
        R = pose[:3, :3]
        tvec = pose[:3, 3]

        vis_ids = []
        masks_this = []
        for oid, center in enumerate(object_centers_world):
            # 场景 2：部分帧该物体被遮挡/离开视野
            if leave_and_reenter and (0.3 * num_frames <= t < 0.5 * num_frames) and oid == 0:
                continue
            # 世界 -> 相机
            x_cam = R.T @ (center - tvec)
            if x_cam[2] <= 0.2:
                continue
            u = intrinsics[0, 0] * x_cam[0] / x_cam[2] + intrinsics[0, 2]
            v = intrinsics[1, 1] * x_cam[1] / x_cam[2] + intrinsics[1, 2]
            if not (0 <= u < w and 0 <= v < h):
                continue
            size = max(6, int(30.0 / x_cam[2]))
            mask = _render_object_instance(h, w, (int(u), int(v)), (size, size), (200, 60, 60))
            if mask.sum() == 0:
                continue
            vis_ids.append(oid)
            masks_this.append(mask)
            ys, xs = np.nonzero(mask)
            # 点图：掩码区域填该物体表面点（相机前表面，近似平面）
            depth = x_cam[2]
            xs_n = (xs - intrinsics[0, 2]) / intrinsics[0, 0] * depth
            ys_n = (ys - intrinsics[1, 2]) / intrinsics[1, 1] * depth
            pts_cam = np.stack([xs_n, ys_n, np.full_like(xs_n, depth)], axis=1)
            pts_world = (R @ pts_cam.T).T + tvec
            pointmap[ys, xs] = pts_world
            valid[ys, xs] = True
            color = np.array([180 + 20 * (oid % 2), 60, 60], dtype=np.uint8)
            image[ys, xs] = color

        gt["visible_ids"].append(vis_ids)
        gt["masks"].append(masks_this)
        gt["pointmaps"].append(pointmap.copy())

        # mock 适配器输入（标注通道）：外观特征与真值身份弱相关 + 视角相关扰动
        instances = []
        for oid, mask in zip(vis_ids, masks_this):
            base_app = np.zeros(8, dtype=np.float32)
            base_app[oid % 8] = 1.0                    # 身份信号（同类物体相似）
            base_app += rng.normal(0, 0.05, size=8)    # 外观噪声
            base_app /= np.linalg.norm(base_app)
            instances.append(
                {
                    "category": "chair",
                    "mask": mask,
                    "score": 0.9,
                    "appearance": base_app,
                }
            )

        frames.append(
            FrameInput(
                image=image,
                frame_index=t,
                timestamp=float(t),
                text_queries=text_queries,
                intrinsics=intrinsics,
                extra={
                    "pose_w_from_c": pose,
                    "pointmap_world": pointmap,
                    "valid_mask": valid,
                    "instances": instances,
                },
            )
        )

    metadata = SequenceMetadata(
        sequence_id="synthetic_orbit",
        image_width=w,
        image_height=h,
        scale_convention="metric_synthetic",
        geometry_model="mock(synthetic)",
        semantic_model="mock(synthetic)",
        config={"num_frames": num_frames, "num_chairs": num_chairs, "seed": seed},
    )
    return SyntheticScene(frames=frames, ground_truth=gt, metadata=metadata,
                          text_queries=text_queries)
