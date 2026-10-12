"""可视化（阶段 1 验收：带持续 ID 的叠加视频与点云输出）。

不依赖 cv2/matplotlib（当前环境不可用）：
  - 视频叠加输出为逐帧 PNG（numpy 手写 PPM/PNG 编码过重，默认输出
    ASCII 预览 + 可选 PPM 序列）；
  - 点云输出为带颜色的 XYZRGB txt/ply（ASCII ply，无外部依赖）。

真实实验阶段可替换为 cv2 视频写出（接口保持不变）。
"""
from __future__ import annotations

import os

import numpy as np

# 持续 ID 的调色板（RGB）
_PALETTE = [
    (230, 60, 60), (60, 180, 75), (70, 110, 240), (240, 180, 40),
    (160, 70, 220), (70, 220, 220), (240, 110, 180), (140, 140, 60),
    (60, 60, 140), (200, 200, 200),
]


def id_color(persistent_id: int) -> tuple:
    return _PALETTE[persistent_id % len(_PALETTE)]


def render_overlay_ascii(image: np.ndarray, observations: list, assignment: dict) -> str:
    """ASCII 预览：掩码区域用持续 ID 数字标注（无外部依赖的验收预览）。"""
    h, w = image.shape[:2]
    step = max(1, h // 16)
    labels = np.full((h, w), ".", dtype=object)
    for cand_idx, obs in enumerate(observations):
        pid = assignment.get(cand_idx)
        ch = "#" if pid is None else str(pid % 10)
        labels[obs.mask] = ch
    rows = []
    for y in range(0, h, step):
        rows.append("".join(labels[y, 0:w:1]))
    return "\n".join(rows)


def save_overlay_ppm(path: str, image: np.ndarray, observations: list, assignment: dict) -> None:
    """叠加持久 ID 颜色并保存为 PPM。"""
    out = image.astype(np.uint8).copy()
    for cand_idx, obs in enumerate(observations):
        pid = assignment.get(cand_idx)
        color = id_color(pid if pid is not None else 999)
        out[obs.mask] = color
    h, w = out.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(out.tobytes())


def save_pointcloud_ply(
    path: str,
    points: list,
    colors: list,
    persistent_ids: list,
) -> None:
    """导出带实例颜色的语义点云（ASCII ply）。

    Args:
        points: 每个元素为 (N, 3)。
        colors: 每个元素为 (N, 3) uint8。
        persistent_ids: 与 points 等长（本实现按实例整体着色，点云内不
            存 ID 列，保持最小格式）。
    """
    total = sum(p.shape[0] for p in points)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {total}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pts, cols in zip(points, colors):
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def build_semantic_pointcloud(results: list, voxel: float = 0.0):
    """从帧结果聚合语义实例点云（增量聚合的最小实现）。

    Returns:
        points, colors, persistent_ids（按实例分组）
    """
    from .memory import InstanceMemoryBank  # noqa: F401（避免循环）

    per_instance: dict[int, list] = {}
    for res in results:
        if res.geometry is None:
            continue
        pm = res.geometry.pointmap_world
        valid = res.geometry.valid
        for cand_idx, obs in enumerate(res.observations):
            pid = res.assignment.get(cand_idx)
            if pid is None:
                continue
            m = obs.mask & valid
            pts = pm[m]
            if pts.shape[0] == 0:
                continue
            color = np.tile(np.array(id_color(pid), dtype=np.uint8), (pts.shape[0], 1))
            per_instance.setdefault(pid, []).append((pts, color))

    points, colors, pids = [], [], []
    for pid, chunks in per_instance.items():
        pts = np.concatenate([c[0] for c in chunks], axis=0)
        cols = np.concatenate([c[1] for c in chunks], axis=0)
        if voxel > 0:
            # 体素下采样（观测计数简版：均匀网格取首点）
            keys = np.floor(pts / voxel).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts, cols = pts[idx], cols[idx]
        points.append(pts)
        colors.append(cols)
        pids.append(pid)
    return points, colors, pids
