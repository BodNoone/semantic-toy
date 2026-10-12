"""实例观测提取（EXECUTION_PLAN.md 5.2）。

对每个候选掩码计算：有效三维点、置信度加权表面中心、空间散布（表面点云
协方差）、有效点比例、区域外观特征、分割质量、面积和边界截断情况。

约定：
  - 没有有效三维点的候选仍可参与外观关联，不得产生 NaN 或伪造空间位置；
  - 表面中心是当前可见表面中心，不等于完整物体中心。
"""
from __future__ import annotations

import numpy as np

from .contracts import GeometryOutput, InstanceObservation, SemanticCandidate


def _border_truncated(mask: np.ndarray) -> bool:
    """掩码是否触及图像边界（边界截断情况）。"""
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    return bool(
        mask[0, :].any() or mask[-1, :].any()
        or mask[:, 0].any() or mask[:, -1].any()
    )


def extract_observation(
    candidate: SemanticCandidate,
    candidate_index: int,
    geometry: GeometryOutput,
    min_valid_points: int = 8,
) -> InstanceObservation:
    """从候选掩码与几何输出提取单个实例观测。"""
    mask = candidate.mask.astype(bool)
    area = int(mask.sum())

    valid = geometry.valid & mask
    # 零坐标不充当有效观测：显式剔除全零（或极近零）点
    pts = geometry.pointmap_world[valid]
    nonzero = np.linalg.norm(pts, axis=1) > 1e-9
    valid_idx = np.flatnonzero(valid.reshape(-1))[nonzero] if pts.size else np.array([], dtype=int)
    valid_ratio = float(valid_idx.size) / float(area) if area > 0 else 0.0

    surface_center = None
    spatial_spread = None
    has_valid_geometry = valid_idx.size >= min_valid_points

    if has_valid_geometry:
        w = geometry.confidence.reshape(-1)[valid_idx].astype(np.float64)
        w = np.clip(w, 1e-6, None)
        w /= w.sum()
        xyz = geometry.pointmap_world.reshape(-1, 3)[valid_idx].astype(np.float64)
        # 置信度加权表面中心
        surface_center = (xyz * w[:, None]).sum(axis=0)
        # 空间散布：置信度加权协方差
        diff = xyz - surface_center
        spatial_spread = (diff * w[:, None]).T @ diff
        spatial_spread = spatial_spread.astype(np.float32)
        surface_center = surface_center.astype(np.float32)

    # 分割质量：类别分数与有效点比例的联合（保守乘积）
    quality = float(candidate.category_score * max(valid_ratio, 0.0))

    return InstanceObservation(
        candidate_index=candidate_index,
        mask=mask,
        category=candidate.category,
        category_score=float(candidate.category_score),
        appearance=np.asarray(candidate.appearance, dtype=np.float32),
        has_valid_geometry=has_valid_geometry,
        surface_center=surface_center,
        spatial_spread=spatial_spread,
        quality=quality,
        valid_point_ratio=valid_ratio,
        area=area,
        border_truncated=_border_truncated(mask),
    )


def extract_observations(
    semantic_output,
    geometry: GeometryOutput,
    min_valid_points: int = 8,
) -> list:
    """对一帧的全部候选提取观测。空候选列表合法返回空列表（空观测覆盖）。"""
    observations = []
    for idx, cand in enumerate(semantic_output.candidates):
        observations.append(
            extract_observation(cand, idx, geometry, min_valid_points)
        )
    return observations
