"""漂移校正器: 用语义锚点对应抑制流式重建的累积漂移。

思路: 同一 track 的历史锚点(已校正坐标系)与当前帧观测锚点应重合,
两者间的最佳刚性变换即漂移估计。求解鲁棒 Sim(3) (Umeyama), 门控通过后
将校正应用到当前帧 pose 与 pointmap。

门控是"protected fallback"思想的体现: 对应不足/残差过大时返回
applied=False, 调用方保持 raw 输出, 绝不劣化。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import DriftCorrection, FrameGeometry


@dataclass(frozen=True)
class DriftConfig:
    min_correspondences: int = 3  # 最少锚点对应数 (>=3 才能唯一解 Sim3)
    max_residual: float = 0.5  # 中位残差上限 (米), 超限拒绝
    allow_scale: bool = False  # 第一版默认刚性 (尺度交给重建本身)
    ransac_iters: int = 64
    ransac_sample: int = 3


def solve_similarity(
    source: torch.Tensor,  # [N,3] 观测锚点 (当前帧)
    target: torch.Tensor,  # [N,3] 历史锚点 (参考系)
    *,
    allow_scale: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Umeyama 最小二乘 Sim(3)/SE(3)。source → target。

    返回 (scale, rotation, translation), 其中 rotation 为列向量约定旋转,
    满足行向量式 ``target ≈ scale * source @ rotation.T + translation``。
    """

    n = source.shape[0]
    src_mean = source.mean(dim=0)
    tgt_mean = target.mean(dim=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean

    # Kabsch: H = src_c^T @ tgt_c, SVD H = U Σ V^T, R = V S U^T (列向量 source→target)
    h = src_c.t() @ tgt_c / n  # [3,3]
    u, singular_values, vt = torch.linalg.svd(h)
    sign = torch.ones(3, device=source.device, dtype=source.dtype)
    if torch.linalg.det(u) * torch.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = vt.t() @ torch.diag(sign) @ u.t()

    scale = 1.0
    if allow_scale:
        src_var = (src_c ** 2).sum() / n
        if src_var > 1e-12:
            scale = float((singular_values * sign).sum() / src_var)

    translation = tgt_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def _transform(points: torch.Tensor, scale: float, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return scale * (points @ rotation.t()) + translation


def estimate_drift(
    source: torch.Tensor,  # [N,3] 当前帧观测
    target: torch.Tensor,  # [N,3] 历史参考
    *,
    config: DriftConfig,
) -> DriftCorrection:
    n = source.shape[0]
    if n < config.min_correspondences:
        return DriftCorrection(applied=False, inliers=0, reason=f"correspondences={n} < {config.min_correspondences}")

    device = source.device
    dtype = source.dtype
    best: DriftCorrection | None = None
    rng = torch.Generator(device="cpu").manual_seed(0)

    for _ in range(max(1, config.ransac_iters)):
        sample_count = min(config.ransac_sample, n)
        idx = torch.randperm(n, generator=rng, device="cpu")[:sample_count]
        scale, rotation, translation = solve_similarity(
            source[idx], target[idx], allow_scale=config.allow_scale
        )
        residual = (_transform(source, scale, rotation, translation) - target).norm(dim=-1)
        median = float(residual.median())
        inliers = int((residual <= 2.0 * median).sum())
        if best is None or inliers > best.inliers:
            best = DriftCorrection(
                applied=False,
                scale=scale,
                rotation=rotation,
                translation=translation,
                inliers=inliers,
                residual=median,
            )

    assert best is not None
    if best.inliers < config.min_correspondences:
        return DriftCorrection(applied=False, inliers=best.inliers, reason="inliers too few")
    if best.residual > config.max_residual:
        return DriftCorrection(applied=False, inliers=best.inliers, residual=best.residual, reason="residual too large")

    return DriftCorrection(
        applied=True,
        scale=best.scale,
        rotation=best.rotation,
        translation=best.translation,
        inliers=best.inliers,
        residual=best.residual,
    )


def apply_correction(
    geometry: FrameGeometry,
    correction: DriftCorrection,
) -> FrameGeometry:
    """将校正作用到几何帧: pointmap 变换, pose 左乘校正。"""

    points = geometry.world_points
    corrected_points = _transform(
        points.reshape(-1, 3),
        correction.scale,
        correction.rotation,
        correction.translation,
    ).reshape_as(points)

    # world_to_camera 左乘: new_w2c = w2c @ inv(correction)
    rot = correction.rotation
    inv_rot = rot.t()
    t = correction.translation
    inv_t = -(inv_rot @ t) / correction.scale
    correction_mat = torch.eye(4, dtype=points.dtype, device=points.device)
    correction_mat[:3, :3] = inv_rot / correction.scale
    correction_mat[:3, 3] = inv_t
    corrected_w2c = geometry.world_to_camera @ correction_mat

    return FrameGeometry(
        world_points=corrected_points,
        confidence=geometry.confidence,
        world_to_camera=corrected_w2c,
        intrinsics=geometry.intrinsics,
        frame_index=geometry.frame_index,
    )


class DriftCorrector:
    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    def correct(
        self,
        geometry: FrameGeometry,
        source: torch.Tensor,  # [N,3] 当前帧观测锚点
        target: torch.Tensor,  # [N,3] 历史参考锚点 (同序对应)
    ) -> tuple[FrameGeometry, DriftCorrection]:
        correction = estimate_drift(source, target, config=self.config)
        if correction.applied:
            return apply_correction(geometry, correction), correction
        return geometry, correction
