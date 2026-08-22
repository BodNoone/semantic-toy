"""语义锚点库: 每个持久 track 的跨帧 3D 锚点。

锚点 = mask 内高置信 3D 点的鲁棒质心(逐维中位数) + 鲁棒尺度。
锚点用 EMA 在"已校正"坐标系中更新, 作为后续帧漂移估计的参考。

这是"语义→几何"约束的物质基础: 同一 track ID 在不同帧观测到的
锚点理应重合, 偏移即漂移信号。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import AnchorObservation, FrameGeometry, FrameTracking, TrackState


@dataclass(frozen=True)
class AnchorConfig:
    min_points: int = 64  # 低于此点数不生成观测
    confidence_threshold: float = 0.3
    ema_momentum: float = 0.8  # 历史锚点权重
    max_scale_ratio: float = 4.0  # 观测尺度与历史尺度比过大则拒绝更新


def observe_anchor(
    geometry: FrameGeometry,
    mask: torch.Tensor,
    *,
    config: AnchorConfig,
) -> AnchorObservation | None:
    """从单帧 mask + 几何提取一个 3D 锚点观测。

    返回 None 表示有效点不足, 无法形成可靠观测。
    """

    if mask.shape != geometry.confidence.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != confidence shape "
            f"{tuple(geometry.confidence.shape)}"
        )
    valid = (
        mask.bool()
        & (geometry.confidence >= config.confidence_threshold)
        & torch.isfinite(geometry.world_points).all(dim=-1)
    )
    count = int(valid.sum())
    if count < config.min_points:
        return None
    points = geometry.world_points[valid]  # [N,3]
    centroid = points.median(dim=0).values
    # 鲁棒尺度: 到质心距离的中位数
    scale = float((points - centroid).norm(dim=-1).median())
    return AnchorObservation(
        centroid=centroid.detach().clone(),
        scale=max(scale, 1e-6),
        count=count,
    )


class SemanticAnchorBank:
    """维护 track_id → TrackState 的持久注册表。"""

    def __init__(self, config: AnchorConfig) -> None:
        self.config = config
        self.tracks: dict[int, TrackState] = {}

    def known_ids(self) -> set[int]:
        return set(self.tracks.keys())

    def lost_tracks(self, frame_index: int, *, min_lost_frames: int) -> list[TrackState]:
        """返回已失联超过 min_lost_frames 帧的 track。"""

        return [
            state
            for state in self.tracks.values()
            if frame_index - state.last_seen >= min_lost_frames
        ]

    def collect_observations(
        self,
        geometry: FrameGeometry,
        tracking: FrameTracking,
    ) -> dict[int, AnchorObservation]:
        """提取当前帧每个可见 track 的锚点观测 (不更新状态)。"""

        observations: dict[int, AnchorObservation] = {}
        for position in range(int(tracking.masks.shape[0])):
            track_id = int(tracking.track_ids[position])
            observation = observe_anchor(
                geometry,
                tracking.masks[position],
                config=self.config,
            )
            if observation is not None:
                observations[track_id] = observation
        return observations

    def update(
        self,
        observations: dict[int, AnchorObservation],
        tracking: FrameTracking,
        *,
        frame_index: int,
    ) -> None:
        """用(已校正坐标系下的)观测 EMA 更新锚点; 新 track 直接注册。"""

        id_to_category = {
            int(tracking.track_ids[position]): tracking.categories[position]
            for position in range(int(tracking.masks.shape[0]))
        }
        for track_id, observation in observations.items():
            state = self.tracks.get(track_id)
            if state is None:
                self.tracks[track_id] = TrackState(
                    track_id=track_id,
                    category=id_to_category.get(track_id, "object"),
                    anchor=observation,
                    last_seen=frame_index,
                )
                continue
            ratio = observation.scale / max(state.anchor.scale, 1e-6)
            if ratio > self.config.max_scale_ratio or ratio < 1.0 / self.config.max_scale_ratio:
                # 尺度突变的观测不可信 (可能 mask 错乱), 只刷新可见时间
                state.last_seen = frame_index
                continue
            momentum = self.config.ema_momentum
            blended = momentum * state.anchor.centroid + (1.0 - momentum) * observation.centroid
            blended_scale = momentum * state.anchor.scale + (1.0 - momentum) * observation.scale
            state.anchor = AnchorObservation(
                centroid=blended,
                scale=blended_scale,
                count=observation.count,
            )
            state.last_seen = frame_index

    def merge(self, source_id: int, target_id: int) -> None:
        """将 source track 合并进 target (重关联成功后调用)。"""

        if source_id == target_id or source_id not in self.tracks:
            return
        source = self.tracks.pop(source_id)
        target = self.tracks.get(target_id)
        if target is None:
            source.track_id = target_id
            self.tracks[target_id] = source
            return
        target.last_seen = max(target.last_seen, source.last_seen)
        target.point_count += source.point_count
