"""几何辅助 track 重关联: 用几何一致性修复跟踪 ID 错乱。

场景: SAM 在物体长时间丢失/大视角变化后重新检出该物体, 但分配了新 ID。
判据: 新 track 的首个 3D 锚点观测若与某个"失联"同类 track 的历史锚点
在几何上足够接近 (距离/尺度双门限), 则认为是同一实例, 合并 ID。

坐标前提: 锚点均位于已漂移校正的世界坐标系, 因此"接近"有意义——
这正是"几何→语义"方向的耦合。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import AnchorObservation, TrackState
from .anchors import SemanticAnchorBank


@dataclass(frozen=True)
class ReassocConfig:
    min_lost_frames: int = 3  # 失联多少帧后才允许被重关联
    max_center_distance_ratio: float = 1.5  # 距离阈值 = ratio * max(两侧尺度)
    max_scale_ratio: float = 2.0  # 尺度比门限
    require_same_category: bool = True


@dataclass(frozen=True)
class ReassocEvent:
    new_track_id: int
    matched_track_id: int
    distance: float
    frame_index: int


class TrackReassociator:
    def __init__(self, config: ReassocConfig) -> None:
        self.config = config

    def match_new_tracks(
        self,
        bank: SemanticAnchorBank,
        new_observations: dict[int, AnchorObservation],
        new_categories: dict[int, str],
        *,
        frame_index: int,
    ) -> list[ReassocEvent]:
        """对每个新出生 track 尝试与失联 track 匹配。

        new_observations: 本帧首次出现的 track_id → 锚点观测
        返回合并事件列表; 调用方需据此更新 bank 与 ID 映射。
        """

        events: list[ReassocEvent] = []
        lost = bank.lost_tracks(frame_index, min_lost_frames=self.config.min_lost_frames)
        claimed: set[int] = set()
        for new_id, observation in sorted(new_observations.items()):
            best: tuple[float, TrackState] | None = None
            for state in lost:
                if state.track_id in claimed:
                    continue
                if (
                    self.config.require_same_category
                    and state.category != new_categories.get(new_id, "object")
                ):
                    continue
                ratio = observation.scale / max(state.anchor.scale, 1e-6)
                if ratio > self.config.max_scale_ratio or ratio < 1.0 / self.config.max_scale_ratio:
                    continue
                distance = float(
                    (observation.centroid - state.anchor.centroid).norm()
                )
                threshold = self.config.max_center_distance_ratio * max(
                    observation.scale, state.anchor.scale
                )
                if distance > threshold:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, state)
            if best is not None:
                distance, state = best
                claimed.add(state.track_id)
                events.append(
                    ReassocEvent(
                        new_track_id=new_id,
                        matched_track_id=state.track_id,
                        distance=distance,
                        frame_index=frame_index,
                    )
                )
        return events


class TrackIdRemapper:
    """维护 SAM 原始 ID → 规范 ID 的映射 (重关联合并的结果)。"""

    def __init__(self) -> None:
        self._mapping: dict[int, int] = {}

    def canonical(self, track_id: int) -> int:
        seen: set[int] = set()
        current = int(track_id)
        while current in self._mapping and current not in seen:
            seen.add(current)
            current = self._mapping[current]
        return current

    def merge(self, source_id: int, target_id: int) -> None:
        source = self.canonical(source_id)
        target = self.canonical(target_id)
        if source != target:
            self._mapping[source] = target
