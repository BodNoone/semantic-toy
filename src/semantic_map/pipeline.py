"""主流水线: 逐帧执行 重建 → 跟踪 → 双向耦合 → 建图。

每帧顺序:
1. 两个引擎各出一帧 (FrameGeometry / FrameTracking)
2. track ID 先经过 remapper 规范化 (吸收历史重关联结果)
3. 从 raw 几何提取当前帧锚点观测
4. [语义→几何] 与锚点库中的历史锚点组成对应, 求解漂移校正,
   门控通过则校正当前帧几何 (pose + pointmap)
5. 在"校正后"几何上重新提取锚点观测
6. [几何→语义] 新出生 track 与失联 track 做几何重关联, 合并 ID
7. 用校正后观测更新锚点库
8. 将校正后几何 + 规范 ID 写入语义地图
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .coupling.anchors import AnchorConfig, SemanticAnchorBank
from .coupling.drift import DriftConfig, DriftCorrector
from .coupling.reassoc import ReassocConfig, ReassocEvent, TrackIdRemapper, TrackReassociator
from .engines.base import ReconstructionEngine, TrackingEngine
from .mapping import MapConfig, SemanticMapBuilder
from .types import DriftCorrection, FrameGeometry, FrameTracking


@dataclass(frozen=True)
class PipelineConfig:
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    reassoc: ReassocConfig = field(default_factory=ReassocConfig)
    map: MapConfig = field(default_factory=MapConfig)
    enable_drift_correction: bool = True
    enable_reassociation: bool = True


@dataclass
class FrameReport:
    frame_index: int
    correction: DriftCorrection | None
    reassoc_events: list[ReassocEvent]
    visible_tracks: int
    map_points: int


class SemanticMappingPipeline:
    def __init__(
        self,
        reconstruction: ReconstructionEngine,
        tracking: TrackingEngine,
        config: PipelineConfig,
    ) -> None:
        self.reconstruction = reconstruction
        self.tracking = tracking
        self.config = config
        self.bank = SemanticAnchorBank(config.anchor)
        self.corrector = DriftCorrector(config.drift)
        self.reassociator = TrackReassociator(config.reassoc)
        self.remapper = TrackIdRemapper()
        self.map_builder = SemanticMapBuilder(config.map)
        self.reports: list[FrameReport] = []
        # 评测用: 保留 raw 与 corrected 双位姿轨迹
        self.raw_poses: list[torch.Tensor] = []
        self.corrected_poses: list[torch.Tensor] = []

    def run(self, *, max_frames: int | None = None) -> list[FrameReport]:
        processed = 0
        while max_frames is None or processed < max_frames:
            try:
                geometry = self.reconstruction.next_frame()
                tracking = self.tracking.next_frame()
            except StopIteration:
                break
            if geometry.frame_index != tracking.frame_index:
                raise RuntimeError(
                    f"engine desync: geometry frame {geometry.frame_index} vs "
                    f"tracking frame {tracking.frame_index}"
                )
            self.step(geometry, tracking)
            processed += 1
        return self.reports

    def step(self, geometry: FrameGeometry, tracking: FrameTracking) -> FrameReport:
        frame_index = geometry.frame_index
        self.raw_poses.append(geometry.world_to_camera.detach().clone())

        # (2) 规范化 track ID
        tracking = self._remap_tracking(tracking)

        # (3) raw 几何上的锚点观测
        raw_observations = self.bank.collect_observations(geometry, tracking)

        # (4) 语义→几何: 漂移校正
        correction: DriftCorrection | None = None
        if self.config.enable_drift_correction:
            source, target = self._correspondences(raw_observations)
            if source is not None:
                geometry, correction = self.corrector.correct(geometry, source, target)
            else:
                correction = DriftCorrection(applied=False, reason="no correspondences")

        self.corrected_poses.append(geometry.world_to_camera.detach().clone())

        # (5) 校正后几何上重新观测 (漂移已消除, 锚点/建图使用该观测)
        observations = (
            self.bank.collect_observations(geometry, tracking)
            if correction is not None and correction.applied
            else raw_observations
        )

        # (6) 几何→语义: 新 track 重关联
        events: list[ReassocEvent] = []
        if self.config.enable_reassociation:
            known = self.bank.known_ids()
            new_ids = {
                track_id: observation
                for track_id, observation in observations.items()
                if track_id not in known
            }
            categories = {
                int(tracking.track_ids[position]): tracking.categories[position]
                for position in range(int(tracking.masks.shape[0]))
            }
            events = self.reassociator.match_new_tracks(
                self.bank,
                new_ids,
                categories,
                frame_index=frame_index,
            )
            for event in events:
                self.remapper.merge(event.new_track_id, event.matched_track_id)
                self.bank.merge(event.new_track_id, event.matched_track_id)
            if events:
                # ID 变了, 重新规范化本帧 tracking 与观测
                tracking = self._remap_tracking(tracking)
                observations = {
                    self.remapper.canonical(track_id): observation
                    for track_id, observation in observations.items()
                }

        # (7) 更新锚点库
        self.bank.update(observations, tracking, frame_index=frame_index)

        # (8) 建图
        added = self.map_builder.add_frame(
            geometry,
            mask=tracking.masks,
            track_ids=tracking.track_ids,
            categories=tracking.categories,
        )

        report = FrameReport(
            frame_index=frame_index,
            correction=correction,
            reassoc_events=events,
            visible_tracks=int(tracking.masks.shape[0]),
            map_points=len(self.map_builder),
        )
        self.reports.append(report)
        return report

    def _remap_tracking(self, tracking: FrameTracking) -> FrameTracking:
        remapped = torch.tensor(
            [self.remapper.canonical(int(v)) for v in tracking.track_ids],
            dtype=tracking.track_ids.dtype,
        )
        if torch.equal(remapped, tracking.track_ids):
            return tracking
        return FrameTracking(
            masks=tracking.masks,
            track_ids=remapped,
            scores=tracking.scores,
            categories=tracking.categories,
            frame_index=tracking.frame_index,
        )

    def _correspondences(
        self,
        observations: dict[int, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """当前帧观测锚点 ↔ 锚点库历史锚点 的成对对应。"""

        source_list = []
        target_list = []
        for track_id, observation in observations.items():
            state = self.bank.tracks.get(track_id)
            if state is None:
                continue
            source_list.append(observation.centroid)
            target_list.append(state.anchor.centroid)
        if len(source_list) < self.config.drift.min_correspondences:
            return None, None
        return torch.stack(source_list), torch.stack(target_list)
