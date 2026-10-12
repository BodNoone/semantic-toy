"""最小实例记忆（EXECUTION_PLAN.md 5.4）。

状态：
  - active：近期可见；
  - dormant：暂不可见，仍保留身份。

规则：
  - 新建实例与保守更新；
  - 低质量匹配不得无条件覆盖历史空间状态；
  - 暂不实现自动身份合并、复杂归档和多假设搜索。
"""
from __future__ import annotations

import numpy as np

from .contracts import InstanceMemory, InstanceObservation, InstanceStatus


class InstanceMemoryBank:
    """因果维护的实例记忆库。"""

    def __init__(
        self,
        dormant_grace_frames: int = 30,
        appearance_momentum: float = 0.8,
        spatial_momentum: float = 0.85,
        min_quality_for_spatial_update: float = 0.15,
    ):
        self.dormant_grace_frames = dormant_grace_frames
        self.appearance_momentum = appearance_momentum
        self.spatial_momentum = spatial_momentum
        self.min_quality_for_spatial_update = min_quality_for_spatial_update
        self._memories: dict[int, InstanceMemory] = {}
        self._next_id = 0

    def reset(self) -> None:
        """每段序列开始时重置全部状态。"""
        self._memories = {}
        self._next_id = 0

    def get(self, persistent_id: int) -> InstanceMemory:
        return self._memories[persistent_id]

    def all_memories(self) -> list:
        return list(self._memories.values())

    def create_instance(self, observation: InstanceObservation, frame_index: int) -> int:
        """新建实例：从观测初始化记忆条目。"""
        pid = self._next_id
        self._next_id += 1
        mem = InstanceMemory(
            persistent_id=pid,
            category=observation.category,
            status=InstanceStatus.ACTIVE,
            last_seen_frame=frame_index,
            observation_count=1,
            appearance=np.array(observation.appearance, dtype=np.float32, copy=True),
        )
        self._maybe_update_spatial(mem, observation, frame_index, force=True)
        self._memories[pid] = mem
        return pid

    def update_instance(
        self, persistent_id: int, observation: InstanceObservation, frame_index: int
    ) -> None:
        """匹配成功后保守更新记忆。"""
        mem = self._memories[persistent_id]
        mem.status = InstanceStatus.ACTIVE
        mem.last_seen_frame = frame_index
        mem.observation_count += 1
        # 外观指数滑动平均
        m = self.appearance_momentum
        if mem.appearance is None:
            mem.appearance = np.array(observation.appearance, dtype=np.float32, copy=True)
        else:
            mem.appearance = (
                m * mem.appearance + (1 - m) * observation.appearance.astype(np.float32)
            )
            norm = np.linalg.norm(mem.appearance)
            if norm > 1e-8:
                mem.appearance /= norm
        # 低质量匹配不得无条件覆盖历史空间状态
        self._maybe_update_spatial(mem, observation, frame_index, force=False)

    def _maybe_update_spatial(
        self,
        mem: InstanceMemory,
        observation: InstanceObservation,
        frame_index: int,
        force: bool,
    ) -> None:
        if not observation.has_valid_geometry:
            return
        if not force and observation.quality < self.min_quality_for_spatial_update:
            return
        center = observation.surface_center
        spread = observation.spatial_spread
        if not mem.has_spatial_state or mem.spatial_center is None:
            mem.spatial_center = np.array(center, dtype=np.float32)
            mem.spatial_covariance = np.array(spread, dtype=np.float32)
            mem.has_spatial_state = True
            mem.last_spatial_update_frame = frame_index
            return
        m = self.spatial_momentum
        mem.spatial_center = m * mem.spatial_center + (1 - m) * center
        mem.spatial_covariance = m * mem.spatial_covariance + (1 - m) * spread
        mem.last_spatial_update_frame = frame_index

    def advance_frame(self, frame_index: int) -> None:
        """每帧结束时推进状态：超过宽限期的 active 转 dormant。

        dormant 实例不删除（暂不可见不会立即删除实例）。
        """
        for mem in self._memories.values():
            if mem.status == InstanceStatus.ACTIVE:
                if frame_index - mem.last_seen_frame >= self.dormant_grace_frames:
                    mem.status = InstanceStatus.DORMANT

    def count(self, status: InstanceStatus | None = None) -> int:
        if status is None:
            return len(self._memories)
        return sum(1 for m in self._memories.values() if m.status == status)
