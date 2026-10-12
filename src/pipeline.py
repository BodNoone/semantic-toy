"""主流程（EXECUTION_PLAN.md 5.2–5.4 与阶段 3 在线主流程）。

两种运行形态共享同一关联核心：
  - 缓存模式（阶段 1 验收）：同一缓存可运行三种关联模式；
  - 在线模式（阶段 3）：当前帧 -> 因果几何预测与文本条件分割 ->
    实例观测提取 -> 历史实例关联 -> 持续 ID -> 更新记忆。

在线模式当前未实现真实模型接入，骨架保留；缓存模式为第一版交付重点。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .association import Associator, AssociationResult
from .contracts import (
    AssociationMode,
    FrameInput,
    FrameResult,
    GeometryOutput,
    InstanceObservation,
    SemanticCandidate,
    SemanticOutput,
)
from .memory import InstanceMemoryBank
from .observations import extract_observations


def associate_frame(
    observations: list,
    bank: InstanceMemoryBank,
    associator: Associator,
    frame_index: int,
    timestamp: float = 0.0,
    geometry: Optional[GeometryOutput] = None,
    semantic: Optional[SemanticOutput] = None,
) -> FrameResult:
    """单帧关联核心：候选观测 -> 持续 ID -> 记忆更新。

    该函数不读取未来帧，不访问真值（关联逻辑与真值隔离）。
    """
    memories = bank.all_memories()
    assoc: AssociationResult = associator.associate(observations, memories)

    # 新建实例：assignment 中的 -1 占位替换为真实持久 ID
    final_assignment = {}
    for cand_idx, pid in assoc.assignment.items():
        if pid == -1:
            pid = bank.create_instance(observations[cand_idx], frame_index)
        final_assignment[cand_idx] = pid

    # 匹配成功的实例保守更新记忆
    for cand_idx, pid in final_assignment.items():
        mem = bank.get(pid)
        if mem.observation_count > 0 and bank.get(pid).last_seen_frame == frame_index:
            continue  # 刚由 create 初始化
        bank.update_instance(pid, observations[cand_idx], frame_index)

    # 帧末推进状态：超时 active -> dormant
    bank.advance_frame(frame_index)

    return FrameResult(
        frame_index=frame_index,
        timestamp=timestamp,
        assignment=final_assignment,
        rejected_candidates=list(assoc.rejected),
        observations=observations,
        geometry=geometry,
        semantic=semantic,
    )


def run_association_from_cache(
    cache,
    mode: AssociationMode,
    associator: Optional[Associator] = None,
    min_valid_points: int = 8,
) -> tuple:
    """缓存模式：从因果缓存重放关联。

    同一缓存可运行三种关联模式（阶段 1 验收）。关联逻辑不访问真值。

    Returns:
        (results: list[FrameResult], bank: InstanceMemoryBank,
         decisions_log: list)
    """
    if associator is None:
        associator = Associator(mode=mode)
    else:
        associator.mode = mode

    bank = InstanceMemoryBank()
    results: list[FrameResult] = []
    decisions_log: list = []

    for t, rec in enumerate(cache.frames):
        geometry = GeometryOutput(
            pose_w_from_c=np.asarray(rec["pose_w_from_c"]),
            pointmap_world=np.asarray(rec["pointmap_world"]),
            valid=np.asarray(rec["valid"], dtype=bool),
            confidence=np.asarray(rec["confidence"]),
        )
        candidates = []
        num = int(rec.get("num_candidates", 0))
        for ci in range(num):
            prefix = f"cand{ci}/"
            candidates.append(
                SemanticCandidate(
                    mask=np.asarray(rec[prefix + "mask"], dtype=bool),
                    category=str(rec[prefix + "category"]),
                    category_score=float(rec[prefix + "score"]),
                    appearance=np.asarray(rec[prefix + "appearance"]),
                )
            )
        semantic = SemanticOutput(candidates=candidates)
        observations = extract_observations(semantic, geometry, min_valid_points)
        result = associate_frame(
            observations, bank, associator, t, float(t), geometry, semantic
        )
        results.append(result)
        for d in _frame_decisions(t, result):
            decisions_log.append(d)

    return results, bank, decisions_log


def _frame_decisions(frame_index: int, result: FrameResult) -> list:
    """将帧结果转为可记录条目（候选分数、拒绝条件、匹配结果）。"""
    entries = []
    for cand_idx, pid in result.assignment.items():
        entries.append(
            {
                "frame": frame_index,
                "candidate": cand_idx,
                "action": "match_or_create",
                "persistent_id": pid,
                "category": result.observations[cand_idx].category,
                "score": result.observations[cand_idx].category_score,
            }
        )
    for cand_idx in result.rejected_candidates:
        obs = result.observations[cand_idx]
        entries.append(
            {
                "frame": frame_index,
                "candidate": cand_idx,
                "action": "reject",
                "persistent_id": None,
                "category": obs.category,
                "score": obs.category_score,
            }
        )
    return entries


class OnlinePipeline:
    """在线主流程骨架（阶段 3 交付，当前未接真实模型）。

    流程：
        当前帧 -> 因果几何预测与文本条件分割 -> 当前实例观测提取
              -> 历史实例关联 -> 输出持续 ID -> 更新记忆
    """

    def __init__(
        self,
        geometry_adapter,
        semantic_adapter,
        mode: AssociationMode = AssociationMode.APPEARANCE_GEOMETRY,
        text_queries: Optional[list] = None,
        min_valid_points: int = 8,
    ):
        self.geometry_adapter = geometry_adapter
        self.semantic_adapter = semantic_adapter
        self.associator = Associator(mode=mode)
        self.bank = InstanceMemoryBank()
        self.text_queries = text_queries or []
        self.min_valid_points = min_valid_points
        self._geo_state = None

    def reset(self) -> None:
        self.bank.reset()
        self.geometry_adapter.reset()
        self._geo_state = None

    def process_frame(self, frame: FrameInput) -> FrameResult:
        if any(a.is_mock for a in (self.geometry_adapter, self.semantic_adapter)):
            raise NotImplementedError(
                "在线模式要求真实模型适配器；当前仅有 mock 适配器。"
                "请先完成阶段 0 模型接入（可行性报告中的阻塞项）。"
            )
        geometry, self._geo_state = self.geometry_adapter.step(frame, self._geo_state)
        semantic = self.semantic_adapter.predict(frame, self.text_queries)
        observations = extract_observations(semantic, geometry, self.min_valid_points)
        return associate_frame(
            observations,
            self.bank,
            self.associator,
            frame.frame_index,
            frame.timestamp,
            geometry,
            semantic,
        )
