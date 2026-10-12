"""无训练关联基线（EXECUTION_PLAN.md 5.3）。

三种模式：
  - appearance：类别兼容性 + 外观相似度
  - geometry：类别兼容性 + 空间兼容性
  - appearance_geometry：外观 + 质量调节的软空间约束

规则：
  - 显式一对一分配（Hungarian 算法）；
  - 支持拒绝匹配和创建新实例；
  - 空间项不使用对所有场景统一的严格距离门限；低几何质量、遮挡和
    视角变化通过观测质量降低其权重（软约束，非硬门限）；
  - 记录候选分数、拒绝条件和匹配结果（验收要求）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import AssociationMode, InstanceMemory, InstanceObservation


@dataclass
class AssociationDecision:
    """单个候选的关联决策记录（用于日志与失败分析）。"""

    candidate_index: int
    action: str                          # "match" | "create" | "reject"
    matched_id: int | None = None
    best_score: float = 0.0
    reason: str = ""


@dataclass
class AssociationResult:
    assignment: dict = field(default_factory=dict)      # 候选下标 -> 持续 ID
    rejected: list = field(default_factory=list)
    decisions: list = field(default_factory=list)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def appearance_score(obs: InstanceObservation, mem: InstanceMemory) -> float:
    """外观相似度，映射到 [0, 1]。"""
    if mem.appearance is None:
        return 0.0
    sim = _cosine(obs.appearance, mem.appearance)
    return 0.5 * (sim + 1.0)


def spatial_score(obs: InstanceObservation, mem: InstanceMemory) -> float:
    """软空间兼容性分数（不是硬距离门限）。

    以记忆空间协方差定义马氏距离，转成高斯型分数；再乘以观测质量权重，
    使低几何质量、遮挡、视角变化自动降低空间项影响。
    任一侧缺空间状态时返回 0（该模式下此对不参与空间判定）。
    """
    if not obs.has_valid_geometry or not mem.has_spatial_state:
        return 0.0
    diff = obs.surface_center - mem.spatial_center
    cov = mem.spatial_covariance
    if cov is None:
        return 0.0
    # 正则化协方差，避免奇异
    reg_cov = cov + np.eye(3, dtype=cov.dtype) * 1e-3
    try:
        inv = np.linalg.inv(reg_cov)
    except np.linalg.LinAlgError:
        return 0.0
    d2 = float(diff @ inv @ diff)
    gauss = float(np.exp(-0.5 * d2))
    # 质量调节的软约束：低质量观测的空间证据被压低
    return gauss * float(np.clip(obs.quality, 0.0, 1.0))


class Associator:
    """帧级候选-记忆关联器。

    Args:
        mode: 关联模式。
        match_threshold: 最佳分数低于该值则不匹配（可能新建实例）。
        min_category_score: 低于该值的候选直接拒绝。
        create_threshold: 新建实例要求的最低类别分数（高于拒绝线）。
        lambda_spatial: appearance_geometry 模式下空间项权重。
    """

    def __init__(
        self,
        mode: AssociationMode = AssociationMode.APPEARANCE_GEOMETRY,
        match_threshold: float = 0.35,
        min_category_score: float = 0.2,
        create_threshold: float = 0.3,
        lambda_spatial: float = 0.4,
    ):
        self.mode = mode
        self.match_threshold = match_threshold
        self.min_category_score = min_category_score
        self.create_threshold = create_threshold
        self.lambda_spatial = lambda_spatial

    def _pair_score(self, obs: InstanceObservation, mem: InstanceMemory) -> float:
        if self.mode == AssociationMode.APPEARANCE:
            return appearance_score(obs, mem)
        if self.mode == AssociationMode.GEOMETRY:
            return spatial_score(obs, mem)
        # appearance_geometry：外观 + 质量调节的软空间约束
        app = appearance_score(obs, mem)
        geo = spatial_score(obs, mem)
        return (1.0 - self.lambda_spatial) * app + self.lambda_spatial * geo

    def associate(
        self,
        observations: list,
        memories: list,
    ) -> AssociationResult:
        """执行一帧关联。空观测列表合法返回空结果（空观测覆盖）。"""
        result = AssociationResult()
        if not observations:
            return result

        # 类别兼容性：仅同类别可匹配（硬过滤，构造候选-记忆可行对）
        n, m = len(observations), len(memories)
        score = np.full((n, m), -np.inf, dtype=np.float64)
        for i, obs in enumerate(observations):
            for j, mem in enumerate(memories):
                if obs.category != mem.category:
                    continue
                score[i, j] = self._pair_score(obs, mem)

        # Hungarian 一对一分配（对 -score；不可行对用大代价）
        finite = np.isfinite(score)
        cost = np.where(finite, -score, 1e9)
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_cand = set()
        for i, j in zip(row_ind, col_ind):
            if not finite[i, j] or score[i, j] < self.match_threshold:
                continue
            result.assignment[int(i)] = memories[j].persistent_id
            matched_cand.add(int(i))
            result.decisions.append(
                AssociationDecision(
                    candidate_index=int(i),
                    action="match",
                    matched_id=memories[j].persistent_id,
                    best_score=float(score[i, j]),
                    reason=f"matched to {memories[j].persistent_id} ({self.mode.value})",
                )
            )

        for i, obs in enumerate(observations):
            if i in matched_cand:
                continue
            row = score[i]
            best = float(row.max()) if row.size and np.isfinite(row).any() else 0.0
            if obs.category_score < self.min_category_score:
                result.rejected.append(i)
                result.decisions.append(
                    AssociationDecision(
                        candidate_index=i,
                        action="reject",
                        best_score=best,
                        reason=f"category_score {obs.category_score:.2f} < min {self.min_category_score}",
                    )
                )
            elif obs.category_score >= self.create_threshold:
                # 新建实例：assignment 用 -1 占位，由流水线替换为真实新 ID
                result.assignment[i] = -1
                result.decisions.append(
                    AssociationDecision(
                        candidate_index=i,
                        action="create",
                        best_score=best,
                        reason="no compatible memory above match_threshold",
                    )
                )
            else:
                result.rejected.append(i)
                result.decisions.append(
                    AssociationDecision(
                        candidate_index=i,
                        action="reject",
                        best_score=best,
                        reason="ambiguous low-confidence candidate",
                    )
                )
        return result
