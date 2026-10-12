"""评测指标（EXECUTION_PLAN.md 第 6 节）。

指标：
  - 身份切换（ID switches）和身份碎片化（fragments）
  - 按失踪时长分组的正确重关联率
  - 错误合并率
  - 漏检率与重现检测率
  - 单帧时间、峰值显存、记忆占用（运行时统计在 pipeline 外部记录）

实现前定义（计划要求）：
  - 实例匹配规则：每帧按预测掩码与真值掩码的 IoU>=0.5 匹配，一对一；
  - 重关联事件定义：真值实例连续缺席 >= min_gap 帧后再次可见，
    且系统在新可见首帧给出与该实例缺席前一致的持续 ID，则记为正确重关联；
  - 错误合并定义：同一帧内两个不同真值实例被赋予同一持续 ID；
  - 各指标分母：重关联率分母为全部重关联事件数；错误合并率分母为
    全部"同帧可见真值实例对"数；漏检率分母为全部真值可见实例-帧；
  - 区分“重现后未检测到”（漏检）和“检测到但关联错误”（切换/合并）。

评测使用真值匹配仅发生在评测模块内部；关联逻辑本身不得访问真值。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvaluationReport:
    id_switches: int = 0
    id_fragments: int = 0
    reassociation_total: int = 0
    reassociation_correct: int = 0
    reassociation_by_gap: dict = field(default_factory=dict)
    merge_total: int = 0
    merge_errors: int = 0
    missed_total: int = 0
    missed: int = 0
    redetected_total: int = 0
    redetected: int = 0
    num_frames: int = 0

    @property
    def reassociation_rate(self) -> float:
        if self.reassociation_total == 0:
            return float("nan")
        return self.reassociation_correct / self.reassociation_total

    @property
    def merge_error_rate(self) -> float:
        if self.merge_total == 0:
            return float("nan")
        return self.merge_errors / self.merge_total

    @property
    def miss_rate(self) -> float:
        if self.missed_total == 0:
            return float("nan")
        return self.missed / self.missed_total

    def as_dict(self) -> dict:
        return {
            "id_switches": self.id_switches,
            "id_fragments": self.id_fragments,
            "reassociation_total": self.reassociation_total,
            "reassociation_correct": self.reassociation_correct,
            "reassociation_rate": self.reassociation_rate,
            "reassociation_by_gap": self.reassociation_by_gap,
            "merge_total": self.merge_total,
            "merge_errors": self.merge_errors,
            "merge_error_rate": self.merge_error_rate,
            "missed_total": self.missed_total,
            "missed": self.missed,
            "miss_rate": self.miss_rate,
            "redetected_total": self.redetected_total,
            "redetected": self.redetected,
            "num_frames": self.num_frames,
        }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def match_frame(pred_masks: list, pred_ids: list, gt_masks: list, iou_thr: float = 0.5):
    """帧级一对一匹配（贪心，IoU 降序）。

    Returns:
        matches: list[(pred_idx, gt_idx)]
    """
    pairs = []
    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            iou = _iou(pm, gm)
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi))
    return matches


def evaluate(
    results: list,
    ground_truth: dict,
    min_gap: int = 3,
    iou_thr: float = 0.5,
) -> EvaluationReport:
    """按上述定义计算全部指标。

    Args:
        results: pipeline 输出的 FrameResult 列表（含 assignment 与
            observations）。
        ground_truth: 合成场景真值：{"visible_ids", "masks", ...}。
        min_gap: 重关联事件的最小缺席帧数。
    """
    report = EvaluationReport(num_frames=len(results))
    # 真值实例 -> 最近一次分配的持续 ID（用于切换/碎片/重关联判定）
    last_pid: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    absent_since: dict[int, int] = {}

    for t, res in enumerate(results):
        gt_vis = ground_truth["visible_ids"][t]
        gt_masks = ground_truth["masks"][t]

        pred_masks = [obs.mask for obs in res.observations]
        pred_ids = [
            res.assignment.get(ci) for ci in range(len(res.observations))
        ]
        detected_gt = set()

        matches = match_frame(pred_masks, pred_ids, gt_masks, iou_thr)
        pred_to_gt = {pi: gi for pi, gi in matches}

        for pi, gid in pred_to_gt.items():
            gt_id = gt_vis[gid]
            pid = pred_ids[pi]
            detected_gt.add(gt_id)
            if gt_id in last_pid:
                if last_pid[gt_id] != pid:
                    report.id_switches += 1
                    if gt_id not in absent_since:  # 未缺席的切换 = 碎片
                        report.id_fragments += 1
            last_pid[gt_id] = pid

        # 漏检：真值可见但未匹配到任何预测
        for gid in gt_vis:
            report.missed_total += 1
            if gid not in detected_gt:
                report.missed += 1

        # 错误合并：同帧两个不同真值实例拿到同一持续 ID
        for a in range(len(gt_vis)):
            for b in range(a + 1, len(gt_vis)):
                report.merge_total += 1
                ga, gb = gt_vis[a], gt_vis[b]
                pa = last_pid.get(ga)
                pb = last_pid.get(gb)
                # 仅当两者本帧都被检测且 ID 相同才计错误
                if ga in detected_gt and gb in detected_gt and pa == pb and pa is not None:
                    report.merge_errors += 1

        # 缺席/重关联状态机
        for gid in last_seen.keys() | set(gt_vis):
            if gid in gt_vis:
                if gid in absent_since:
                    gap = t - absent_since[gid] - 1
                    if gap >= min_gap:
                        report.reassociation_total += 1
                        bucket = f"gap_{gap // 5 * 5}-{gap // 5 * 5 + 4}"
                        grp = report.reassociation_by_gap.setdefault(
                            bucket, [0, 0]
                        )
                        grp[1] += 1
                        # 正确：本帧分配的 ID 与缺席前一致
                        if last_pid.get(gid) is not None and gid in detected_gt:
                            idx = gt_vis.index(gid)
                            cur = None
                            for pi, gi in matches:
                                if gi == idx:
                                    cur = pred_ids[pi]
                                    break
                            if cur == last_pid.get(gid):
                                report.reassociation_correct += 1
                                grp[0] += 1
                        report.redetected_total += 1
                        if gid in detected_gt:
                            report.redetected += 1
                    del absent_since[gid]
                last_seen[gid] = t
            else:
                if gid not in absent_since and last_seen.get(gid, -1) == t - 1:
                    absent_since[gid] = t - 1

    return report
