"""阶段 2：评测入口（两条轨道）。

轨道：
  - predicted：使用重建输出（预测几何），评估实际系统；
  - gt_diagnostic：使用真值几何，隔离空间关联机制问题，结果单独报告。

用法（仓库根目录）：
    python scripts/evaluate.py --cache outputs/cache/orbit.npz \
        --gt outputs/cache/orbit_gt.npz --out-dir outputs/eval

注意：超参数只在训练/验证集调整；小规模结果只作为可行性证据。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import SequenceCache
from src.contracts import AssociationMode
from src.evaluation.metrics import evaluate
from src.pipeline import run_association_from_cache


def load_ground_truth(gt_path: str) -> dict:
    data = np.load(gt_path, allow_pickle=True)
    return {
        "object_centers": data["object_centers"],
        "visible_ids": [list(v) for v in data["visible_ids"]],
        "masks": [list(m) for m in data["masks"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 2 评测")
    parser.add_argument("--cache", default="outputs/cache/orbit.npz")
    parser.add_argument("--gt", default="outputs/cache/orbit_gt.npz")
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--min-gap", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = SequenceCache.load(args.cache)
    gt = load_ground_truth(args.gt)

    table = []
    for mode in AssociationMode:
        results, bank, _ = run_association_from_cache(cache, mode)
        report = evaluate(results, gt, min_gap=args.min_gap)
        row = {"mode": mode.value}
        row.update(report.as_dict())
        table.append(row)
        print(f"[{mode.value}] 切换={row['id_switches']} "
              f"重关联={row['reassociation_correct']}/{row['reassociation_total']} "
              f"合并错误={row['merge_errors']}/{row['merge_total']} "
              f"漏检={row['missed']}/{row['missed_total']}")

    with open(os.path.join(args.out_dir, "association_table.json"), "w",
              encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print(f"结果表已保存: {os.path.join(args.out_dir, 'association_table.json')}")
    print("注：当前为 mock 模型 + 合成几何（预测轨道=真值轨道），"
          "真实模型接入后需分两轨道单独报告。")


if __name__ == "__main__":
    main()
