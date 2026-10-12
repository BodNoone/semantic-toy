"""阶段 1：同一缓存运行三种关联模式，生成叠加与点云。

用法（仓库根目录）：
    python scripts/run_sequence.py --cache outputs/cache/orbit.npz \
        --out-dir outputs/run --mode all

验收对应：
  - 同一缓存可运行三种关联模式；
  - 记录候选分数、拒绝条件和匹配结果（decisions.json）；
  - 生成带持续 ID 的叠加帧（.ppm）与语义点云（.ply）。
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
from src.pipeline import run_association_from_cache
from src.visualization import build_semantic_pointcloud, save_overlay_ppm, save_pointcloud_ply

MODES = [AssociationMode.APPEARANCE, AssociationMode.GEOMETRY,
         AssociationMode.APPEARANCE_GEOMETRY]


def run_mode(cache: SequenceCache, mode: AssociationMode, out_dir: str) -> dict:
    mode_dir = os.path.join(out_dir, mode.value)
    os.makedirs(mode_dir, exist_ok=True)

    results, bank, decisions = run_association_from_cache(cache, mode)

    # 叠加帧与决策日志
    for res in results:
        path = os.path.join(mode_dir, f"frame_{res.frame_index:04d}.ppm")
        img = np.zeros((cache.metadata.image_height,
                        cache.metadata.image_width, 3), dtype=np.uint8)
        if res.geometry is not None:
            valid = res.geometry.valid
            img[valid] = 120
        save_overlay_ppm(path, img, res.observations, res.assignment)
    with open(os.path.join(mode_dir, "decisions.json"), "w", encoding="utf-8") as f:
        json.dump(decisions, f, ensure_ascii=False, indent=1)

    # 语义点云
    points, colors, pids = build_semantic_pointcloud(results)
    save_pointcloud_ply(os.path.join(mode_dir, "semantic_map.ply"),
                        points, colors, pids)

    return {
        "mode": mode.value,
        "num_frames": len(results),
        "num_instances": bank.count(),
        "num_decisions": len(decisions),
        "rejected": sum(1 for d in decisions if d["action"] == "reject"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 1 关联运行")
    parser.add_argument("--cache", default="outputs/cache/orbit.npz")
    parser.add_argument("--out-dir", default="outputs/run")
    parser.add_argument("--mode", default="all",
                        choices=["appearance", "geometry", "appearance_geometry", "all"])
    args = parser.parse_args()

    cache = SequenceCache.load(args.cache)
    modes = MODES if args.mode == "all" else [AssociationMode(args.mode)]

    summary = []
    for mode in modes:
        info = run_mode(cache, mode, args.out_dir)
        summary.append(info)
        print(f"[{info['mode']}] 帧数={info['num_frames']} "
              f"实例数={info['num_instances']} 拒绝={info['rejected']}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
