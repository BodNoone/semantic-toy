"""阶段 0/1：构建因果缓存（mock 适配器 + 合成数据）。

用法（仓库根目录）：
    python scripts/cache_sequence.py --out outputs/cache/orbit.npz \
        --num-frames 40 --num-chairs 4

说明：
  - 当前环境缺少 torch 与真实权重，使用明确标识的 mock 适配器
    （计划执行规则 4：模拟输出不冒充模型结果）；
  - 真值独立存放于 *_gt.npz，系统关联逻辑不读取真值。

完成真实模型实验所需条件（阻塞项记录）：
  1. PyTorch 与 CUDA 环境；
  2. SAM3 权重与上游仓库（文本条件分割接口核对）；
  3. 流式重建骨干权重（CUT3R / Spann3R 二选一起步）；
  4. 带实例身份真值的评测数据（合成或 ScanNet++ 等）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import MockGeometryAdapter, MockSemanticAdapter
from src.cache import build_cache
from src.data.synthetic import generate_orbit_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="构建因果缓存（阶段 0/1）")
    parser.add_argument("--out", default="outputs/cache/orbit.npz")
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument("--num-chairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene = generate_orbit_scene(
        num_frames=args.num_frames,
        num_chairs=args.num_chairs,
        seed=args.seed,
    )
    geometry_adapter = MockGeometryAdapter()
    semantic_adapter = MockSemanticAdapter()

    cache = build_cache(
        scene.frames,
        geometry_adapter,
        semantic_adapter,
        scene.metadata,
        scene.text_queries,
    )
    cache.save(args.out)

    # 真值独立存放（关联逻辑不得访问）
    gt = scene.ground_truth
    gt_path = args.out.replace(".npz", "_gt.npz")
    np.savez_compressed(
        gt_path,
        object_centers=gt["object_centers"],
        visible_ids=np.array(gt["visible_ids"], dtype=object),
        masks=np.array(gt["masks"], dtype=object),
        poses=np.array(gt["poses"]),
    )

    print(f"缓存已保存: {args.out} ({cache.num_frames} 帧)")
    print(f"真值已独立保存: {gt_path}")
    print(f"几何模型: {cache.metadata.geometry_model}")
    print(f"语义模型: {cache.metadata.semantic_model}")
    print(f"尺度约定: {cache.metadata.scale_convention}")


if __name__ == "__main__":
    main()
