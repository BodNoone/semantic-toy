"""因果缓存（EXECUTION_PLAN.md 5.1）。

按时间顺序运行冻结模型（或 mock 适配器），保存几何、候选掩码和外观特征。
缓存只用于加速：阶段 3 在线运行仍遵循重建模型的因果状态更新，
不依赖未来帧。

缓存以 npz 持久化，并记录模型版本、配置、输入变换、尺度和坐标约定
（SequenceMetadata）。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from .adapters.base import GeometryAdapter, SemanticAdapter
from .contracts import FrameInput, SequenceMetadata


class SequenceCache:
    """单序列因果缓存。"""

    def __init__(self, metadata: SequenceMetadata):
        self.metadata = metadata
        self.frames: list[dict] = []

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def add_frame(self, record: dict) -> None:
        self.frames.append(record)

    def save(self, path: str) -> None:
        """保存缓存（frames.npz + metadata.json）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            **self._flatten_frames(),
        )
        meta_path = path.replace(".npz", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": asdict(self.metadata),
                    "num_frames": self.num_frames,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _flatten_frames(self) -> dict:
        out: dict[str, Any] = {}
        for t, rec in enumerate(self.frames):
            for key, val in rec.items():
                if val is None:
                    continue
                out[f"t{t}/{key}"] = np.asarray(val)
        out["_frame_keys"] = np.array(
            [json.dumps(sorted(rec.keys())) for rec in self.frames], dtype=object
        )
        return out

    @staticmethod
    def load(path: str) -> "SequenceCache":
        meta_path = path.replace(".npz", "_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        metadata = SequenceMetadata(**meta["metadata"])
        cache = SequenceCache(metadata)
        data = np.load(path, allow_pickle=True)
        frame_keys = [json.loads(k) for k in data["_frame_keys"]]
        num_frames = len(frame_keys)
        for t, keys in enumerate(frame_keys):
            rec = {}
            for key in keys:
                full = f"t{t}/{key}"
                if full in data:
                    rec[key] = data[full]
            cache.frames.append(rec)
        return cache


def build_cache(
    frames: list,
    geometry_adapter: GeometryAdapter,
    semantic_adapter: SemanticAdapter,
    metadata: SequenceMetadata,
    text_queries: list,
) -> SequenceCache:
    """按时间顺序因果运行两个适配器并缓存输出。

    注意：几何适配器携带因果状态（reset 后逐帧 step），缓存仅保存
    输出结果，不替代在线状态更新。
    """
    cache = SequenceCache(metadata)
    geometry_adapter.reset()
    geo_state: Optional[Any] = None
    for frame in frames:
        assert isinstance(frame, FrameInput)
        geo_out, geo_state = geometry_adapter.step(frame, geo_state)
        sem_out = semantic_adapter.predict(frame, text_queries)
        record = {
            "pose_w_from_c": geo_out.pose_w_from_c,
            "pointmap_world": geo_out.pointmap_world,
            "valid": geo_out.valid,
            "confidence": geo_out.confidence,
            "geo_model_note": np.array(geo_out.model_note, dtype=object),
            "sem_model_note": np.array(sem_out.model_note, dtype=object),
            "num_candidates": np.array(len(sem_out.candidates)),
        }
        for ci, cand in enumerate(sem_out.candidates):
            record[f"cand{ci}/mask"] = cand.mask
            record[f"cand{ci}/category"] = np.array(cand.category, dtype=object)
            record[f"cand{ci}/score"] = np.array(cand.category_score)
            record[f"cand{ci}/appearance"] = cand.appearance
        cache.add_frame(record)
    return cache
