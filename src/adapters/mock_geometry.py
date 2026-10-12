"""模拟重建骨干适配器（明确标识为 mock，仅用于接口与逻辑测试）。

数据来源：FrameInput.extra 中的合成真值（由 data/synthetic.py 生成）。
不虚构模型输出：model_note 显式标注 mock，结果不得冒充真实模型
（计划执行规则 4）。

行为约定与真实适配器一致：
  - 逐帧因果调用，只读当前帧与历史状态；
  - reset() 重置状态；
  - 输出满足 GeometryOutput 的坐标约定（T_world_from_c、世界系点图、
    显式有效掩码、置信度）。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import FrameInput, GeometryOutput
from .base import GeometryAdapter


class MockGeometryAdapter(GeometryAdapter):
    """以合成真值充当预测的几何适配器（接口测试用）。

    可选注入噪声/失效，模拟真实骨干的不完美：
      - pose_noise: 位姿平移噪声标准差（世界系）
      - invalid_ratio: 随机置无效的像素比例（模拟无效几何）
      - 若某帧 extra 缺少标注，则输出全无效几何（模拟失败帧）。
    """

    is_mock = True

    def __init__(
        self,
        pose_noise: float = 0.0,
        invalid_ratio: float = 0.0,
        rng_seed: int = 0,
    ):
        self.pose_noise = pose_noise
        self.invalid_ratio = invalid_ratio
        self._rng = np.random.default_rng(rng_seed)
        self._frame_count = 0

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._rng.integers(0, 2**31))
        self._frame_count = 0

    def step(self, frame: FrameInput, state: Any) -> tuple:
        self._frame_count += 1
        extra = frame.extra or {}
        h, w = frame.image.shape[:2]

        pose = extra.get("pose_w_from_c")
        pointmap = extra.get("pointmap_world")
        conf = extra.get("confidence")

        if pose is None or pointmap is None:
            # 模拟失败帧：输出全无效几何（显式有效掩码，不用零坐标冒充）
            geo = GeometryOutput(
                pose_w_from_c=np.eye(4),
                pointmap_world=np.zeros((h, w, 3), dtype=np.float32),
                valid=np.zeros((h, w), dtype=bool),
                confidence=np.zeros((h, w), dtype=np.float32),
                intrinsics=frame.intrinsics,
                model_note=self.model_note,
            )
            return geo, state

        pose = np.array(pose, dtype=np.float64, copy=True)
        if self.pose_noise > 0:
            pose[:3, 3] += self._rng.normal(0.0, self.pose_noise, size=3)

        valid = np.ones((h, w), dtype=bool)
        if "valid_mask" in extra:
            valid &= np.asarray(extra["valid_mask"], dtype=bool)
        if self.invalid_ratio > 0:
            drop = self._rng.random((h, w)) < self.invalid_ratio
            valid &= ~drop

        pointmap = np.array(pointmap, dtype=np.float32, copy=True)
        if conf is None:
            conf = np.where(valid, 0.9, 0.0).astype(np.float32)
        conf = np.asarray(conf, dtype=np.float32)

        geo = GeometryOutput(
            pose_w_from_c=pose,
            pointmap_world=pointmap,
            valid=valid,
            confidence=conf,
            intrinsics=frame.intrinsics,
            model_note=self.model_note,
        )
        return geo, state

    @property
    def model_note(self) -> str:
        return "MOCK geometry (synthetic annotations; not a real model output)"
