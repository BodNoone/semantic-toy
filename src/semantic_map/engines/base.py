"""引擎接口与公共工具。

ReconstructionEngine 与 TrackingEngine 是 Protocol: mock 与真实模型实现
只要满足接口即可被 pipeline 无差别调用。
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..types import FrameGeometry, FrameTracking


class ReconstructionEngine(Protocol):
    """流式重建引擎: 逐帧推进, 返回几何帧。"""

    def next_frame(self) -> FrameGeometry:
        """返回下一帧几何, 序列结束时 raise StopIteration。"""
        ...

    def reset(self) -> None:
        """回到序列起点。"""
        ...


class TrackingEngine(Protocol):
    """语义跟踪引擎: 逐帧推进, 返回跟踪帧。"""

    categories: Sequence[str]

    def next_frame(self) -> FrameTracking:
        """返回下一帧跟踪, 序列结束时 raise StopIteration。"""
        ...

    def reset(self) -> None:
        ...


class SequenceExhausted(Exception):
    """序列已结束。"""
