"""帧级共享数据结构。

所有引擎、耦合层与建图模块都通过这些 dataclass 交互，避免直接依赖
具体模型输出格式。字段含 shape 约定说明。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FrameGeometry:
    """单帧几何输出。

    world_points:   [H, W, 3] 世界坐标系稠密点 (StreamVGGT 首帧参考系)
    confidence:     [H, W]    逐点置信度, 已归一化到 [0, 1]
    world_to_camera:[4, 4]    世界→相机外参 (行向量约定)
    intrinsics:     [3, 3]    相机内参
    frame_index:    int       序列内帧序号
    """

    world_points: Any  # torch.Tensor [H,W,3]
    confidence: Any  # torch.Tensor [H,W]
    world_to_camera: Any  # torch.Tensor [4,4]
    intrinsics: Any  # torch.Tensor [3,3]
    frame_index: int


@dataclass(frozen=True)
class FrameTracking:
    """单帧语义跟踪输出。

    masks:          [I, H, W] bool, I 为当前可见实例数
    track_ids:      [I] int, 跨帧持久 ID (>=0)
    scores:         [I] float, 跟踪置信度
    categories:     [I] str, 类别文本
    frame_index:    int
    """

    masks: Any  # torch.Tensor [I,H,W] bool
    track_ids: Any  # torch.Tensor [I] int
    scores: Any  # torch.Tensor [I] float
    categories: list[str]  # [I]
    frame_index: int


@dataclass(frozen=True)
class AnchorObservation:
    """一个 track 在某一帧的 3D 锚点观测。

    centroid:  [3] mask 内高置信点的鲁棒质心 (中位数)
    scale:     标量, 观测的几何尺度 (robust RMS)
    count:     参与的有效点数
    """

    centroid: Any  # torch.Tensor [3]
    scale: float
    count: int


@dataclass(frozen=True)
class DriftCorrection:
    """漂移校正结果。

    applied:    是否采用该校正 (门控通过)
    scale:      Sim(3) 尺度 (默认为 1.0, 刚性)
    rotation:   [3,3]
    translation:[3]
    inliers:    参与求解的锚点对应数
    residual:   中位残差 (3D 距离)
    reason:     未采用时的原因 (applied=False)
    """

    applied: bool
    scale: float = 1.0
    rotation: Any = None  # torch.Tensor [3,3]
    translation: Any = None  # torch.Tensor [3]
    inliers: int = 0
    residual: float = 0.0
    reason: str = ""


@dataclass
class TrackState:
    """一个持久实例的跨帧状态。"""

    track_id: int
    category: str
    anchor: AnchorObservation
    last_seen: int  # 帧序号
    point_count: int = 0  # 累计建图点数


@dataclass
class MapPoint:
    """一个建图点。"""

    position: Any  # torch.Tensor [3]
    track_id: int
    category: str
    confidence: float
