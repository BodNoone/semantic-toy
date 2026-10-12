"""数据契约定义（EXECUTION_PLAN.md 第 3 节）。

所有跨模块传递的数据结构集中在此文件，避免各模块自行定义坐标与格式。
第一版：Python、numpy 实现、单序列、batch size 为 1。
真实模型接入后，保持本契约不变（计划第 2 节：先统一数据接口，再并行开发）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class InstanceStatus(Enum):
    """实例记忆状态（EXECUTION_PLAN.md 5.4）。"""

    ACTIVE = "active"      # 近期可见
    DORMANT = "dormant"    # 暂不可见，仍保留身份


class AssociationMode(Enum):
    """无训练关联模式（EXECUTION_PLAN.md 5.3）。"""

    APPEARANCE = "appearance"
    GEOMETRY = "geometry"
    APPEARANCE_GEOMETRY = "appearance_geometry"


@dataclass
class SequenceMetadata:
    """序列级元数据：缓存记录模型版本、配置、输入变换、尺度和坐标约定。"""

    sequence_id: str
    image_width: int
    image_height: int
    # 单目尺度约定：预测坐标所处尺度空间名称。
    # 不混合未对齐预测坐标与米制真值（评测分轨道报告）。
    scale_convention: str = "monocular_arbitrary"
    world_frame_note: str = "world frame fixed at sequence start (camera 0)"
    # 记录输入预处理变换（resize / crop / padding），内参随变换同步调整。
    preprocess_transforms: list = field(default_factory=list)
    geometry_model: str = "unknown"
    semantic_model: str = "unknown"
    config: dict = field(default_factory=dict)


@dataclass
class FrameInput:
    """当前帧输入。

    extra: mock 适配器专用的标注通道；真实适配器不得依赖该字段
    （计划：不将模拟输出冒充模型结果，真实接口不可用时明确记录）。
    """

    image: np.ndarray                    # (H, W, 3) uint8, RGB
    frame_index: int
    timestamp: float
    text_queries: list                   # 待构建语义的类别文本
    intrinsics: Optional[np.ndarray] = None  # (3, 3)，与 image 尺寸对应
    extra: Optional[dict] = None


@dataclass
class GeometryOutput:
    """重建骨干每帧输出。

    坐标约定：
      - pose_w_from_c 为 T_world_from_camera，满足 X_world = T @ X_camera（齐次）。
      - pointmap_world: (H, W, 3) 世界坐标点图，与图像像素一一对应。
      - valid: (H, W) bool，显式有效掩码；不将零坐标当作有效观测。
      - confidence: (H, W) float，[0, 1] 逐像素几何置信度。
    """

    pose_w_from_c: np.ndarray            # (4, 4)
    pointmap_world: np.ndarray           # (H, W, 3)
    valid: np.ndarray                    # (H, W) bool
    confidence: np.ndarray               # (H, W) float
    intrinsics: Optional[np.ndarray] = None
    model_note: str = ""                 # 模型版本/来源标识


@dataclass
class SemanticCandidate:
    """单个候选掩码（SemanticOutput 的元素）。

    注意：自定义关联不得读取原生跟踪 ID（计划第 3 节）。
    native_track_id 仅用于 SAM3 原生视频跟踪基线对照。
    """

    mask: np.ndarray                     # (H, W) bool
    category: str
    category_score: float
    appearance: np.ndarray               # (D,) 区域外观特征
    native_track_id: Optional[int] = None


@dataclass
class SemanticOutput:
    """语义分支每帧输出。"""

    candidates: list = field(default_factory=list)   # list[SemanticCandidate]
    model_note: str = ""


@dataclass
class InstanceObservation:
    """当前帧单个实例的观测。

    术语约定（计划第 3 节）：
      - surface_center 是“当前可见表面中心”，不等于完整物体中心；
      - spatial_spread 是表面点云协方差（命名沿用计划术语），
        不能直接视为中心估计的不确定性；
      - has_valid_geometry=False 时空间字段为 None；该候选仍可参与外观关联，
        不得产生 NaN 或伪造空间位置。
    """

    candidate_index: int
    mask: np.ndarray                     # (H, W) bool
    category: str
    category_score: float
    appearance: np.ndarray               # (D,)
    has_valid_geometry: bool
    surface_center: Optional[np.ndarray] = None      # (3,) 世界系
    spatial_spread: Optional[np.ndarray] = None      # (3, 3)
    quality: float = 0.0
    valid_point_ratio: float = 0.0
    area: int = 0
    border_truncated: bool = False


@dataclass
class InstanceMemory:
    """持续实例记忆条目，因果维护（计划 5.4）。"""

    persistent_id: int
    category: str
    status: InstanceStatus = InstanceStatus.ACTIVE
    last_seen_frame: int = -1
    observation_count: int = 0
    # 历史空间摘要（仅由质量足够的观测保守更新）
    spatial_center: Optional[np.ndarray] = None      # (3,)
    spatial_covariance: Optional[np.ndarray] = None  # (3, 3)
    has_spatial_state: bool = False
    last_spatial_update_frame: int = -1
    # 外观摘要（指数滑动平均）
    appearance: Optional[np.ndarray] = None          # (D,)


@dataclass
class FrameResult:
    """单帧最终输出，用于保存和评测。"""

    frame_index: int
    timestamp: float
    # 候选下标 -> 持续 ID
    assignment: dict = field(default_factory=dict)
    # 被拒绝（不匹配也不新建）的候选下标
    rejected_candidates: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    geometry: Optional[GeometryOutput] = None
    semantic: Optional[SemanticOutput] = None
