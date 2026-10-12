"""SAM3 真实模型适配器骨架（阶段 0 交付）。

当前环境缺少 torch 与 SAM3 权重，本文件仅提供接口实现与可行性检查入口；
实际接入前必须检查上游公开代码的真实接口（计划：不虚构模型接口）。

上游参考：
  - facebookresearch/sam3（SAM 3: Segment Anything with Concepts）
  - 图像：build_sam3_image_model + Sam3Processor.set_text_prompt
  - 视频：build_sam3_video_predictor，handle_request(start_session / add_prompt /
    propagate)
  - 局部特征：需检查 PE（Perception Encoder）主干能否输出逐像素特征；
    若不能直接获取，阶段 0 可行性报告必须记录该限制。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import FrameInput, GeometryOutput, SemanticOutput
from .base import GeometryAdapter, SemanticAdapter


class Sam3SemanticAdapter(SemanticAdapter):
    """SAM3 文本条件分割（检测式，逐帧独立，供自定义关联使用）。

    注意：此适配器输出不含原生跟踪 ID；原生视频跟踪走
    Sam3NativeTrackerAdapter（基线专用，不与自定义关联混用）。
    """

    is_mock = False

    def __init__(self, checkpoint: str = "facebook/sam3", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            # 接入前按上游仓库实际接口核对（不虚构接口）
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:  # pragma: no cover - 环境检查路径
            raise RuntimeError(
                "SAM3 未安装。请先安装上游仓库并下载权重，"
                "参见阶段 0 可行性报告记录的阻塞项。"
            ) from exc
        self._model = build_sam3_image_model(self.checkpoint)
        self._processor = Sam3Processor(self._model)

    def predict(self, frame: FrameInput, text_queries: list) -> SemanticOutput:
        self._lazy_load()
        raise NotImplementedError(
            "阶段 0 待接入：按上游实际接口实现逐类别文本提示分割、"
            "掩码后处理与区域外观特征提取，并记录局部特征可用性。"
        )

    @property
    def model_note(self) -> str:
        return f"sam3({self.checkpoint})"


class Sam3NativeTrackerAdapter:
    """SAM3 原生视频跟踪基线（独立适配器，不与自定义关联混用）。

    输出中的 native_track_id 仅供基线评测使用。
    """

    is_mock = False

    def __init__(self, checkpoint: str = "facebook/sam3", device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = device
        self._predictor = None

    def start_session(self, frames_source: str) -> None:
        raise NotImplementedError("阶段 0 待接入：start_session 接口")

    def step(self, frame: FrameInput, text_queries: list) -> SemanticOutput:
        raise NotImplementedError(
            "阶段 0 待接入：add_prompt(frame_index, text=...) 与逐帧传播"
        )

    @property
    def model_note(self) -> str:
        return f"sam3_native_tracker({self.checkpoint})"


class FeedforwardGeometryAdapter(GeometryAdapter):
    """流式前馈重建骨干适配器骨架（阶段 0 交付）。

    候选骨干（接入前逐一检查逐帧因果接口、位姿/点图/置信度约定）：
      - CUT3R（恒定内存循环状态）
      - Spann3R（3D 锚定空间记忆，增量全局系点图）
    """

    is_mock = False

    def __init__(self, backbone: str = "cut3r", checkpoint: str | None = None):
        self.backbone = backbone
        self.checkpoint = checkpoint
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        raise RuntimeError(
            "重建骨干权重不可用：当前环境缺少 torch 与模型权重。"
            "此阻塞已记录在阶段 0 可行性报告中。"
        )

    def reset(self) -> None:
        self._model = None

    def step(self, frame: FrameInput, state: Any) -> tuple:
        self._lazy_load()
        raise NotImplementedError(
            "阶段 0 待接入：实现逐帧因果前向，输出 T_world_from_camera、"
            "全局系点图、有效掩码与置信度。"
        )

    @property
    def model_note(self) -> str:
        return f"feedforward({self.backbone})"
