"""适配器基类（EXECUTION_PLAN.md 第 3 节“模型适配器”）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..contracts import FrameInput, GeometryOutput, SemanticOutput


class GeometryAdapter(ABC):
    """重建骨干适配器：逐帧因果接口。

    约束：
      - 只读取当前帧和历史状态，不读取未来帧；
      - 不引入离线全序列预处理；
      - 每段序列开始时必须调用 reset() 重置全部状态。
    """

    is_mock: bool = False

    @abstractmethod
    def reset(self) -> None:
        """重置因果状态（新序列开始）。"""

    @abstractmethod
    def step(self, frame: FrameInput, state: Any) -> tuple:
        """因果前进一步。

        Returns:
            (GeometryOutput, next_state)
        """

    @property
    def model_note(self) -> str:
        return self.__class__.__name__


class SemanticAdapter(ABC):
    """语义分支适配器：文本条件分割（可选局部特征）。"""

    is_mock: bool = False

    @abstractmethod
    def predict(self, frame: FrameInput, text_queries: list) -> SemanticOutput:
        """对当前帧做文本条件实例分割。"""

    @property
    def model_note(self) -> str:
        return self.__class__.__name__
