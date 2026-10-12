"""模型适配器包。

接口约定（EXECUTION_PLAN.md 第 3 节）：
    geometry.step(frame, state) -> geometry_output, next_state
    semantic.predict(frame, text_queries) -> semantic_output

SAM3 原生视频跟踪为独立适配器（Sam3NativeTrackerAdapter），仅作为基线，
不与自定义关联混用。

注意：本环境当前没有 torch 与真实权重，Mock 适配器仅用于接口与逻辑测试，
其输出在 model_note 中显式标注，不得冒充真实模型结果（计划执行规则 4）。
"""
from .base import GeometryAdapter, SemanticAdapter
from .mock_geometry import MockGeometryAdapter
from .mock_semantic import MockSemanticAdapter

__all__ = [
    "GeometryAdapter",
    "SemanticAdapter",
    "MockGeometryAdapter",
    "MockSemanticAdapter",
]
