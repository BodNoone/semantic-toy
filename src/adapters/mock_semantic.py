"""模拟语义分支适配器（明确标识为 mock，仅用于接口与逻辑测试）。

数据来源：FrameInput.extra 中的合成实例标注（由 data/synthetic.py 生成）。
model_note 显式标注 mock，不冒充真实模型输出。

行为约定与真实适配器一致：
  - 逐帧独立预测（检测式），不维护跨帧跟踪状态；
  - 不输出原生跟踪 ID（native_track_id 为 None）；
  - 支持掩码缺失（漏检）模拟。
"""
from __future__ import annotations

import numpy as np

from ..contracts import FrameInput, SemanticCandidate, SemanticOutput
from .base import SemanticAdapter


class MockSemanticAdapter(SemanticAdapter):
    """以合成标注充当检测结果的语义适配器。

    Args:
        miss_ratio: 每个实例在该帧被漏检的概率（模拟检测器漏检）。
        score_noise: 类别分数噪声。
        rng_seed: 随机种子。
    """

    is_mock = True

    def __init__(
        self,
        miss_ratio: float = 0.0,
        score_noise: float = 0.02,
        rng_seed: int = 0,
    ):
        self.miss_ratio = miss_ratio
        self.score_noise = score_noise
        self._rng = np.random.default_rng(rng_seed)

    def predict(self, frame: FrameInput, text_queries: list) -> SemanticOutput:
        extra = frame.extra or {}
        instances = extra.get("instances", [])
        queries = set(text_queries)
        candidates: list[SemanticCandidate] = []

        for inst in instances:
            # 类别不在查询列表中则不输出（文本条件）
            if inst["category"] not in queries:
                continue
            if self.miss_ratio > 0 and self._rng.random() < self.miss_ratio:
                continue  # 模拟漏检
            mask = np.asarray(inst["mask"], dtype=bool)
            score = float(np.clip(inst.get("score", 0.9) + self._rng.normal(0, self.score_noise), 0.0, 1.0))
            appearance = np.asarray(inst["appearance"], dtype=np.float32)
            candidates.append(
                SemanticCandidate(
                    mask=mask,
                    category=inst["category"],
                    category_score=score,
                    appearance=appearance,
                    native_track_id=None,
                )
            )

        return SemanticOutput(candidates=candidates, model_note=self.model_note)

    @property
    def model_note(self) -> str:
        return "MOCK semantic (synthetic annotations; not a real model output)"
