"""SAM3 文本提示多目标视频跟踪适配器。

依赖 externals/sam3 仓库。使用 SAM3 video predictor 的文本提示 + 前向
传播模式: 对每个类别文本自动发现并跟踪所有实例, 输出持久 track ID。

由于 SAM3 的会话式 API 需要整段视频目录, 本适配器采用"预跑缓存"策略:
首次 next_frame 前对所有类别一次性运行 propagate, 结果按帧缓存后逐帧吐出。
(严格逐帧在线模式需接入 SAM3 streaming API, 接口预留在 _run_tracking 内。)

用法 (configs/real.yaml):
    tracking:
      repo_path: externals/sam3
      checkpoint_path: /path/to/sam3.pt
      image_dir: /path/to/frames
      categories: ["chair", "desk", "box"]
      device: cuda:1
      score_threshold: 0.5
      max_objects: 32
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

from ..types import FrameTracking


class SAM3TrackingEngine:
    def __init__(
        self,
        *,
        repo_path: str,
        checkpoint_path: str,
        image_dir: str,
        categories: Sequence[str],
        device: str = "cuda:1",
        score_threshold: float = 0.5,
        max_objects: int = 32,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self.categories = list(categories)
        self.device = device
        self.score_threshold = float(score_threshold)
        self.max_objects = int(max_objects)
        self.output_size = output_size
        self._image_dir = Path(image_dir).expanduser()
        self._frame = 0
        self._cache: list[FrameTracking] | None = None

        repo = Path(repo_path).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"SAM3 repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        self._checkpoint_path = str(checkpoint_path)
        self._predictor = None

    def reset(self) -> None:
        self._frame = 0

    def next_frame(self) -> FrameTracking:
        if self._cache is None:
            self._cache = self._run_tracking()
        if self._frame >= len(self._cache):
            raise StopIteration
        tracking = self._cache[self._frame]
        self._frame += 1
        return tracking

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from sam3.build_sam import build_sam3_video_predictor  # type: ignore

        self._predictor = build_sam3_video_predictor(
            checkpoint=self._checkpoint_path,
            device=self.device,
        )
        return self._predictor

    def _run_tracking(self) -> list[FrameTracking]:
        predictor = self._load_predictor()
        frame_paths = sorted(
            p for p in self._image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        num_frames = len(frame_paths)
        if num_frames == 0:
            raise FileNotFoundError(f"No frames in {self._image_dir}")

        # 逐类别运行文本提示跟踪, 合并到统一 slot 空间
        # track_key = (category_index, sam_obj_id) → 全局持久 ID
        per_frame: list[dict[int, tuple[torch.Tensor, float, str]]] = [
            {} for _ in range(num_frames)
        ]
        global_ids: dict[tuple[int, int], int] = {}
        next_id = 0

        for cat_index, category in enumerate(self.categories):
            session = predictor.start_session(resource_path=str(self._image_dir))
            session_id = getattr(session, "session_id", session)
            try:
                predictor.add_prompt(
                    session_id=session_id,
                    frame_idx=0,
                    text=category,
                )
                for response in predictor.propagate_in_video(
                    session_id=session_id,
                ):
                    frame_idx = int(response["frame_idx"])
                    obj_ids = response.get("obj_ids", [])
                    masks = response.get("masks")
                    scores = response.get("scores", [1.0] * len(obj_ids))
                    for position, obj_id in enumerate(obj_ids):
                        score = float(scores[position])
                        if score < self.score_threshold:
                            continue
                        key = (cat_index, int(obj_id))
                        if key not in global_ids:
                            if next_id >= self.max_objects:
                                continue
                            global_ids[key] = next_id
                            next_id += 1
                        mask = torch.as_tensor(masks[position]).bool()
                        if mask.ndim == 3:
                            mask = mask[0]
                        per_frame[frame_idx][global_ids[key]] = (
                            mask,
                            score,
                            category,
                        )
            finally:
                predictor.close_session(session_id=session_id)

        # 组装 FrameTracking
        output: list[FrameTracking] = []
        for frame_idx in range(num_frames):
            entries = per_frame[frame_idx]
            if entries:
                ids = sorted(entries.keys())
                masks = torch.stack([entries[i][0] for i in ids])
                scores = torch.tensor([entries[i][1] for i in ids])
                categories = [entries[i][2] for i in ids]
                if self.output_size is not None:
                    masks = _resize_masks(masks, self.output_size)
            else:
                height, width = self.output_size or (1, 1)
                masks = torch.zeros(0, height, width, dtype=torch.bool)
                scores = torch.zeros(0)
                ids = []
                categories = []
            output.append(
                FrameTracking(
                    masks=masks,
                    track_ids=torch.tensor(ids, dtype=torch.long),
                    scores=scores,
                    categories=categories,
                    frame_index=frame_idx,
                )
            )
        return output


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    return (
        F.interpolate(
            masks.float().unsqueeze(1),
            size=size,
            mode="nearest",
        )
        .squeeze(1)
        .bool()
    )
