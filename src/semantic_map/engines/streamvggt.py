"""StreamVGGT 流式重建引擎适配器。

依赖 externals/streamvggt 仓库 (git submodule 或已安装包)。
逐帧因果推理, 维护 KV cache; 输出 FrameGeometry。

用法 (configs/real.yaml):
    reconstruction:
      repo_path: externals/streamvggt
      checkpoint_path: /path/to/checkpoints/StreamVGGT.pth
      image_dir: /path/to/frames        # 或 image_paths 列表
      device: cuda:0
      amp_dtype: bfloat16
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch

from ..types import FrameGeometry


class StreamVGGTEngine:
    def __init__(
        self,
        *,
        repo_path: str,
        checkpoint_path: str,
        image_dir: str | None = None,
        image_paths: Sequence[str] | None = None,
        device: str = "cuda:0",
        amp_dtype: str = "bfloat16",
        image_mode: str = "crop",
    ) -> None:
        self.device = device
        self.amp_dtype = _resolve_dtype(amp_dtype)
        self.image_mode = image_mode
        self._paths = _resolve_images(image_dir, image_paths)
        self._frame = 0

        repo = Path(repo_path).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"StreamVGGT repo not found: {repo}")
        for candidate in (repo / "src", repo):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))

        from streamvggt.models.streamvggt import StreamVGGT  # type: ignore

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        self.model = StreamVGGT()
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(device)

        self._past_key_values: list = [None] * self.model.aggregator.depth
        self._past_camera: list = [None] * self.model.camera_head.trunk_depth
        self._images: torch.Tensor | None = None

    def reset(self) -> None:
        self._frame = 0
        self._past_key_values = [None] * self.model.aggregator.depth
        self._past_camera = [None] * self.model.camera_head.trunk_depth
        self._images = None

    @torch.inference_mode()
    def next_frame(self) -> FrameGeometry:
        if self._frame >= len(self._paths):
            raise StopIteration
        if self._images is None:
            from streamvggt.utils.load_fn import load_and_preprocess_images  # type: ignore

            self._images = load_and_preprocess_images(
                [str(p) for p in self._paths],
                mode=self.image_mode,
            )

        frame_index = self._frame
        frame = self._images[frame_index : frame_index + 1].to(self.device)
        batch = frame.unsqueeze(0)  # [1,1,3,H,W]

        autocast = torch.autocast(device_type="cuda", dtype=self.amp_dtype) if (
            self.amp_dtype is not None and "cuda" in self.device
        ) else _null_context()
        with autocast:
            output = self.model.aggregator(
                batch,
                past_key_values=self._past_key_values,
                use_cache=True,
                past_frame_idx=frame_index,
            )
        if isinstance(output, tuple) and len(output) == 3:
            tokens_list, patch_start_idx, self._past_key_values = output
        else:
            tokens_list, patch_start_idx = output

        with torch.autocast(device_type="cuda", enabled=False):
            pose_list, self._past_camera = self.model.camera_head(
                tokens_list,
                past_key_values_camera=self._past_camera,
                use_cache=True,
            )
            pts3d, conf = self.model.point_head(
                tokens_list,
                images=batch,
                patch_start_idx=patch_start_idx,
            )

        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        image_hw = tuple(int(v) for v in frame.shape[-2:])
        w2c, intrinsics = pose_encoding_to_extri_intri(
            pose_list[-1].float(),
            image_size_hw=image_hw,
        )

        points = pts3d[0, 0].detach().float().cpu()  # [H,W,3]
        confidence = _normalize_confidence(conf[0, 0].detach().float().cpu())

        geometry = FrameGeometry(
            world_points=points,
            confidence=confidence,
            world_to_camera=_to_4x4(w2c[0, 0].detach().float().cpu()),
            intrinsics=intrinsics[0, 0].detach().float().cpu(),
            frame_index=frame_index,
        )
        self._frame += 1
        return geometry


def _resolve_images(
    image_dir: str | None,
    image_paths: Sequence[str] | None,
) -> list[Path]:
    if image_paths:
        return [Path(p) for p in image_paths]
    if image_dir:
        directory = Path(image_dir).expanduser()
        paths = sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not paths:
            raise FileNotFoundError(f"No images in {directory}")
        return paths
    raise ValueError("Provide image_dir or image_paths.")


def _resolve_dtype(value: str) -> torch.dtype | None:
    normalized = value.strip().lower()
    if normalized in {"float32", "fp32", "none"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unknown amp dtype {value!r}")


def _normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    if confidence.ndim == 3 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    flat = confidence.flatten()
    low = torch.quantile(flat, 0.05)
    high = torch.quantile(flat, 0.95)
    return ((confidence - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def _to_4x4(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        output = torch.eye(4, dtype=matrix.dtype)
        output[:3, :] = matrix
        return output
    raise ValueError(f"Unexpected extrinsic shape {tuple(matrix.shape)}")


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
