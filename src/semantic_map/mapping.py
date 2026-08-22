"""增量体素化语义实例点云地图。

用固定体素栅格做下采样 (每体素保留最高置信度点), 支持流式写入与 PLY/PT 导出。
点携带实例 ID (规范后)、类别、置信度与 RGB(可选)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .types import FrameGeometry


@dataclass(frozen=True)
class MapConfig:
    voxel_size: float = 0.02  # 米
    confidence_threshold: float = 0.3
    store_rgb: bool = False


@dataclass
class _Cell:
    key: tuple[int, int, int]
    position: np.ndarray  # [3]
    confidence: float
    track_id: int
    category: str
    rgb: tuple[int, int, int] | None


class SemanticMapBuilder:
    def __init__(self, config: MapConfig) -> None:
        self.config = config
        self._cells: dict[tuple[int, int, int], _Cell] = {}
        self._voxel_size = float(config.voxel_size)

    def _key(self, position: np.ndarray) -> tuple[int, int, int]:
        return tuple((position / self._voxel_size).round().astype(np.int64).tolist())

    def add_frame(
        self,
        geometry: FrameGeometry,
        *,
        mask: torch.Tensor,  # [I,H,W] bool
        track_ids: torch.Tensor,  # [I] int (规范后)
        categories: list[str],
        rgb: torch.Tensor | None = None,  # [H,W,3] float in [0,1]
    ) -> int:
        """将一帧已校正几何中每个实例的点写入地图, 返回新增点数。"""

        added = 0
        for position in range(int(mask.shape[0])):
            track_id = int(track_ids[position])
            category = categories[position]
            valid = (
                mask[position].bool()
                & (geometry.confidence >= self.config.confidence_threshold)
                & torch.isfinite(geometry.world_points).all(dim=-1)
            )
            points = geometry.world_points[valid].float().cpu().numpy()
            conf = geometry.confidence[valid].float().cpu().numpy()
            if points.shape[0] == 0:
                continue
            rgbs = None
            if rgb is not None and self.config.store_rgb:
                rgbs = (rgb[valid].clamp(0, 1).mul(255).round().byte().cpu().numpy())
            for index in range(points.shape[0]):
                key = self._key(points[index])
                cell = self._cells.get(key)
                if cell is None or conf[index] > cell.confidence:
                    self._cells[key] = _Cell(
                        key=key,
                        position=points[index],
                        confidence=float(conf[index]),
                        track_id=track_id,
                        category=category,
                        rgb=tuple(rgbs[index].tolist()) if rgbs is not None else None,
                    )
                    added += 1
        return added

    def points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """返回 (positions [N,3], track_ids [N], confidence [N], categories)。"""

        cells = list(self._cells.values())
        positions = np.stack([c.position for c in cells])
        track_ids = np.array([c.track_id for c in cells], dtype=np.int64)
        confidence = np.array([c.confidence for c in cells], dtype=np.float32)
        categories = [c.category for c in cells]
        return positions, track_ids, confidence, categories

    def export_ply(self, path: str) -> None:
        positions, track_ids, confidence, categories = self.points()
        count = positions.shape[0]
        array = np.empty(
            count,
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("track_id", "<i4"), ("confidence", "<f4"),
            ],
        )
        array["x"], array["y"], array["z"] = positions.T
        array["track_id"] = track_ids.astype(np.int32)
        array["confidence"] = confidence
        header = "\n".join(
            (
                "ply", "format binary_little_endian 1.0", f"element vertex {count}",
                "property float x", "property float y", "property float z",
                "property int track_id", "property float confidence",
                "end_header", "",
            )
        ).encode("ascii")
        with open(path, "wb") as handle:
            handle.write(header)
            array.tofile(handle)

    def export_pt(self, path: str) -> None:
        positions, track_ids, confidence, categories = self.points()
        torch.save(
            {
                "positions": torch.from_numpy(positions),
                "track_ids": torch.from_numpy(track_ids),
                "confidence": torch.from_numpy(confidence),
                "categories": categories,
                "voxel_size": self._voxel_size,
            },
            path,
        )

    def __len__(self) -> int:
        return len(self._cells)
