"""YAML 配置加载。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .coupling.anchors import AnchorConfig
from .coupling.drift import DriftConfig
from .coupling.reassoc import ReassocConfig
from .mapping import MapConfig
from .pipeline import PipelineConfig


@dataclass(frozen=True)
class SystemConfig:
    pipeline: PipelineConfig
    engine: str  # "mock" | "real"
    reconstruction: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "outputs"
    max_frames: int | None = None


def load_config(path: str | Path) -> SystemConfig:
    path = Path(path)
    with path.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}

    anchor = AnchorConfig(**dict(raw.get("anchor", {})))
    drift = DriftConfig(**dict(raw.get("drift", {})))
    reassoc = ReassocConfig(**dict(raw.get("reassoc", {})))
    mapping = MapConfig(**dict(raw.get("map", {})))
    pipeline_raw = raw.get("pipeline", {})
    pipeline = PipelineConfig(
        anchor=anchor,
        drift=drift,
        reassoc=reassoc,
        map=mapping,
        enable_drift_correction=bool(
            pipeline_raw.get("enable_drift_correction", True)
        ),
        enable_reassociation=bool(
            pipeline_raw.get("enable_reassociation", True)
        ),
    )
    return SystemConfig(
        pipeline=pipeline,
        engine=str(raw.get("engine", "mock")).strip().lower(),
        reconstruction=dict(raw.get("reconstruction", {})),
        tracking=dict(raw.get("tracking", {})),
        output_dir=str(raw.get("output_dir", "outputs")),
        max_frames=raw.get("max_frames"),
    )
