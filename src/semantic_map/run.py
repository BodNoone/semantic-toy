"""CLI 入口: python -m semantic_map.run --config configs/mock.yaml"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import SystemConfig, load_config
from .pipeline import SemanticMappingPipeline


def build_engines(config: SystemConfig):
    if config.engine == "mock":
        from .engines.mock import (
            MockReconstructionEngine,
            MockSceneConfig,
            MockTrackingEngine,
        )

        scene = MockSceneConfig(**{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in config.reconstruction.items()
        })
        return MockReconstructionEngine(scene), MockTrackingEngine(scene), scene
    if config.engine == "real":
        from .engines.streamvggt import StreamVGGTEngine
        from .engines.sam3 import SAM3TrackingEngine

        reconstruction = StreamVGGTEngine(**config.reconstruction)
        tracking = SAM3TrackingEngine(**config.tracking)
        return reconstruction, tracking, None
    raise ValueError(f"Unknown engine {config.engine!r}; expected mock or real.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction, tracking, scene = build_engines(config)
    pipeline = SemanticMappingPipeline(reconstruction, tracking, config.pipeline)
    reports = pipeline.run(max_frames=config.max_frames)

    # 汇总
    corrections_applied = sum(
        1 for report in reports if report.correction and report.correction.applied
    )
    reassoc_total = sum(len(report.reassoc_events) for report in reports)
    summary = {
        "frames": len(reports),
        "corrections_applied": corrections_applied,
        "reassociation_events": reassoc_total,
        "map_points": len(pipeline.map_builder),
        "tracks": sorted(pipeline.bank.known_ids()),
    }

    # mock 模式下有 GT, 输出位姿误差对比
    if scene is not None and hasattr(reconstruction, "ground_truth_pose"):
        raw_err, corrected_err = _pose_errors(pipeline, reconstruction)
        summary["raw_pose_center_error_mean"] = raw_err
        summary["corrected_pose_center_error_mean"] = corrected_err
        if raw_err > 0:
            summary["center_error_reduction_percent"] = (
                100.0 * (raw_err - corrected_err) / raw_err
            )

    pipeline.map_builder.export_ply(str(output_dir / "semantic_map.ply"))
    pipeline.map_builder.export_pt(str(output_dir / "semantic_map.pt"))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"map -> {output_dir / 'semantic_map.ply'}")


def _pose_errors(pipeline: SemanticMappingPipeline, reconstruction) -> tuple[float, float]:
    def center(w2c: torch.Tensor) -> torch.Tensor:
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        return -(rotation.t() @ translation)

    raw_errors = []
    corrected_errors = []
    for frame_index, (raw, corrected) in enumerate(
        zip(pipeline.raw_poses, pipeline.corrected_poses)
    ):
        gt = reconstruction.ground_truth_pose(frame_index)
        gt_center = center(gt)
        raw_errors.append(float((center(raw) - gt_center).norm()))
        corrected_errors.append(float((center(corrected) - gt_center).norm()))
    mean = lambda values: sum(values) / max(len(values), 1)
    return mean(raw_errors), mean(corrected_errors)


if __name__ == "__main__":
    main()
