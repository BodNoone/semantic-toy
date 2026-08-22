"""端到端测试: mock 场景下验证漂移校正与重关联。"""

from __future__ import annotations

import torch

from semantic_map.config import load_config
from semantic_map.pipeline import SemanticMappingPipeline
from semantic_map.engines.mock import (
    MockReconstructionEngine,
    MockSceneConfig,
    MockTrackingEngine,
)


def _build_pipeline(*, drift: bool, reassoc: bool, id_switch: bool = True):
    scene = MockSceneConfig() if id_switch else MockSceneConfig(switch_frame=10_000)
    reconstruction = MockReconstructionEngine(scene)
    tracking = MockTrackingEngine(scene)
    config = load_config("configs/mock.yaml")
    config = _with_toggles(config, drift=drift, reassoc=reassoc)
    pipeline = SemanticMappingPipeline(reconstruction, tracking, config.pipeline)
    return pipeline, reconstruction, scene


def _with_toggles(config, *, drift, reassoc):
    from dataclasses import replace

    return replace(
        config,
        pipeline=replace(
            config.pipeline,
            enable_drift_correction=drift,
            enable_reassociation=reassoc,
        ),
    )


def _mean_center_error(pipeline, reconstruction):
    def center(w2c):
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        return -(rotation.t() @ translation)

    errors = []
    for frame_index, corrected in enumerate(pipeline.corrected_poses):
        gt = reconstruction.ground_truth_pose(frame_index)
        errors.append(float((center(corrected) - center(gt)).norm()))
    return sum(errors) / len(errors)


class TestEndToEnd:
    def test_drift_correction_reduces_error(self):
        # 无 ID 切换的干净场景: 隔离验证漂移校正本身
        pipeline_on, reconstruction, _ = _build_pipeline(
            drift=True, reassoc=False, id_switch=False
        )
        pipeline_on.run()

        pipeline_off, reconstruction, _ = _build_pipeline(
            drift=False, reassoc=False, id_switch=False
        )
        pipeline_off.run()

        error_on = _mean_center_error(pipeline_on, reconstruction)
        error_off = _mean_center_error(pipeline_off, reconstruction)
        assert error_on < error_off * 0.5, (
            f"drift correction should cut error substantially: "
            f"on={error_on:.4f} off={error_off:.4f}"
        )

    def test_reassociation_rescues_drift_correction(self):
        # ID 切换场景: 重关联恢复第 3 个对应点, 漂移校正得以继续 →
        # 开启 reassoc 的位姿误差应低于关闭 reassoc
        pipeline_with, reconstruction, _ = _build_pipeline(drift=True, reassoc=True)
        pipeline_with.run()

        pipeline_without, reconstruction, _ = _build_pipeline(drift=True, reassoc=False)
        pipeline_without.run()

        error_with = _mean_center_error(pipeline_with, reconstruction)
        error_without = _mean_center_error(pipeline_without, reconstruction)
        assert error_with < error_without, (
            f"reassociation should rescue drift correction: "
            f"with={error_with:.4f} without={error_without:.4f}"
        )

    def test_reassociation_merges_switched_id(self):
        pipeline, _, _ = _build_pipeline(drift=True, reassoc=True)
        pipeline.run()
        # 切换后的新 ID 100 应被合并回原 ID 0
        canonical = pipeline.remapper.canonical(100)
        assert canonical == 0
        # 地图中的 track 集合应只含 0/1/2, 不含 100
        _, track_ids, _, _ = pipeline.map_builder.points()
        assert 100 not in set(track_ids.tolist())

    def test_reassociation_disabled_keeps_new_id(self):
        pipeline, _, _ = _build_pipeline(drift=True, reassoc=False)
        pipeline.run()
        _, track_ids, _, _ = pipeline.map_builder.points()
        assert 100 in set(track_ids.tolist())

    def test_runs_without_crash_both_off(self):
        pipeline, _, _ = _build_pipeline(drift=False, reassoc=False)
        reports = pipeline.run()
        assert len(reports) == 30
        assert len(pipeline.map_builder) > 0
