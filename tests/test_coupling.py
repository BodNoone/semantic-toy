"""耦合层单元测试: Sim(3) 求解、漂移门控、重关联匹配。"""

from __future__ import annotations

import torch

from semantic_map.coupling.anchors import AnchorConfig, SemanticAnchorBank, observe_anchor
from semantic_map.coupling.drift import DriftConfig, estimate_drift, solve_similarity
from semantic_map.coupling.reassoc import (
    ReassocConfig,
    TrackIdRemapper,
    TrackReassociator,
)
from semantic_map.types import AnchorObservation, FrameGeometry, TrackState


def _geometry(points: torch.Tensor, confidence: torch.Tensor) -> FrameGeometry:
    return FrameGeometry(
        world_points=points,
        confidence=confidence,
        world_to_camera=torch.eye(4),
        intrinsics=torch.eye(3),
        frame_index=0,
    )


class TestSolveSimilarity:
    def test_recovers_pure_translation(self):
        source = torch.randn(10, 3)
        offset = torch.tensor([0.5, -0.2, 0.1])
        target = source + offset
        scale, rotation, translation = solve_similarity(
            source, target, allow_scale=False
        )
        assert abs(scale - 1.0) < 1e-6
        assert torch.allclose(rotation, torch.eye(3), atol=1e-5)
        assert torch.allclose(translation, offset, atol=1e-5)

    def test_recovers_rotation(self):
        angle = torch.tensor(0.3)
        rotation_gt = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        source = torch.randn(20, 3)
        target = source @ rotation_gt.t()
        _, rotation, translation = solve_similarity(
            source, target, allow_scale=False
        )
        assert torch.allclose(rotation, rotation_gt, atol=1e-4)
        assert translation.norm() < 1e-4


class TestEstimateDrift:
    def test_gate_rejects_insufficient_correspondences(self):
        config = DriftConfig(min_correspondences=3)
        correction = estimate_drift(
            torch.randn(2, 3), torch.randn(2, 3), config=config
        )
        assert not correction.applied
        assert "correspondences" in correction.reason

    def test_gate_rejects_large_residual(self):
        config = DriftConfig(max_residual=0.01)
        source = torch.randn(8, 3)
        target = torch.randn(8, 3)  # 无关点, 残差必然大
        correction = estimate_drift(source, target, config=config)
        assert not correction.applied

    def test_accepts_clean_translation(self):
        config = DriftConfig()
        source = torch.randn(8, 3)
        target = source + torch.tensor([0.3, 0.0, 0.0])
        correction = estimate_drift(source, target, config=config)
        assert correction.applied
        assert torch.allclose(
            correction.translation, torch.tensor([0.3, 0.0, 0.0]), atol=1e-4
        )


class TestObserveAnchor:
    def test_returns_none_for_few_points(self):
        points = torch.zeros(10, 10, 3)
        confidence = torch.ones(10, 10)
        mask = torch.zeros(10, 10, dtype=torch.bool)
        mask[0, 0] = True
        config = AnchorConfig(min_points=64)
        assert observe_anchor(_geometry(points, confidence), mask, config=config) is None

    def test_centroid_is_median(self):
        points = torch.zeros(20, 20, 3)
        points[..., 0] = 2.0
        confidence = torch.ones(20, 20)
        mask = torch.ones(20, 20, dtype=torch.bool)
        config = AnchorConfig(min_points=64)
        observation = observe_anchor(_geometry(points, confidence), mask, config=config)
        assert observation is not None
        assert torch.allclose(
            observation.centroid, torch.tensor([2.0, 0.0, 0.0]), atol=1e-6
        )


class TestReassociation:
    def _bank_with_lost_chair(self) -> SemanticAnchorBank:
        bank = SemanticAnchorBank(AnchorConfig())
        bank.tracks[0] = TrackState(
            track_id=0,
            category="chair",
            anchor=AnchorObservation(
                centroid=torch.tensor([1.0, 0.0, 3.0]),
                scale=0.3,
                count=100,
            ),
            last_seen=5,
        )
        return bank

    def test_matches_nearby_same_category(self):
        bank = self._bank_with_lost_chair()
        reassociator = TrackReassociator(ReassocConfig(min_lost_frames=3))
        events = reassociator.match_new_tracks(
            bank,
            {
                100: AnchorObservation(
                    centroid=torch.tensor([1.05, 0.0, 3.0]),
                    scale=0.3,
                    count=90,
                )
            },
            {100: "chair"},
            frame_index=10,
        )
        assert len(events) == 1
        assert events[0].matched_track_id == 0

    def test_rejects_different_category(self):
        bank = self._bank_with_lost_chair()
        reassociator = TrackReassociator(ReassocConfig(min_lost_frames=3))
        events = reassociator.match_new_tracks(
            bank,
            {
                100: AnchorObservation(
                    centroid=torch.tensor([1.05, 0.0, 3.0]),
                    scale=0.3,
                    count=90,
                )
            },
            {100: "desk"},
            frame_index=10,
        )
        assert events == []

    def test_rejects_far_object(self):
        bank = self._bank_with_lost_chair()
        reassociator = TrackReassociator(ReassocConfig(min_lost_frames=3))
        events = reassociator.match_new_tracks(
            bank,
            {
                100: AnchorObservation(
                    centroid=torch.tensor([5.0, 0.0, 3.0]),
                    scale=0.3,
                    count=90,
                )
            },
            {100: "chair"},
            frame_index=10,
        )
        assert events == []

    def test_not_lost_long_enough(self):
        bank = self._bank_with_lost_chair()
        reassociator = TrackReassociator(ReassocConfig(min_lost_frames=3))
        events = reassociator.match_new_tracks(
            bank,
            {
                100: AnchorObservation(
                    centroid=torch.tensor([1.05, 0.0, 3.0]),
                    scale=0.3,
                    count=90,
                )
            },
            {100: "chair"},
            frame_index=6,  # 只丢了 1 帧
        )
        assert events == []


class TestRemapper:
    def test_chain_merge(self):
        remapper = TrackIdRemapper()
        remapper.merge(100, 0)
        remapper.merge(200, 100)
        assert remapper.canonical(200) == 0
        assert remapper.canonical(100) == 0
        assert remapper.canonical(0) == 0
