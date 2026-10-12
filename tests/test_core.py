"""必要测试（EXECUTION_PLAN.md 第 11 节）。

覆盖：坐标变换、掩码对齐、空观测、无效几何、拒绝匹配、休眠后恢复、
序列重置和因果状态更新。运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from src.contracts import (
    AssociationMode,
    FrameInput,
    GeometryOutput,
    InstanceObservation,
    InstanceStatus,
    SemanticCandidate,
    SemanticOutput,
)
from src.coords import (
    adjust_intrinsics_for_crop,
    adjust_intrinsics_for_padding,
    adjust_intrinsics_for_resize,
    check_mask_image_alignment,
    compose_poses,
    invert_pose,
    transform_points,
)
from src.adapters import MockGeometryAdapter, MockSemanticAdapter
from src.observations import extract_observations
from src.memory import InstanceMemoryBank
from src.association import Associator
from src.pipeline import associate_frame, run_association_from_cache
from src.cache import build_cache
from src.data.synthetic import generate_orbit_scene


def make_pose(tvec):
    T = np.eye(4)
    T[:3, 3] = tvec
    return T


class TestCoordinates(unittest.TestCase):
    def test_transform_roundtrip(self):
        pose = make_pose([1.0, 2.0, 3.0])
        pts_cam = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pts_w = transform_points(pose, pts_cam)
        self.assertTrue(np.allclose(pts_w[0], [1, 2, 3]))
        back = transform_points(invert_pose(pose), pts_w)
        self.assertTrue(np.allclose(back, pts_cam))

    def test_compose(self):
        a = make_pose([1, 0, 0])
        b = make_pose([0, 1, 0])
        c = compose_poses(a, b)
        self.assertTrue(np.allclose(c[:3, 3], [1, 1, 0]))

    def test_intrinsics_adjust(self):
        K = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
        K2 = adjust_intrinsics_for_resize(K, (100, 80), (200, 160))
        self.assertAlmostEqual(K2[0, 0], 200.0)
        K3 = adjust_intrinsics_for_crop(K, (10, 5))
        self.assertAlmostEqual(K3[0, 2], 40.0)
        K4 = adjust_intrinsics_for_padding(K, (3, 2))
        self.assertAlmostEqual(K4[1, 2], 42.0)

    def test_mask_alignment(self):
        img = np.zeros((10, 12, 3), dtype=np.uint8)
        self.assertTrue(check_mask_image_alignment(np.zeros((10, 12), bool), img))
        self.assertFalse(check_mask_image_alignment(np.zeros((5, 5), bool), img))


def make_geo(h=8, w=8, pts_value=1.0, valid=True, conf=0.9):
    pm = np.full((h, w, 3), pts_value, dtype=np.float32)
    valid_mask = np.full((h, w), valid, dtype=bool)
    return GeometryOutput(
        pose_w_from_c=np.eye(4),
        pointmap_world=pm,
        valid=valid_mask,
        confidence=np.full((h, w), conf, dtype=np.float32),
    )


def make_obs(h=8, w=8, region=(slice(0, 4), slice(0, 4)), cat="chair", score=0.9,
             appearance=None, with_geometry=True):
    mask = np.zeros((h, w), dtype=bool)
    mask[region] = True
    geo = make_geo(h, w, valid=with_geometry)
    sem = SemanticOutput(candidates=[SemanticCandidate(
        mask=mask, category=cat, category_score=score,
        appearance=appearance if appearance is not None else np.ones(8, np.float32) / np.sqrt(8),
    )])
    obs = extract_observations(sem, geo)
    return obs[0], geo, sem


class TestObservations(unittest.TestCase):
    def test_empty_candidates(self):
        geo = make_geo()
        obs = extract_observations(SemanticOutput(candidates=[]), geo)
        self.assertEqual(obs, [])

    def test_invalid_geometry_no_nan(self):
        obs, _, _ = make_obs(with_geometry=False)
        self.assertFalse(obs.has_valid_geometry)
        self.assertIsNone(obs.surface_center)
        self.assertEqual(obs.valid_point_ratio, 0.0)

    def test_zero_points_not_valid(self):
        geo = make_geo(pts_value=0.0)
        sem = SemanticOutput(candidates=[SemanticCandidate(
            mask=np.ones((8, 8), bool), category="chair", category_score=0.9,
            appearance=np.ones(8, np.float32) / np.sqrt(8))])
        obs = extract_observations(sem, geo)[0]
        self.assertFalse(obs.has_valid_geometry)

    def test_surface_center(self):
        obs, _, _ = make_obs()
        self.assertTrue(obs.has_valid_geometry)
        self.assertTrue(np.allclose(obs.surface_center, [1.0, 1.0, 1.0]))


class TestAssociation(unittest.TestCase):
    def test_reject_low_score(self):
        obs, _, _ = make_obs(score=0.1)
        bank = InstanceMemoryBank()
        assoc = Associator(mode=AssociationMode.APPEARANCE)
        res = associate_frame([obs], bank, assoc, 0)
        self.assertIn(0, res.rejected_candidates)
        self.assertEqual(bank.count(), 0)

    def test_create_then_match(self):
        obs1, _, _ = make_obs()
        bank = InstanceMemoryBank()
        assoc = Associator(mode=AssociationMode.APPEARANCE)
        res1 = associate_frame([obs1], bank, assoc, 0)
        self.assertEqual(bank.count(), 1)
        pid = res1.assignment[0]
        obs2, _, _ = make_obs()
        res2 = associate_frame([obs2], bank, assoc, 1)
        self.assertEqual(res2.assignment[0], pid)
        self.assertEqual(bank.count(), 1)

    def test_category_incompatible(self):
        obs1, _, _ = make_obs(cat="chair")
        obs2, _, _ = make_obs(cat="table", region=(slice(0, 4), slice(0, 4)))
        bank = InstanceMemoryBank()
        assoc = Associator(mode=AssociationMode.APPEARANCE_GEOMETRY)
        associate_frame([obs1], bank, assoc, 0)
        res2 = associate_frame([obs2], bank, assoc, 1)
        # 不同类别必须新建实例，不合并
        self.assertEqual(bank.count(), 2)


class TestMemory(unittest.TestCase):
    def test_dormant_and_recover(self):
        obs, _, _ = make_obs()
        bank = InstanceMemoryBank(dormant_grace_frames=2)
        assoc = Associator(mode=AssociationMode.APPEARANCE)
        res = associate_frame([obs], bank, assoc, 0)
        pid = res.assignment[0]
        # 连续空帧超过宽限期 -> dormant
        for t in range(1, 5):
            associate_frame([], bank, assoc, t)
        self.assertEqual(bank.get(pid).status, InstanceStatus.DORMANT)
        # 再次可见 -> 恢复为 active
        obs2, _, _ = make_obs()
        res2 = associate_frame([obs2], bank, assoc, 5)
        self.assertEqual(res2.assignment[0], pid)
        self.assertEqual(bank.get(pid).status, InstanceStatus.ACTIVE)

    def test_low_quality_no_overwrite(self):
        obs, _, _ = make_obs(score=0.9)
        bank = InstanceMemoryBank(min_quality_for_spatial_update=0.5)
        assoc = Associator(mode=AssociationMode.APPEARANCE)
        res = associate_frame([obs], bank, assoc, 0)
        pid = res.assignment[0]
        center_before = bank.get(pid).spatial_center.copy()
        # 低质量观测（质量分数低于阈值）不应覆盖空间状态
        low = InstanceObservation(
            candidate_index=0, mask=obs.mask, category="chair",
            category_score=0.2, appearance=obs.appearance,
            has_valid_geometry=True,
            surface_center=np.array([99.0, 99.0, 99.0], dtype=np.float32),
            spatial_spread=np.eye(3, dtype=np.float32),
            quality=0.1, valid_point_ratio=0.5, area=16,
        )
        bank.update_instance(pid, low, 1)
        self.assertTrue(np.allclose(bank.get(pid).spatial_center, center_before))

    def test_sequence_reset(self):
        obs, _, _ = make_obs()
        bank = InstanceMemoryBank()
        assoc = Associator(mode=AssociationMode.APPEARANCE)
        associate_frame([obs], bank, assoc, 0)
        self.assertEqual(bank.count(), 1)
        bank.reset()
        self.assertEqual(bank.count(), 0)


class TestCausalCacheAndPipeline(unittest.TestCase):
    def test_cache_roundtrip_and_three_modes(self):
        scene = generate_orbit_scene(num_frames=12, num_chairs=3, seed=1)
        cache = build_cache(
            scene.frames, MockGeometryAdapter(), MockSemanticAdapter(),
            scene.metadata, scene.text_queries,
        )
        self.assertEqual(cache.num_frames, 12)
        for mode in AssociationMode:
            results, bank, decisions = run_association_from_cache(cache, mode)
            self.assertEqual(len(results), 12)
            self.assertGreaterEqual(bank.count(), 1)

    def test_mock_geometry_failure_frame(self):
        """因果状态更新：失败帧输出全无效几何，后续帧恢复正常。"""
        frames = []
        for t in range(3):
            extra = {}
            if t != 1:
                extra = {
                    "pose_w_from_c": np.eye(4),
                    "pointmap_world": np.ones((4, 4, 3), np.float32),
                    "valid_mask": np.ones((4, 4), bool),
                    "instances": [],
                }
            frames.append(FrameInput(
                image=np.zeros((4, 4, 3), np.uint8), frame_index=t,
                timestamp=float(t), text_queries=[], extra=extra))
        adapter = MockGeometryAdapter()
        adapter.reset()
        geo0, st = adapter.step(frames[0], None)
        geo1, st = adapter.step(frames[1], st)
        geo2, st = adapter.step(frames[2], st)
        self.assertTrue(geo0.valid.all())
        self.assertFalse(geo1.valid.any())
        self.assertTrue(geo2.valid.all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
