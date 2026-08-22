"""Mock 引擎: 合成场景, 用于无 GPU 端到端验证耦合逻辑。

场景: 若干"物体"(3D 高斯点团)散布在房间内, 相机绕场景平移。
- MockReconstructionEngine 输出每帧稠密 pointmap, 并注入**逐帧累积的
  平移漂移** (模拟流式重建的漂移)。
- MockTrackingEngine 输出每个物体的 mask 与 track ID, 并在指定帧注入
  **ID 切换** (模拟 SAM 长时丢失后重新分配 ID)。

期望行为:
- 漂移校正开启时, corrected pose/pointmap 相对 GT 的误差远小于 raw;
- 重关联开启时, 切换后的新 ID 被合并回原 ID, 地图中同一物体只有一个实例。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..types import FrameGeometry, FrameTracking


@dataclass(frozen=True)
class MockObject:
    center: tuple[float, float, float]
    radius: float
    category: str
    track_id: int


@dataclass(frozen=True)
class MockSceneConfig:
    image_size: tuple[int, int] = (60, 80)  # (H, W)
    num_frames: int = 30
    objects: tuple[MockObject, ...] = (
        MockObject(center=(-1.0, 0.0, 3.0), radius=0.3, category="chair", track_id=0),
        MockObject(center=(1.0, 0.2, 3.5), radius=0.35, category="desk", track_id=1),
        MockObject(center=(0.0, -0.5, 4.0), radius=0.25, category="box", track_id=2),
    )
    # 漂移: 每帧沿 x 平移 drift_per_frame 米 (累积)
    drift_per_frame: float = 0.02
    # ID 切换: 在 switch_frame 帧起, victim_track 的 ID 变为 new_id,
    # 且切换前 lost_frames 帧内该物体不可见 (模拟丢失)
    switch_frame: int = 15
    switch_victim: int = 0
    switch_new_id: int = 100
    lost_frames: int = 4
    seed: int = 0


def _camera_pose(frame: int, num_frames: int) -> torch.Tensor:
    """相机沿 x 缓慢平移的简单轨迹 (world_to_camera)。"""

    t = frame / max(num_frames - 1, 1)
    cam_x = -0.5 + 1.0 * t
    w2c = torch.eye(4)
    w2c[0, 3] = -cam_x  # world→camera: 相机在 x=cam_x 处
    return w2c


def _render_scene(
    config: MockSceneConfig,
    frame: int,
    *,
    drift_offset: torch.Tensor,  # [3] 注入到输出点云的漂移
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """返回 (world_points [H,W,3], confidence [H,W], obj_masks {track_id: [H,W]})。

    简化渲染: 图像划分为若干水平条带, 每个物体占一个条带区域,
    条带内的点从该物体的高斯团采样; 背景为远平面墙点。
    """

    height, width = config.image_size
    generator = torch.Generator().manual_seed(config.seed * 100003 + frame)
    # 背景: z=6 的墙
    xs = torch.linspace(-2.0, 2.0, width)
    ys = torch.linspace(-1.5, 1.5, height)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack(
        [grid_x, grid_y, torch.full_like(grid_x, 6.0)], dim=-1
    )
    confidence = torch.full((height, width), 0.6)
    masks: dict[int, torch.Tensor] = {}

    band_height = height // max(len(config.objects), 1)
    for index, obj in enumerate(config.objects):
        row0 = index * band_height + band_height // 4
        row1 = index * band_height + (3 * band_height) // 4
        col0 = width // 4
        col1 = (3 * width) // 4
        mask = torch.zeros(height, width, dtype=torch.bool)
        mask[row0:row1, col0:col1] = True
        count = int(mask.sum())
        center = torch.tensor(obj.center)
        samples = center + obj.radius * 0.5 * torch.randn(
            count, 3, generator=generator
        )
        points[mask] = samples
        confidence[mask] = 0.9
        masks[obj.track_id] = mask

    points = points + drift_offset  # 漂移注入: 整帧点云平移
    return points, confidence, masks


class MockReconstructionEngine:
    def __init__(self, config: MockSceneConfig) -> None:
        self.config = config
        self._frame = 0

    def reset(self) -> None:
        self._frame = 0

    def next_frame(self) -> FrameGeometry:
        if self._frame >= self.config.num_frames:
            raise StopIteration
        frame = self._frame
        drift = torch.tensor(
            [self.config.drift_per_frame * frame, 0.0, 0.0]
        )
        points, confidence, _ = _render_scene(self.config, frame, drift_offset=drift)
        w2c = _camera_pose(frame, self.config.num_frames)
        # 漂移同样污染位姿 (world 坐标整体被平移了 drift)
        drifted_w2c = w2c.clone()
        drifted_w2c[:3, 3] = w2c[:3, 3] - w2c[:3, :3] @ drift
        geometry = FrameGeometry(
            world_points=points,
            confidence=confidence,
            world_to_camera=drifted_w2c,
            intrinsics=torch.eye(3),
            frame_index=frame,
        )
        self._frame += 1
        return geometry

    def ground_truth_pose(self, frame: int) -> torch.Tensor:
        return _camera_pose(frame, self.config.num_frames)


class MockTrackingEngine:
    categories: list[str]

    def __init__(self, config: MockSceneConfig) -> None:
        self.config = config
        self.categories = sorted({obj.category for obj in config.objects})
        self._frame = 0

    def reset(self) -> None:
        self._frame = 0

    def next_frame(self) -> FrameTracking:
        if self._frame >= self.config.num_frames:
            raise StopIteration
        frame = self._frame
        # 注意: tracking 分支观测的是同一被漂移污染的场景,
        # 但 mask 是图像域概念, 与漂移无关, 直接用渲染 mask。
        _, _, masks = _render_scene(
            self.config,
            frame,
            drift_offset=torch.zeros(3),
        )
        mask_list = []
        id_list = []
        category_list = []
        for obj in self.config.objects:
            visible = True
            track_id = obj.track_id
            if obj.track_id == self.config.switch_victim:
                lost_start = self.config.switch_frame - self.config.lost_frames
                if lost_start <= frame < self.config.switch_frame:
                    visible = False  # 丢失期
                elif frame >= self.config.switch_frame:
                    track_id = self.config.switch_new_id  # ID 切换
            if not visible:
                continue
            mask_list.append(masks[obj.track_id])
            id_list.append(track_id)
            category_list.append(obj.category)
        tracking = FrameTracking(
            masks=torch.stack(mask_list) if mask_list else torch.zeros(
                0, *self.config.image_size, dtype=torch.bool
            ),
            track_ids=torch.tensor(id_list, dtype=torch.long),
            scores=torch.ones(len(id_list)),
            categories=category_list,
            frame_index=frame,
        )
        self._frame += 1
        return tracking
