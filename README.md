# semantic-map: 端到端流式语义地图系统

双分支流式架构：

- **分支A 流式重建**（StreamVGGT）：逐帧输出 pointmap / confidence / camera pose
- **分支B 语义跟踪**（SAM3 文本提示视频跟踪）：逐帧输出实例 mask / 持久 track ID / score
- **耦合层**（免训练在线优化）：
  - 语义→几何：以每个 track 的 3D 语义锚点为跨帧一致性约束，鲁棒求解 Sim(3) 校正当前帧漂移（带门控，失败即回退 raw）
  - 几何→语义：新 track 出生时与失联 track 做"类别+3D 锚点"匹配，几何一致则合并 ID，修复跟踪错乱
- **输出**：增量体素化语义实例点云地图（.ply / .pt，含实例 ID、类别、置信度）

## 快速开始（mock 模式，无 GPU）

```bash
pip install -e .
python -m semantic_map.run --config configs/mock.yaml
```

mock 引擎生成一个合成房间场景（含注入的位姿漂移与 track ID 切换），
用于端到端验证耦合逻辑：漂移被语义锚点校正、切换的 ID 被几何重关联。

## 真实模型模式

```bash
# 需要 externals/ 下的 StreamVGGT 与 SAM3 仓库及权重, 见 configs/real.yaml
python -m semantic_map.run --config configs/real.yaml
```

## 结构

```
src/semantic_map/
├── types.py            # 帧级数据结构 (FrameGeometry / FrameTracking / ...)
├── engines/
│   ├── base.py         # ReconstructionEngine / TrackingEngine Protocol
│   ├── mock.py         # 合成场景引擎 (漂移注入 + ID 切换注入)
│   ├── streamvggt.py   # StreamVGGT 流式适配器
│   └── sam3.py         # SAM3 文本提示跟踪适配器
├── coupling/
│   ├── anchors.py      # SemanticAnchorBank: track → 3D 锚点 (鲁棒质心+EMA)
│   ├── drift.py        # DriftCorrector: 锚点对应 → 鲁棒 Sim(3) → 门控校正
│   └── reassoc.py      # TrackReassociator: 几何辅助 track 重关联
├── mapping.py          # 增量体素语义地图 + PLY/PT 导出
├── pipeline.py         # 主流水线 (逐帧: 重建→跟踪→耦合→建图)
├── config.py           # YAML 配置
└── run.py              # CLI 入口
```

## 测试

```bash
pytest tests -v
```
