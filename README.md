# 流式语义实例建图（阶段 0–1 实现）

按 `EXECUTION_PLAN.md` 实现：冻结重建骨干与语义分支下，验证三维空间状态
是否比仅外观关联更准确地恢复历史实例身份。

## 当前状态（阶段 0–1 交付）

- 阶段 0：模型适配器接口 + 可行性检查结论（见下）。
- 阶段 1：因果缓存、实例观测提取、三种无训练关联模式、活跃/休眠记忆。
- 阶段 2 准备：合成场景、评测指标（身份切换/碎片化、按失踪时长分组的重
  关联率、错误合并率、漏检率）、评测脚本。

## 环境与运行

依赖：Python 3.10+、numpy、scipy（当前无 torch，核心逻辑不依赖它）。

```bash
# 测试（覆盖坐标变换/掩码对齐/空观测/无效几何/拒绝匹配/休眠恢复/序列重置/因果状态）
python -m unittest discover -s tests -v

# 端到端：缓存 -> 三种关联模式 -> 评测
python scripts/cache_sequence.py --out outputs/cache/orbit.npz --num-frames 40 --num-chairs 4
python scripts/run_sequence.py --cache outputs/cache/orbit.npz --out-dir outputs/run --mode all
python scripts/evaluate.py --cache outputs/cache/orbit.npz --gt outputs/cache/orbit_gt.npz --out-dir outputs/eval
```

## 阶段 0 可行性报告

环境检查结论（2026-10-12）：

| 检查项 | 结论 |
| --- | --- |
| PyTorch | 基础环境与 agentic-gts 环境均未安装 |
| SAM3 权重/仓库 | 不可用（需 `facebookresearch/sam3` 与权重） |
| 重建骨干权重（CUT3R/Spann3R） | 不可用 |
| numpy / scipy | 可用（核心逻辑已用其实现） |

因此第一版按计划执行规则 4 处理：**使用明确标识的 mock 适配器**
（`model_note` 含 "MOCK" 前缀）完成接口与逻辑验证，模拟输出不冒充模型
结果。真实模型适配器骨架位于 `src/adapters/sam3_semantic.py`，接入前必须
核对上游真实接口，不虚构接口。

### 完成真实模型实验所需条件（阻塞项）

1. PyTorch + CUDA 环境；
2. SAM3 权重与上游仓库（核对文本条件分割、局部特征可提取性）；
3. 流式重建骨干权重（建议 CUT3R 或 Spann3R 起步，核对逐帧因果接口、
   位姿/点图/置信度约定）；
4. 带实例身份真值的评测数据（合成场景已具备；真实数据建议 ScanNet++）。

## 目录结构

```
src/
  contracts.py      数据契约（唯一来源，禁止各模块自定义坐标格式）
  coords.py         坐标与预处理约定（T_world_from_camera 等）
  adapters/         适配器基类、mock 适配器、真实模型骨架
  observations.py   实例观测提取（表面中心/空间散布/质量）
  memory.py         活跃/休眠实例记忆（保守更新）
  association.py    三种无训练关联模式（Hungarian 一对一分配）
  cache.py          因果缓存（npz + 元数据）
  pipeline.py       关联核心与在线骨架
  data/synthetic.py 合成环视场景（五类场景覆盖）
  evaluation/       指标定义与计算（定义写于模块 docstring）
  visualization.py  叠加帧(ppm)/语义点云(ply) 输出
scripts/            cache_sequence / run_sequence / evaluate
tests/              16 项必要测试（全部通过）
configs/            默认超参（只在训练/验证集调整）
```

## 关键约定（与计划一致）

- 位姿为 `T_world_from_camera`，世界系由序列起始帧固定；
- 自定义关联不读取原生跟踪 ID；
- 表面中心不等于完整物体中心；空间散布不是中心不确定性；
- 无效几何用显式有效掩码，零坐标不冒充观测；
- 真值独立存放（`*_gt.npz`），关联逻辑不访问真值；
- 每段序列开始重置全部状态。

## 初步结果（合成场景，40 帧、4 把椅子、含遮挡离场与回环）

| 模式 | 身份切换 | 正确重关联 | 合并错误 | 漏检 |
| --- | --- | --- | --- | --- |
| appearance | 2 | 4/4 | 0/39 | 0/75 |
| geometry | 1 | 4/4 | 0/39 | 0/75 |
| appearance_geometry | 0 | 4/4 | 0/39 | 0/75 |

仅为小规模可行性证据，不能直接支撑泛化结论（计划第 6 节）。

## 已知问题与限制

- 纯外观模式会将长期缺席实例与同类相似实例跨时间合并（同帧合并指标
  无法捕捉，体现为身份切换）；
- 合成场景物体为平面矩形块，无真实外观变化；
- 在线模式（`OnlinePipeline`）要求真实适配器，当前显式报错。

## 下一阶段进入条件（阶段 3）

空间信息在诊断/预测几何实验中显示可解释的价值（当前合成实验初步满足：
加入空间项后身份切换单调下降），且真实模型接入完成后，进入学习型关联
与在线运行。
