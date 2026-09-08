# Formal v1：五中心留一验证实验矩阵

## 锁定队列

正式队列使用 984 个同时具备阴道镜和 OCT 的检查位点，每位患者仅一个位点。外层按医院逐中心留一（LOCO），源中心内部以固定哈希拆出 20% 验证集。测试中心保持自然患病率，不参与训练、checkpoint 选择、阈值选择或校准。

| 留出中心 | 训练 | 验证 | 测试 | 测试阳性率 |
|---|---:|---:|---:|---:|
| 十堰 | 725 | 182 | 77 | 7.79% |
| 恩施 | 464 | 116 | 404 | 31.93% |
| 武汉 | 716 | 179 | 89 | 91.01% |
| 荆州 | 731 | 183 | 70 | 8.57% |
| 襄阳 | 512 | 128 | 344 | 13.08% |

武汉与其他中心的患病率方向相反，因此仅报告随机同中心划分会产生严重误导。正式结论以五个外部中心 AUPRC 的宏平均为主要估计量，同时给出逐中心及全体患者汇总结果。

## 五个主要比较方法

1. `clinical_only_lr`：年龄、HPV、TCT 的 class-balanced logistic regression。
2. `frozen_multimodal_linear`：既有冻结阴道镜/OCT 特征拼接后的线性分类器。
3. `qwen3vl_zero_shot`：未经本项目训练的 Qwen3-VL-2B-Instruct。
4. `diagnosis_only_sft`：只学习最终诊断，训练预算与主方法相同。
5. `hierarchical_sft`：临床背景 → 阴道镜形态 → OCT 微结构 → 跨模态整合 → 诊断的完整分层 SFT；实现名为 `full_hierarchy`。

主方法和 diagnosis-only SFT 分别运行 3 个预先锁定的随机种子；正式主表采用三个模型的验证/测试概率平均，并在平均后的源中心验证概率上重新锁定阈值。前三个固定对照不通过重复随机种子制造伪重复。

SFT 使用 Qwen3-VL-2B-Instruct、LoRA rank 16/alpha 32、所有线性层、学习率 `1e-4`、cosine、最大长度 2048，并冻结视觉编码器和 aligner。这些关键设置与参考论文一致。受两张 48 GB GPU 和本队列规模约束，本试验采用 3 epochs、单卡 batch 4、梯度累积 2（有效 batch 8），而不是论文八卡的固定 400 steps/global batch 128；因此是资源适配迁移，不是计算预算完全相同的复现。

## 三个消融板块

消融统一使用种子 `20260902`，并与同一种子的 `full_hierarchy` 配对比较。

| 板块 | 变体 1 | 变体 2 | 完整参照 | 回答的问题 |
|---|---|---|---|---|
| 认知结构 | diagnosis-only | without-integration | full-hierarchy | 分层监督及跨模态整合文本是否有贡献 |
| 模态贡献 | colposcopy-only | OCT-only | full-multimodal | 收益来自哪一种图像模态 |
| 监督完整性 | without-clinical | shuffled-cognition | full-hierarchy | 临床上下文和配对认知目标是否真实有效 |

`full-multimodal` 是 `full_hierarchy` 在模态消融板块中的语义别名，不是额外训练任务。因此整个训练矩阵为 30 个主要 SFT 任务加 25 个消融任务，共 55 个任务。

## 统计规则

- 主要指标：五中心宏平均 AUPRC。
- 次要指标：AUROC、balanced accuracy、灵敏度、特异度、F1、Brier、ECE、格式通过率和认知字段得分。
- 阈值：逐折只由源中心验证集按 balanced accuracy 选择。
- 不确定性：在中心和结局分层内进行患者聚类 bootstrap，5,000 次；本队列每位患者一个位点，仍保留患者标识以防后续扩展时破坏聚类单位。
- 方法差异：同一测试患者的配对 bootstrap，报告候选方法减参照方法的点估计、95% CI 与双侧 bootstrap p 值。
- 多重比较：每个比较家族的宏平均 AUPRC 使用 Holm 校正。
- 解析门槛：诊断概率覆盖率低于 95% 的运行不进入有效正式结论。

## 当前不能开展的板块

论文式病灶擦除 GRPO 仍不属于已开启的 55 个正式任务。项目没有经临床复核的 mask、polygon 或 bbox；用图像中心框或自动显著性区域冒充病灶会制造虚假的因果监督。收到带标注者、版本和坐标的复核 ROI 后，该板块才能解锁。当前正式实验检验的是分层认知 SFT，而不是完整 CogAlign 的反事实强化学习效果。
