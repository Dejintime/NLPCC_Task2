# 实验记录

## 数据集简介

- **训练集**：3,520 条 / **验证集**：514 条
- **标签**：19 类 Schwartz 基本价值观
- **输入格式**：`Scenario [SEP] Question [SEP] Consistent Value Response`
- **类别不均衡**：最少 68 条（Universalism–tolerance），最多 400 条（Stimulation），极差约 6 倍

| 类别 | 训练集 | 类别 | 训练集 |
|------|--------|------|--------|
| Self-direction–thought | 119 | Tradition | 90 |
| Self-direction–action | 124 | Conformity–rules | 385 |
| Stimulation | 400 | Conformity–interpersonal | 236 |
| Hedonism | 164 | Humility | 100 |
| Achievement | 174 | Benevolence–dependability | 189 |
| Power–dominance | 156 | Benevolence–caring | 317 |
| Power–resources | 237 | Universalism–concern | 160 |
| Face | 258 | Universalism–nature | 71 |
| Security–personal | 202 | Universalism–tolerance | 68 |
| Security–societal | 70 | | |

## Track 1 
### Track 1实验总览

| Exp | 类型 | 模型 | 框架 | LR | Epochs | Batch | Max Len | LoRA r/α | Dev Accuracy | Dev Macro F1 | 备注 |
|-----|------|------|----|--------|----------|------|---------|-------------|-------------|------|------|
| 01 | Encoder | deberta-v3-large | Unsloth + WeightedTrainer | 5e-5 | 15 | 16 | 512 | 16/32 | **91.63%** | **0.9027** | 使用类别加权交叉熵，缓解类别不均衡 |
| 02 | Encoder | bert-base-uncased | Transformers + R-Drop | 5e-5 | 5 | 4 | 512 | - | **93.39%** | **0.9225** | 双前向一致性正则，当前最优 |

### 技术路线
#### Encoder 分类（DeBERTa-v3-large + LoRA）

1. **数据构造** — 将 `Scenario`、`Question`、`Consistent Value Response` 用 ` [SEP] ` 拼接为单条文本，映射 19 类标签为分类 ID。

2. **模型选择** — 采用 **`microsoft/deberta-v3-large`** 作为 backbone，接 `AutoModelForSequenceClassification` 做 19 分类。因 DeBERTa-v3 不支持默认 `sdpa`，**必须设置 `attn_implementation="eager"`**。使用 `bfloat16` 在单卡 RTX 3090 上训练。

3. **参数高效微调** — 使用 **Unsloth `FastModel`** 加载并 patch 模型，挂载 **LoRA**（`r=16, α=32`），仅训练 **0.72%** 参数（3.17M / 438M），大幅降低显存开销。

4. **类别不均衡处理（关键）** — 类别样本数极差约 6 倍（68~400）。通过 `compute_class_weight("balanced")` 计算类别权重，自定义 **`WeightedTrainer`** 在 `cross_entropy` 中加权，**显著提升小样本类别 F1**。

5. **训练设置** — `lr=5e-5`，`epochs=15`，`batch=16`，`max_len=512`。优化器 `adamw_torch` + `linear` 调度 + `warmup_ratio=0.1`。每 epoch 评估，按 **macro_f1** 择优保存。

6. **评估** — 主指标 **Dev Accuracy 91.63%** / **Dev Macro F1 0.9027**，同时输出每类 precision/recall/F1 及混淆矩阵。

#### Encoder 分类（BERT-base + R-Drop）

1. **模型结构** — 采用 **`bert-base-uncased`** 作为 Encoder backbone，自定义 `BertScratch`，在 `BertModel` 后接 dropout 与线性分类头完成 19 分类。

2. **R-Drop 正则（关键）** — 同一批输入执行 **两次带 dropout 的前向传播**，分别得到 `logits` 与 `kl_logits`，利用 dropout 随机性构造两个预测分布。

3. **损失函数设计** — 总损失为 **两次交叉熵损失 + 双向 KL 散度一致性损失**：`loss + ce_loss + kl_loss`，其中 `kl_loss` 约束两次前向输出分布保持一致，提升模型泛化能力。

4. **训练设置** — 使用 Transformers `Trainer` 全量微调 **109.50M** 参数，`lr=5e-5`，`epochs=5`，训练 batch size 为 `4`，验证 batch size 为 `8`，`warmup_steps=500`，`weight_decay=0.01`。

5. **效果提升** — R-Drop 实验在 dev 集上达到 **93.39% Accuracy** / **0.9225 Macro F1**，优于 DeBERTa-v3-large + LoRA 的 `91.63% / 0.9027`，成为当前 Track 1 最优结果。
