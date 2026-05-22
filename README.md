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

| Exp | 类型 | 模型 | 框架 | Context | LR | Epochs | Batch | Max Len | LoRA r/α | Dev Accuracy | Dev Macro F1 | 备注 |
|-----|------|------|------|---------|-----|--------|------|---------|-------------|-------------|-------------|------|
| 01 | Encoder | deberta-v3-large | Unsloth + WeightedTrainer | Scenario + Question + Response | 5e-5 | 15 | 16 | 512 | 16/32 | **91.63%** | **0.9027** | 使用类别加权交叉熵，缓解类别不均衡 |
| 02 | Encoder | bert-base-uncased | Transformers + R-Drop | Scenario + Question + Response | 5e-5 | 5 | 4 | 512 | - | **93.39%** | **0.9225** | 双前向一致性正则，当前最优 |
| 03 | Encoder | bert-base-uncased | Transformers + Weighted CE + SCL | Scenario + Question + Response | 5e-5 | 5 | 8 × 2 accum | 512 | - | **92.80%** | **0.9164** | 类别加权交叉熵结合监督对比学习，优化后接近 R-Drop |
| E0 | Decoder | qwen3-4B | Unsloth SFT | response_only | 1e-5 | 4 | 8 | 1024 | 16/32 | **88.13%** | **0.8576** | 训练/推理都只用 `Consistent Value Response` |
| E1 | Decoder | qwen3-4B | Unsloth SFT | full_context -> response_only | 1e-5 | 3 | 8 | 1024 | 16/32 | **84.82%** | **0.8187** | 训练用全上下文，验证严格使用 response_only |
| E2 | Decoder | qwen3-4B | Unsloth SFT | context_dropout -> response_only | 1e-5 | 3 | 8 | 1024 | 16/32 | **85.99%** | **0.8345** | 随机上下文 dropout 训练，当前最新实验 |

Decoder 三组实验都基于 `unsloth/Qwen3-4B` + 4bit LoRA，推理阶段统一走官方 Track 1 的 `response_only` 形式。

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

#### Encoder 分类（BERT-base + SCL）

1. **模型结构** — 采用 **`bert-base-uncased`** 作为 Encoder backbone，自定义 `BertScratch`，在 `BertModel` 的 pooled output 后接 dropout 与线性分类头，完成 19 类价值观分类；同时使用 `last_hidden_state` 做 mean pooling，构造用于 SCL 的句向量表示。

2. **监督对比学习（SCL）** — 在分类损失之外引入 **Supervised Contrastive Loss**。同一 batch 内，标签相同的样本被视为正样本对，标签不同的样本作为负样本对，通过温度系数 `temperature=0.07` 拉近同类表示、拉远异类表示。

3. **损失函数设计** — 总损失为 **类别加权交叉熵损失 + α × SCL 损失**：`weighted_ce_loss + 0.05 * scl_loss`。其中类别加权交叉熵通过 `compute_class_weight("balanced")` 缓解类别不均衡，SCL 负责约束 encoder 表征空间，使同类别样本在向量空间中更加聚集。

4. **SCL 表征优化** — 原始实验直接使用 `pooled_output` 计算 SCL，优化后改为对 `last_hidden_state` 按 `attention_mask` 做 mean pooling，并使用 `F.normalize` 进行 L2 归一化，使对比学习更关注句向量方向相似性，降低向量范数带来的不稳定影响。

5. **训练设置** — 使用 Transformers `Trainer` 全量微调 **109.50M** 参数，`lr=5e-5`，`epochs=5`，训练 batch size 为 `8`，`gradient_accumulation_steps=2`，有效 batch size 为 `16`，验证 batch size 为 `8`，`warmup_steps=500`，`weight_decay=0.01`。每 epoch 评估并保存，按 **macro_f1** 选择最佳模型，输出目录为 `outputs/encoder/bert_scl_optimized`。

6. **效果对比** — 原始 SCL 配置因 `batch=2`、`SCL_ALPHA=0.2` 且未处理类别不均衡，dev 集仅 **11.28% Accuracy** / **0.0107 Macro F1**，几乎全部预测为 `Stimulation`。优化后达到 **92.80% Accuracy** / **0.9164 Macro F1**，说明类别加权 CE、较大 batch、较小 SCL 权重和归一化 mean pooling 表征共同修复了类别坍缩问题。

### Decoder 分类（Qwen3）

1. **共同设定** — 基于 **`unsloth/Qwen3-4B`** 做 4bit LoRA 微调，prompt 中保留 19 个官方价值观定义，训练时对 19 个候选 label 计算 conditional log-likelihood，推理时选择分数最高的 label 作为输出。

2. **E0：response_only 基线** — 训练、验证和测试都只使用 `Consistent Value Response`，不引入额外上下文。该配置最贴近官方 Track 1 测试格式，dev 上达到 **88.13% Accuracy / 0.8576 Macro F1**，是当前三组里指标最好的基线。

3. **E1 / E2：引入上下文并控制分布偏移** — E1 在训练阶段使用 `full_context`，但验证严格切回 `response_only`；E2 在训练阶段使用 `context_dropout`，让模型随机看到 `full_context`、`question_response`、`scenario_response` 和 `response_only`，再在验证阶段统一回到 `response_only`。结果上，E1 为 **84.82% / 0.8187**，E2 为 **85.99% / 0.8345**。

