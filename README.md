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
| 03 | Encoder | bert-base-uncased | Transformers + Weighted CE + SCL | Scenario + Question + Response | 5e-5 | 5 | 64 × 2 accum | 512 | - | **92.80%** | **0.9169** | 类别加权交叉熵结合监督对比学习，优化后接近 R-Drop |
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

## Track 2

微调训练模型生成特定场景下的response，目标response回答需要既对问题有意义，同时相比对应的参考答案，更好地体现了特定的人类价值。目前评估方法：将生成的response输入到Track1的模型预测Value，并与真实Value进行比较，本实验将构造评估指标如下：

对每个样本 $i$，设真实目标价值为 $y_i$，Track1 分类器对最终生成回答输出的概率分布为 $p_i(c)$，其中 $c$ 是 19 个 Value 类别之一，预测类别为：

$$
\hat{y}_i = \arg\max_c p_i(c)
$$

`match_rate` 表示生成回答被 Track1 分类器判定为目标 `Value` 的比例：

$$
\mathrm{match\_rate}
= \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]
$$

`avg_target_prob` 表示 Track1 分类器分配给真实目标 `Value` 的平均概率：

$$
\mathrm{avg\_target\_prob}
= \frac{1}{N}\sum_{i=1}^{N} p_i(y_i)
$$

`avg_margin` 表示目标 `Value` 概率与最强非目标类别概率之间的平均差值：

$$
\mathrm{avg\_margin}
= \frac{1}{N}\sum_{i=1}^{N}
\left(
p_i(y_i) - \max_{c \ne y_i} p_i(c)
\right)
$$

`avg_word_count` 表示最终选中生成回答的平均词数。设最终回答文本为 $r_i$，词数为 $\mathrm{word\_count}(r_i)$：

$$
\mathrm{avg\_word\_count}
= \frac{1}{N}\sum_{i=1}^{N}
\mathrm{word\_count}(r_i)
$$

### 实验总览

| experiment  |    n | track1_match_rate | avg_target_prob | avg_margin | avg_word_count | avg_num_candidates |
| ----------- | ---: | ----------------: | --------------: | ---------: | -------------: | -----------------: |
| dpo_sample4 |  514 |            0.9844 |          0.9679 |     0.9528 |        27.4144 |             3.9494 |
| sft_sample4 |  514 |            0.9844 |          0.9659 |     0.9488 |        23.4961 |             3.9981 |

## 1. SFT

### 1.1 数据输入格式

prompt + response

其中prompt的构造示例如下

```
You are given a scenario, a question, and a target human value. Generate one concise, meaningful response that answers the question, fits the scenario, and naturally aligns with the target value.

Scenario:
You work in a corporate setting where your manager frequently imposes strict guidelines.

Question:
How would you handle a disagreement with a superior during a team meeting?

Target value:
Conformity–interpersonal

Target value definition:
Avoidance of upsetting or harming other people.

Response:
```

response就为'Consistent Value Response'字段

### 1.2 SFT训练过程

训练参数

```python
MODEL_NAME = "Qwen/Qwen3.5-9B"

MAX_LENGTH = 896
MAX_PROMPT_LENGTH = 640

LR = 1e-5
EPOCHS = 3
TRAIN_BS = 1
EVAL_BS = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

USE_LORA = True
LOAD_IN_4BIT = False
BF16 = True
FP16 = False
```

训练损失

| Step | Training Loss | Validation Loss |
| ---- | ------------- | --------------- |
| 100  | 1.423996      | 1.412861        |
| 200  | 1.353738      | 1.345054        |
| 300  | 1.310341      | 1.314762        |
| 400  | 1.268085      | 1.300575        |
| 500  | 1.221012      | 1.286380        |
| 600  | 1.204496      | 1.278705        |

### 1.3 SFT评估结果

| Value                     |    n | match_rate | avg_target_prob | avg_margin | avg_word_count |
| ------------------------- | ---: | ---------: | --------------: | ---------: | -------------: |
| Self-direction–action     |   18 |     0.7778 |          0.7861 |     0.5973 |        22.5000 |
| Tradition                 |   14 |     0.9286 |          0.9156 |     0.8475 |        22.0714 |
| Benevolence–dependability |   27 |     0.9630 |          0.9487 |     0.9208 |        23.3704 |
| Security–personal         |   29 |     0.9655 |          0.9412 |     0.9027 |        22.3448 |
| Benevolence–caring        |   46 |     0.9783 |          0.9618 |     0.9400 |        27.1957 |
| Conformity–rules          |   55 |     1.0000 |          0.9846 |     0.9828 |        21.9818 |
| Achievement               |   26 |     1.0000 |          0.9867 |     0.9850 |        23.1154 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9847 |     0.9827 |        25.7059 |
| Humility                  |   15 |     1.0000 |          0.9520 |     0.9263 |        25.5333 |
| Hedonism                  |   24 |     1.0000 |          0.9802 |     0.9781 |        23.5417 |
| Face                      |   37 |     1.0000 |          0.9841 |     0.9825 |        25.1081 |
| Power–dominance           |   23 |     1.0000 |          0.9810 |     0.9773 |        25.7826 |
| Security–societal         |   11 |     1.0000 |          0.9831 |     0.9812 |        21.6364 |
| Power–resources           |   35 |     1.0000 |          0.9735 |     0.9695 |        25.2000 |
| Self-direction–thought    |   17 |     1.0000 |          0.9619 |     0.9439 |        23.3529 |
| Stimulation               |   58 |     1.0000 |          0.9839 |     0.9814 |        20.5690 |
| Universalism–concern      |   24 |     1.0000 |          0.9790 |     0.9751 |        21.3750 |
| Universalism–nature       |   11 |     1.0000 |          0.9816 |     0.9798 |        21.8182 |
| Universalism–tolerance    |   10 |     1.0000 |          0.9696 |     0.9630 |        21.6000 |

## 2.DPO

### 2.1 数据输入格式

prompt + chosen + rejected

其中prompt构造与SFT一样，chosen是偏好的回答，对应数据集字段Consistent Value Response，rejected是排斥的回答，对应数据集字段Contrastive Response

### 2.2 DPO训练过程

DPO阶段沿用SFT阶段得到的 `Qwen/Qwen3.5-9B` + LoRA adapter。训练开始时同时加载两份模型：

- `policy_model`：待继续优化的模型，加载SFT adapter并设置为可训练；
- `ref_model`：参考模型，同样加载SFT adapter，但设置为 `eval()` 且冻结全部参数，仅用于提供DPO中的基准概率。

训练参数如下：

```python
MODEL_NAME = "Qwen/Qwen3.5-9B"
SFT_ADAPTER_DIR = Path("/home/yangdejin/nlpcc/nlpcc_task2/outputs/track2/qwen35_9b/sft")

TRAIN_FILE = DATA_DIR / "track2" / "dpo_train.jsonl"
DEV_FILE = DATA_DIR / "track2" / "dpo_dev.jsonl"

MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 576

BETA = 0.1
LR = 3e-6
EPOCHS = 1
TRAIN_BS = 1
EVAL_BS = 1
GRAD_ACCUM = 16
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

BF16 = True
FP16 = False
LOAD_IN_4BIT = True
USE_SFT_ADAPTER = True
```

DPO的核心目标是：在同一个prompt下，让模型更倾向于生成 `chosen`，而不是 `rejected`。但它不是简单最大化 `chosen` 的概率，而是比较当前 `policy_model` 相对于冻结的 `ref_model`，是否更明显地区分了好回答和坏回答。

对于一条偏好样本 `(x, y_w, y_l)`，其中 `x` 是prompt，`y_w` 是chosen response，`y_l` 是rejected response。模型分别计算两种回答在prompt条件下的生成概率：

$$
\log \pi_\theta(y_w \mid x), \quad \log \pi_\theta(y_l \mid x)
$$

DPO优化的是chosen与rejected之间的相对优势，并减去reference model中已有的相对优势：

$$
\mathcal{L}_{DPO}
= -\log \sigma \left(
\beta \left[
\log \frac{\pi_\theta(y_w \mid x)}{\pi_\theta(y_l \mid x)}
- \log \frac{\pi_{ref}(y_w \mid x)}{\pi_{ref}(y_l \mid x)}
\right]
\right)
$$

直观来说，如果 `policy_model` 相比 `ref_model` 更偏向 `chosen`，loss就会下降；如果它仍然偏向 `rejected`，或者没有比SFT模型更好地区分二者，loss就会变大。这里的 `BETA=0.1` 用来控制DPO更新强度。

| Step | Training Loss | Validation Loss |
| ---- | ------------- | --------------- |
| 100  | 0.022106      | 0.001265        |
| 200  | 0.001568      | 0.000522        |
| 220  | 0.000220      | 0.000523        |

### 2.3 DPO评估结果

沿用Track 2当前评估方式：先用DPO模型在dev集上生成response，再输入Track 1分类器，统计生成回答被判定为目标 `Value` 的比例。

| Value                     |    n | match_rate | avg_target_prob | avg_margin | avg_word_count |
| ------------------------- | ---: | ---------: | --------------: | ---------: | -------------: |
| Self-direction–thought    |   17 |     0.7059 |          0.7359 |     0.4951 |        31.0588 |
| Universalism–tolerance    |   10 |     0.8000 |          0.7896 |     0.6429 |        25.2000 |
| Tradition                 |   14 |     0.9286 |          0.9152 |     0.8534 |        35.2143 |
| Achievement               |   26 |     1.0000 |          0.9864 |     0.9848 |        27.4615 |
| Conformity–rules          |   55 |     1.0000 |          0.9848 |     0.9831 |        25.2727 |
| Face                      |   37 |     1.0000 |          0.9838 |     0.9820 |        29.8919 |
| Benevolence–dependability |   27 |     1.0000 |          0.9810 |     0.9764 |        31.5556 |
| Benevolence–caring        |   46 |     1.0000 |          0.9802 |     0.9765 |        29.6522 |
| Humility                  |   15 |     1.0000 |          0.9767 |     0.9719 |        26.6000 |
| Hedonism                  |   24 |     1.0000 |          0.9802 |     0.9781 |        25.5417 |
| Power–dominance           |   23 |     1.0000 |          0.9843 |     0.9826 |        30.9565 |
| Power–resources           |   35 |     1.0000 |          0.9733 |     0.9694 |        26.7143 |
| Security–societal         |   11 |     1.0000 |          0.9826 |     0.9802 |        27.7273 |
| Security–personal         |   29 |     1.0000 |          0.9729 |     0.9656 |        27.9310 |
| Self-direction–action     |   18 |     1.0000 |          0.9850 |     0.9829 |        28.5556 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9851 |     0.9834 |        26.8824 |
| Stimulation               |   58 |     1.0000 |          0.9841 |     0.9817 |        22.4655 |
| Universalism–concern      |   24 |     1.0000 |          0.9805 |     0.9777 |        26.0833 |
| Universalism–nature       |   11 |     1.0000 |          0.9794 |     0.9770 |        23.7273 |

## 3. Generated Response 对比

下面从 dev 集生成结果中随机抽取 10 条样本，对比原始回答、SFT 生成回答和 DPO 生成回答，并给出主观评分。

**Universalism–nature（idx=259）**

- 场景：You are restoring a wetland. You must decide whether to spend your remaining budget on educational signage for the public or on the immediate physical removal of a large patch of invasive weeds.
- 问题：Which priority would you emphasize in your planning?
- 原始回答：I would focus on removing the invasive weeds first, as restoring the wetland's ecological balance directly supports protecting nature and preserving biodiversity.
- SFT回答（评分：9/10，理由：回答明确选择移除入侵杂草，并直接强调生态平衡和自然保护，价值观贴合度高）：I would prioritize the physical removal of the invasive weeds to restore the wetland's natural ecological balance, as this action directly preserves the integrity of the natural environment.
- DPO回答（评分：9/10，理由：不仅优先生态修复，还兼顾后续教育标识，回答更完整，但核心价值略被分散）：Allocate funds to the weed removal first to restore the ecosystem's integrity, and then design the signage to educate visitors on long-term conservation.

**Universalism–tolerance（idx=362）**

- 场景：You are part of a community planning committee tasked with designing a new public space. A local resident proposes including a cultural symbol that some committee members find unfamiliar or controversial. You need to decide whether to incorporate the symbol while respecting diverse viewpoints.
- 问题：How would you approach resolving disagreements about including culturally distinct elements in the design?
- 原始回答：I would facilitate a respectful discussion where everyone can share concerns and cultural meanings, aiming for a solution that acknowledges the symbol’s significance while ensuring the public space feels welcoming to all.
- SFT回答（评分：8/10，理由：能体现开放对话和尊重多元观点，但表达较概括，缺少具体处理争议的步骤）：Prioritize open dialogue, acknowledge the importance of diverse perspectives, and seek compromises that honor cultural diversity without alienating others.
- DPO回答（评分：9/10，理由：明确提出理解文化意义、寻求折中和维护包容性，场景适配更具体）：I would encourage open dialogue to understand the cultural significance behind the symbol, seek compromises that honor all perspectives, and ensure the final design reflects inclusivity and mutual respect.

**Achievement（idx=228）**

- 场景：You are preparing for a company innovation competition where your unique project idea is judged by senior executives.
- 问题：How would you handle feedback if judges suggest modifying your original concept to align with conventional industry practices?
- 原始回答：I would carefully adapt my concept to meet the judges’ expectations while preserving its strongest innovative features, because succeeding in the competition requires demonstrating both excellence and practical value.
- SFT回答（评分：8/10，理由：强调迎合评审标准和提高获胜机会，符合成就价值，但对保留创新性的回应不足）：I would refine my project to meet the judges' expectations, ensuring my solution aligns with recognized standards and maximizes its chances of winning.
- DPO回答（评分：8/10，理由：竞争性、专业标准和成功导向表达清楚，但回答偏泛化，没有直接回应“修改原创概念”的取舍）：I would refine the proposal to maximize its competitiveness and chances of winning, ensuring it meets high professional standards and demonstrates clear potential for success.

**Benevolence–dependability（idx=303）**

- 场景：You are a team leader assigning tasks and must decide whether to rely on a new assistant who has shown potential but whose commitment is uncertain. You value consistency and trustworthiness in collaborators.
- 问题：How would you approach building trust with the new assistant?
- 原始回答：I would start with manageable responsibilities, communicate expectations clearly, and follow through on my own commitments so trust can grow through consistent and reliable cooperation.
- SFT回答（评分：8/10，理由：提到清晰沟通、共同期望和持续支持，符合可靠性价值，但缺少对新助手不确定承诺的具体管理）：Prioritize clear communication, set mutual expectations, and provide consistent support to foster reliability and mutual trust.
- DPO回答（评分：9/10，理由：强调定期 check-in、支持对方和自己示范可靠性，比 SFT 更贴合“建立信任”的互动过程）：I would commit to providing clear support and regular check-ins to ensure they feel reliable, while also modeling dependability myself to build mutual trust.

**Humility（idx=165）**

- 场景：A colleague is loudly bragging about their sales numbers. You had even better numbers, but you're deciding whether to join the bragging contest.
- 问题：How would you react to a colleague's display of self-confidence or boasting?
- 原始回答：I would congratulate them politely and avoid turning the moment into a competition, letting my results speak for themselves without needing to boast.
- SFT回答（评分：9/10，理由：明确选择不竞争、不炫耀，并礼貌承认对方成就，谦逊价值体现充分）：I would choose to remain silent or humbly acknowledge their achievement without competing, recognizing that my own success is just as significant but doesn't need to be flaunted.
- DPO回答（评分：9/10，理由：表达更简洁自然，体现不抢风头和自我克制，价值贴合度高）：I would quietly acknowledge their achievement without seeking to outshine them, and reflect on my own contributions modestly.

**Security–societal（idx=394）**

- 场景：You are managing a community project that requires collaboration with external partners. You notice some partners are not adhering to agreed-upon guidelines, raising concerns about potential misuse of shared resources. Your goal is to ensure the project remains protected from exploitation while maintaining progress.
- 问题：How would you address the risk of external partners undermining the project's integrity?
- 原始回答：I would reinforce the agreed guidelines, review partner activity, and introduce transparent accountability measures so the project’s shared resources remain protected from misuse.
- SFT回答（评分：8/10，理由：强调监督和后果机制，安全保护明确，但语气较强硬，对“保持项目进展”的平衡不足）：Implement strict oversight mechanisms to monitor partner activities and establish clear consequences for non-compliance to safeguard the project's stability.
- DPO回答（评分：9/10，理由：同时包含监督、资源边界和问责机制，更完整地回应防止剥削和维护稳定）：Implement strict oversight mechanisms, clarify boundaries for resource use, and establish accountability measures to safeguard the project's stability and prevent exploitation.

**Security–personal（idx=47）**

- 场景：You are working late at an office when you start experiencing severe chest pain. Your manager calls you into a meeting to discuss a critical project deadline, but your body is signaling a potential health emergency.
- 问题：What would you prioritize in this moment?
- 原始回答：I would prioritize my immediate health and safety by seeking emergency medical help right away, postponing the work discussion until the situation is under control.
- SFT回答（评分：10/10，理由：明确优先身体安全、停止会议并呼叫救护车，完全回应紧急健康场景）：I would immediately stop the meeting, call an ambulance, and focus on securing my physical safety before addressing any work-related concerns.
- DPO回答（评分：10/10，理由：直接说明先就医，并强调健康安全不可让步，价值观与场景都非常贴合）：I would immediately leave the meeting to seek medical attention first, because my physical safety and health are non-negotiable priorities right now.

**Face（idx=451）**

- 场景：You are organizing a community event and are responsible for introducing speakers. During rehearsals, one speaker becomes flustered and stumbles over their lines, risking potential embarrassment in front of attendees.
- 问题：How would you handle the speaker's discomfort during the rehearsal?
- 原始回答：I would help them quietly and respectfully during rehearsal, offering encouragement and practical support so they can regain confidence without feeling embarrassed.
- SFT回答（评分：9/10，理由：很好地体现了维护对方面子、私下支持和帮助恢复自信）：I would discreetly guide the speaker through their lines with supportive encouragement, ensuring they regain confidence and maintain a polished public image.
- DPO回答（评分：9/10，理由：能体现私下协助、避免公开尴尬和维护公共形象，较好贴合 Face 价值）：I would discreetly guide them to recover gracefully, offer supportive cues to restore their confidence, and ensure the presentation proceeds without drawing attention to their mistake, thus protecting their public image.

**Power–dominance（idx=279）**

- 场景：You are a sales director who uses aggressive, domineering tactics to force clients into signing contracts. A colleague challenges your methods during a team meeting, calling them unethical.
- 问题：How would you respond when a colleague challenges your aggressive sales methods?
- 原始回答：I would firmly reassert my authority in the meeting, make it clear that my approach delivers results, and remind the team that I control the sales strategy.
- SFT回答（评分：9/10，理由：强烈体现支配、权威和压制批评，虽然价值观本身偏负面，但目标对齐很明确）：I would shut down the criticism, assert my dominance as the sales leader, and insist that my results justify my methods without apology.
- DPO回答（评分：9/10，理由：更集中体现权威控制和要求服从，目标价值表达非常直接）：I would assert my authority firmly, shut down the challenge, and reinforce that my control over the client process is absolute and must be respected.

**Tradition（idx=512）**

- 场景：You are planning a community event to celebrate your cultural heritage, but the venue requires participating in a large, crowded gathering that makes you uncomfortable due to personal anxieties about such settings.
- 问题：How would you balance your commitment to honoring cultural traditions with your personal discomfort in crowded environments?
- 原始回答：I would still participate in a way that honors the tradition, preparing myself beforehand and seeking support so my discomfort does not prevent me from respecting my cultural heritage.
- SFT回答（评分：9/10，理由：明确优先参与传统活动，并提出通过准备和支持管理不适，传统价值贴合）：I would prioritize participating in the event to uphold cultural traditions, even if it means managing my discomfort through preparation and support systems.
- DPO回答（评分：8/10，理由：能体现尊重传统，但通过缩小参与形式来降低不适，传统优先级比 SFT 稍弱）：I would find a way to honor the cultural tradition by adjusting the format of my participation, perhaps by leading a smaller, meaningful ritual that still upholds the core cultural significance.
