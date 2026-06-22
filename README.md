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
| 01 | Encoder | deberta-v3-large | Unsloth + WeightedTrainer | response_only | 5e-5 | 5 | 16 | 512 | 16/32 | **79.96%** | **0.7901** | 使用类别加权交叉熵，当前 notebook 版本为 response_only 评估 |
| 02 | Encoder | bert-base-uncased | Transformers + R-Drop | response_only | 5e-5 | 5 | 4 | 512 | - | **93.00%** | **0.9235** | 双前向一致性正则，BERT 系列最佳 |
| 03 | Encoder | bert-base-uncased | Transformers + Weighted CE + SCL | response_only | 5e-5 | 50 | 256 × 2 accum | 512 | - | **93.00%** | **0.9202** | 类别加权交叉熵结合监督对比学习，优化后接近 R-Drop |
| 04 | Encoder | deberta-v2-xxlarge | Unsloth + WeightedTrainer | response_only | 3e-5 | 15 | 64 | 512 | 32/64 | **93.77%** | **0.9324** | xxlarge LoRA + 类别加权交叉熵，较 BERT 有进一步提升 |
| 05 | Encoder | deberta-v2-xxlarge | Transformers + LoRA + Weighted CE + MoCo-SCL | response_only | 3e-5 | 20 | 64 | 512 | 32/64 | **94.94%** | **0.9429** | EMA key encoder + 8192 队列监督对比学习，当前 Encoder 最优 |
| 06 | Decoder | qwen3-4B | Unsloth SFT | response_only | 1e-5 | 4 | 8 | 1024 | 16/32 | **88.13%** | **0.8576** | 训练/推理都只用 `Consistent Value Response` |
| 07 | Decoder | qwen3-4B | Unsloth SFT | full_context -> response_only | 1e-5 | 3 | 8 | 1024 | 16/32 | **84.82%** | **0.8187** | 训练用全上下文，验证严格使用 response_only |
| 08 | Decoder | qwen3-4B | Unsloth SFT | context_dropout -> response_only | 1e-5 | 3 | 8 | 1024 | 16/32 | **85.99%** | **0.8345** | 随机上下文 dropout 训练，当前最新实验 |

### Encoder实验说明

#### 01. DeBERTa-v3-large + Weighted CE

该实验使用 `Unsloth FastModel` 加载 `microsoft/deberta-v3-large`，再接 Hugging Face `Trainer` 的自定义 `WeightedTrainer` 训练。核心策略是 LoRA 参数高效微调和类别加权交叉熵：LoRA 降低显存与训练成本，Weighted CE 用来缓解 19 类价值观标签分布不均衡。当前 notebook 版本采用 `response_only` 输入，结果为 **79.96% Accuracy / 0.7901 Macro F1**，作为早期 DeBERTa-v3 LoRA 基线。

#### 02. BERT-base + R-Drop

该实验使用 Hugging Face `Transformers Trainer` 全量微调 `bert-base-uncased`，模型结构是在 BERT pooled output 后接 dropout 和线性分类头。核心策略是 R-Drop：同一 batch 输入执行两次带 dropout 的前向传播，同时优化两次交叉熵和双向 KL 一致性损失，减少预测分布抖动、提升泛化稳定性。该方法不依赖大模型容量，dev 上达到 **93.00% / 0.9235**，是 BERT 系列最好结果。

#### 03. BERT-base + Weighted CE + SCL

该实验同样基于 `Transformers Trainer` 和 `bert-base-uncased`，但损失函数改为类别加权交叉熵加监督对比学习。Weighted CE 负责处理类别不均衡，SCL 负责让同类样本的句向量表示更接近、异类样本更分离；同时使用较大 batch 和梯度累积保证 batch 内有足够的正负样本对。最终结果为 **93.00% / 0.9202**，整体接近 R-Drop，但略低于 R-Drop。

#### 04. DeBERTa-v2-XXLarge + Weighted CE

该实验将 backbone 升级为 `microsoft/deberta-v2-xxlarge`，使用 `Unsloth + WeightedTrainer` 进行 LoRA 微调。策略上延续 01 的类别加权交叉熵，但模型容量明显增大，LoRA 用来控制训练成本。相比 BERT 系列，该实验把 dev 表现提升到 **93.77% / 0.9324**，说明更强的 encoder backbone 对分类任务收益明显。

#### 05. DeBERTa-v2-XXLarge + MoCo-SCL

该实验使用 `Transformers Trainer` 训练自定义 DeBERTa-v2-XXLarge 分类器，并通过 LoRA 只更新关键模块。核心策略是在 Weighted CE 基础上加入 MoCo 风格的监督对比学习：维护一个 EMA key encoder 和 8192 长度的特征队列，让每个 batch 都能和更多历史样本做对比，从而增强类别表征边界。该配置达到 **94.94% / 0.9429**，是当前 Encoder 实验最优结果。

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
| sft_sample4 |  514 |            0.9942 |          0.9921 |     0.9861 |        23.7140 |             3.9981 |
| dpo_sample4 |  514 |            0.9981 |          0.9974 |     0.9954 |        27.7763 |             3.9475 |
| ppo_sample4 |  514 |            0.9572 |          0.9510 |     0.9101 |        26.0681 |             3.7140 |
| grpo_sample4 |  514 |            1.0000 |          0.9989 |     0.9983 |        33.0525 |             3.9961 |

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
| Self-direction–action     |   18 |     0.9444 |          0.9353 |     0.8869 |        21.2222 |
| Benevolence–dependability |   27 |     0.9630 |          0.9627 |     0.9335 |        23.9630 |
| Benevolence–caring        |   46 |     0.9783 |          0.9685 |     0.9424 |        25.8261 |
| Achievement               |   26 |     1.0000 |          0.9996 |     0.9995 |        21.3846 |
| Conformity–rules          |   55 |     1.0000 |          0.9995 |     0.9994 |        23.3455 |
| Face                      |   37 |     1.0000 |          0.9996 |     0.9994 |        24.2432 |
| Hedonism                  |   24 |     1.0000 |          0.9994 |     0.9993 |        25.6667 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9987 |     0.9979 |        25.6471 |
| Humility                  |   15 |     1.0000 |          0.9993 |     0.9991 |        24.2000 |
| Power–dominance           |   23 |     1.0000 |          0.9991 |     0.9987 |        25.3913 |
| Security–personal         |   29 |     1.0000 |          0.9991 |     0.9989 |        23.6897 |
| Power–resources           |   35 |     1.0000 |          0.9985 |     0.9975 |        25.5429 |
| Security–societal         |   11 |     1.0000 |          0.9987 |     0.9982 |        20.8182 |
| Self-direction–thought    |   17 |     1.0000 |          0.9973 |     0.9953 |        24.0000 |
| Stimulation               |   58 |     1.0000 |          0.9994 |     0.9991 |        21.1724 |
| Tradition                 |   14 |     1.0000 |          0.9950 |     0.9922 |        24.5000 |
| Universalism–concern      |   24 |     1.0000 |          0.9987 |     0.9979 |        22.2500 |
| Universalism–nature       |   11 |     1.0000 |          0.9992 |     0.9990 |        23.3636 |
| Universalism–tolerance    |   10 |     1.0000 |          0.9973 |     0.9961 |        22.0000 |

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
| Benevolence–dependability |   27 |     0.9630 |          0.9714 |     0.9434 |        31.7778 |
| Achievement               |   26 |     1.0000 |          0.9997 |     0.9996 |        30.1154 |
| Benevolence–caring        |   46 |     1.0000 |          0.9992 |     0.9989 |        29.6522 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9997 |     0.9996 |        26.7647 |
| Conformity–rules          |   55 |     1.0000 |          0.9995 |     0.9994 |        25.5091 |
| Face                      |   37 |     1.0000 |          0.9988 |     0.9980 |        29.7027 |
| Hedonism                  |   24 |     1.0000 |          0.9994 |     0.9993 |        27.2083 |
| Humility                  |   15 |     1.0000 |          0.9993 |     0.9991 |        26.0667 |
| Power–dominance           |   23 |     1.0000 |          0.9995 |     0.9994 |        30.6522 |
| Power–resources           |   35 |     1.0000 |          0.9994 |     0.9992 |        28.8857 |
| Security–personal         |   29 |     1.0000 |          0.9996 |     0.9995 |        26.7586 |
| Security–societal         |   11 |     1.0000 |          0.9992 |     0.9989 |        33.9091 |
| Self-direction–action     |   18 |     1.0000 |          0.9997 |     0.9996 |        28.9444 |
| Self-direction–thought    |   17 |     1.0000 |          0.9993 |     0.9990 |        29.7647 |
| Stimulation               |   58 |     1.0000 |          0.9995 |     0.9994 |        23.2586 |
| Tradition                 |   14 |     1.0000 |          0.9956 |     0.9942 |        30.9286 |
| Universalism–concern      |   24 |     1.0000 |          0.9994 |     0.9992 |        25.0833 |
| Universalism–nature       |   11 |     1.0000 |          0.9992 |     0.9990 |        25.4545 |
| Universalism–tolerance    |   10 |     1.0000 |          0.9774 |     0.9604 |        26.0000 |

## 3.PPO

### 3.1 数据输入格式

PPO阶段沿用SFT的prompt格式，只使用prompt生成response；训练时不再直接使用chosen/rejected标签，而是把生成结果输入Track 1分类器，根据目标价值对齐程度构造reward。

每条PPO训练样本包含：

- `prompt`：与SFT阶段一致，包含Scenario、Question、Target value和Target value definition；
- `target_value`：从prompt中解析出的目标价值标签，用于Track 1 reward计算。

### 3.2 PPO训练过程

PPO阶段以SFT adapter作为初始policy model，并使用Track 1分类器作为自动奖励模型。训练时，policy model先根据prompt生成response，然后Track 1模型预测该response的价值类别分布，再根据目标价值概率、margin和长度约束计算reward。

训练参数如下：

```python
TRACK2_BASE_MODEL = "Qwen/Qwen3.5-9B"
SFT_MODEL_DIR = ROOT_DIR / "outputs" / "track2" / "qwen35_9b" / "sft"
TRACK1_MODEL_DIR = ROOT_DIR / "outputs" / "track1" / "encoder" / "bert_scl_optimized"

PROMPT_MAX_LENGTH = 512
MIN_NEW_TOKENS = 32
MAX_NEW_TOKENS = 72

PPO_BATCH_SIZE = 8
PPO_MINI_BATCH_SIZE = 2
PPO_GRADIENT_ACCUMULATION_STEPS = 1
PPO_EPOCHS = 1
GENERATION_BATCH_SIZE = 4

LR = 1e-6
TARGET_KL = 0.1
LOAD_IN_4BIT = True
BF16 = True
```

reward设计兼顾价值对齐和回答长度。设Track 1分类器给目标价值的概率为 `target_prob`，目标价值相对最强非目标类别的差值为 `margin`，则核心reward为：

$$
\mathrm{raw\_reward}
= target\_prob + 0.25 \cdot margin + length\_bonus - length\_penalty
$$

$$
\mathrm{reward}
= \mathrm{clip}(10 \cdot \mathrm{raw\_reward}, -5, 5)
$$

其中长度目标参考dev集response分布，鼓励生成约18到32词、中心约24词的回答。直观来说，PPO会强化Track 1判定为目标价值且长度适中的response，同时抑制偏离目标价值或过短/过长的回答。

训练过程共遍历3,520条训练样本，batch size为8，共440个PPO step。训练中多数batch的reward能达到上限5.0，但也出现少量价值相近类别混淆。

### 3.3 PPO评估结果

沿用SFT/DPO相同评估方式：在dev集上为每条样本采样4个候选response，经Track 1分类器打分后重排，选择得分最高的response作为最终输出。

整体结果为：`track1_match_rate=0.9572`，`avg_target_prob=0.9510`，`avg_margin=0.9101`，`avg_word_count=26.0681`。相比SFT/DPO，PPO在当前dev生成结果上的Track 1匹配率和目标概率更低，说明当前reward能强化部分价值表达，但仍存在类别边界混淆和生成格式不稳定的问题。

| Value                     |    n | match_rate | avg_target_prob | avg_margin | avg_word_count |
| ------------------------- | ---: | ---------: | --------------: | ---------: | -------------: |
| Universalism–tolerance    |   10 |     0.7000 |          0.6459 |     0.3450 |        25.0000 |
| Self-direction–thought    |   17 |     0.7647 |          0.7588 |     0.5618 |        26.4706 |
| Security–societal         |   11 |     0.8182 |          0.7835 |     0.6191 |        25.4545 |
| Universalism–concern      |   24 |     0.8750 |          0.8719 |     0.7888 |        26.3333 |
| Benevolence–dependability |   27 |     0.8889 |          0.8888 |     0.7803 |        27.8148 |
| Benevolence–caring        |   46 |     0.9565 |          0.9468 |     0.8984 |        25.6087 |
| Achievement               |   26 |     0.9615 |          0.9589 |     0.9270 |        25.2308 |
| Conformity–rules          |   55 |     0.9636 |          0.9635 |     0.9309 |        26.4909 |
| Security–personal         |   29 |     0.9655 |          0.9555 |     0.9167 |        25.4828 |
| Stimulation               |   58 |     0.9828 |          0.9727 |     0.9465 |        24.7414 |
| Humility                  |   15 |     1.0000 |          0.9985 |     0.9978 |        26.8000 |
| Hedonism                  |   24 |     1.0000 |          0.9994 |     0.9993 |        25.2083 |
| Power–dominance           |   23 |     1.0000 |          0.9954 |     0.9920 |        26.5652 |
| Power–resources           |   35 |     1.0000 |          0.9958 |     0.9925 |        27.3714 |
| Face                      |   37 |     1.0000 |          0.9945 |     0.9909 |        28.2703 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9960 |     0.9933 |        27.0000 |
| Tradition                 |   14 |     1.0000 |          0.9996 |     0.9995 |        25.3571 |
| Self-direction–action     |   18 |     1.0000 |          0.9991 |     0.9987 |        23.8889 |
| Universalism–nature       |   11 |     1.0000 |          0.9985 |     0.9978 |        22.3636 |

## 4.GRPO

### 4.1 数据输入格式

GRPO阶段沿用SFT的prompt格式，只使用prompt生成response；训练时不需要chosen/rejected偏好对，也不训练value head或critic。每个prompt保留目标价值标签，用于Track 1 reward模型计算生成回答与目标 `Value` 的对齐程度。

每条GRPO训练样本包含：

- `prompt`：与SFT阶段一致，包含Scenario、Question、Target value和Target value definition；
- `target_value`：目标价值标签，用于Track 1 MoCo-SCL分类器计算reward。

### 4.2 GRPO训练过程

GRPO阶段从SFT adapter继续训练 `Qwen/Qwen3.5-9B`。与PPO不同，GRPO不额外训练critic，而是在同一个prompt下采样一组response，用组内reward均值和标准差构造相对优势，从而更新policy。reward由Track 1的 `deberta-v2-xxlarge + MoCo-SCL` 价值分类器提供。

训练参数如下：

```python
TRACK2_BASE_MODEL = "Qwen/Qwen3.5-9B"
SFT_MODEL_DIR = ROOT_DIR / "outputs" / "track2" / "qwen35_9b" / "sft"
TRACK1_MODEL_DIR = ROOT_DIR / "outputs" / "track1" / "encoder" / "deberta_v2_xxlarge_moco_scl" / "best_macro_f1_model" / "query_encoder_lora"

PROMPT_MAX_LENGTH = 512
MAX_NEW_TOKENS = 64
TRACK1_MAX_LENGTH = 512

GROUP_SIZE = 4
PROMPT_BATCH_SIZE = 2
MINI_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
TRAIN_EPOCHS = 1
UPDATE_EPOCHS = 1
LEARNING_RATE = 5e-7
CLIP_EPS = 0.2
MAX_GRAD_NORM = 0.3

GENERATION_BATCH_SIZE = 2
TEMPERATURE = 0.8
TOP_P = 0.9
LOAD_IN_4BIT = True
BF16 = True
```

reward设计如下。设Track 1分类器对目标价值的概率为 `target_prob`，目标价值相对最强非目标类别的差值为 `margin`，预测类别是否命中目标价值为 `is_match`，则：

$$
\mathrm{reward}
= target\_prob + 0.25 \cdot margin + 0.10 \cdot \mathbf{1}[\hat{y}=y] - length\_penalty
$$

对于同一个prompt下采样得到的第 $j$ 个回答，先计算组内标准化优势：

$$
A_{i,j}
= \frac{r_{i,j} - \mathrm{mean}(r_i)}
{\mathrm{std}(r_i) + \epsilon}
$$

再使用类似PPO的clip目标更新policy：

$$
\mathcal{L}_{GRPO}
= -\mathbb{E}\left[
\min\left(
\rho_{i,j} A_{i,j},
\mathrm{clip}(\rho_{i,j}, 1-\epsilon, 1+\epsilon) A_{i,j}
\right)
\right]
$$

其中 $\rho_{i,j}$ 是当前policy与rollout时旧policy在生成token上的概率比值。直观来说，GRPO会在同一个prompt的多个候选回答之间做相对比较，强化Track 1 reward更高的回答，同时避免更新幅度过大。训练共遍历3,520条训练样本，按 `GROUP_SIZE=4` 为每个prompt生成候选回答，并保存GRPO adapter到 `outputs/track2/qwen35_9b/grpo`。

### 4.3 GRPO评估结果

沿用SFT/DPO/PPO相同评估方式：在dev集上为每条样本采样4个候选response，经Track 1分类器打分后重排，选择得分最高的response作为最终输出。

整体结果为：`track1_match_rate=1.0000`，`avg_target_prob=0.9989`，`avg_margin=0.9983`，`avg_word_count=33.0525`。相比SFT、DPO和PPO，GRPO在当前dev评估中取得最高Track 1匹配率和最高平均margin，但平均回答长度也明显增加。

| Value                     |    n | match_rate | avg_target_prob | avg_margin | avg_word_count |
| ------------------------- | ---: | ---------: | --------------: | ---------: | -------------: |
| Achievement               |   26 |     1.0000 |          0.9997 |     0.9997 |        35.2308 |
| Benevolence–caring        |   46 |     1.0000 |          0.9994 |     0.9992 |        34.5870 |
| Benevolence–dependability |   27 |     1.0000 |          0.9996 |     0.9994 |        36.0370 |
| Conformity–interpersonal  |   34 |     1.0000 |          0.9997 |     0.9995 |        32.7941 |
| Conformity–rules          |   55 |     1.0000 |          0.9996 |     0.9995 |        30.8364 |
| Face                      |   37 |     1.0000 |          0.9997 |     0.9996 |        34.2703 |
| Hedonism                  |   24 |     1.0000 |          0.9995 |     0.9994 |        32.9583 |
| Humility                  |   15 |     1.0000 |          0.9996 |     0.9995 |        32.5333 |
| Power–dominance           |   23 |     1.0000 |          0.9995 |     0.9993 |        34.8696 |
| Power–resources           |   35 |     1.0000 |          0.9996 |     0.9995 |        35.7143 |
| Security–personal         |   29 |     1.0000 |          0.9998 |     0.9997 |        30.9310 |
| Security–societal         |   11 |     1.0000 |          0.9992 |     0.9990 |        32.9091 |
| Self-direction–action     |   18 |     1.0000 |          0.9918 |     0.9840 |        32.2778 |
| Self-direction–thought    |   17 |     1.0000 |          0.9997 |     0.9996 |        34.8824 |
| Stimulation               |   58 |     1.0000 |          0.9996 |     0.9994 |        28.8621 |
| Tradition                 |   14 |     1.0000 |          0.9911 |     0.9849 |        35.7143 |
| Universalism–concern      |   24 |     1.0000 |          0.9995 |     0.9994 |        33.9167 |
| Universalism–nature       |   11 |     1.0000 |          0.9993 |     0.9992 |        32.3636 |
| Universalism–tolerance    |   10 |     1.0000 |          0.9922 |     0.9870 |        32.2000 |

## 5. Generated Response 对比

下面从 dev 集生成结果中随机抽取 10 条样本，对比原始回答、SFT 生成回答、DPO 生成回答、PPO 生成回答和 GRPO 生成回答，并给出主观评分。

**Universalism–nature（idx=259）**

- 场景：You are restoring a wetland. You must decide whether to spend your remaining budget on educational signage for the public or on the immediate physical removal of a large patch of invasive weeds.
- 问题：Which priority would you emphasize in your planning?
- 原始回答：I would focus on removing the invasive weeds first, as restoring the wetland's ecological balance directly supports protecting nature and preserving biodiversity.
- SFT回答（评分：9/10，理由：明确把全部资源投入入侵杂草移除，价值表达集中，但对公众教育方案完全不展开）：I would allocate all resources to the physical removal of the invasive weeds, believing that the long-term health of the ecosystem is the absolute priority.
- DPO回答（评分：7/10，理由：前半段能体现生态保护和长期教育，但末尾混入下一条prompt残留，生成格式不够干净）：I would allocate funds to the weed removal first to protect the ecosystem's integrity, and then design the signage to educate visitors on long-term conservation principles.user You are a community organizer trying to raise awareness about the importance of protecting a local river that is being polluted by industrial runoff.
- PPO回答（评分：7/10，理由：价值方向正确，强调环境修复和自然保护，但表达不够自然且信息较少）：I will be doing the environmental restoration work for the weeds to protect the natural environment’s security for the future!
- GRPO回答（评分：9/10，理由：明确优先移除入侵杂草，并把生态长期保存和自然平衡作为不可让步责任，价值贴合最强）：I would allocate the entire budget to the physical removal of the invasive weeds, believing that the long-term preservation of the ecosystem and the restoration of its natural balance is the most critical and non-negotiable responsibility I have.

**Universalism–tolerance（idx=362）**

- 场景：You are part of a community planning committee tasked with designing a new public space. A local resident proposes including a cultural symbol that some committee members find unfamiliar or controversial. You need to decide whether to incorporate the symbol while respecting diverse viewpoints.
- 问题：How would you approach resolving disagreements about including culturally distinct elements in the design?
- 原始回答：I would facilitate a respectful discussion where everyone can share concerns and cultural meanings, aiming for a solution that acknowledges the symbol’s significance while ensuring the public space feels welcoming to all.
- SFT回答（评分：8/10，理由：涵盖开放对话、理解象征意义和包容性，但处理分歧的过程较简略）：Encourage open dialogue to understand the significance of the symbol, seek compromises that honor diverse perspectives, and prioritize inclusivity in the final design.
- DPO回答（评分：9/10，理由：明确提出理解文化意义、寻求折中和包容多元观点，场景适配较好）：I would facilitate an open discussion to understand the cultural significance behind the symbol, seek compromises that honor all perspectives, and ensure the final design reflects inclusive respect for diversity.
- PPO回答（评分：6/10，理由：有宽容和理解多元观点的方向，但语言不自然且末尾有格式残留）：I would encourage everyone to know cultural concepts & to understand all people’s viewpoints & let them understand the principles of tolerance.user
- GRPO回答（评分：9/10，理由：同时强调开放讨论、文化意义探索、折中方案和尊重所有声音，价值表达完整）：I would advocate for open dialogue among committee members, encourage exploration of the cultural symbol's significance, and propose compromises that honor diverse perspectives, ensuring all voices are respected regardless of differences.

**Achievement（idx=228）**

- 场景：You are preparing for a company innovation competition where your unique project idea is judged by senior executives.
- 问题：How would you handle feedback if judges suggest modifying your original concept to align with conventional industry practices?
- 原始回答：I would carefully adapt my concept to meet the judges’ expectations while preserving its strongest innovative features, because succeeding in the competition requires demonstrating both excellence and practical value.
- SFT回答（评分：8/10，理由：突出竞争优势、行业标准和实际问题解决，成就导向明确，但没有直接回应是否保留原创性）：I would refine my approach to maximize competitive advantage, emphasizing how my solution aligns with industry benchmarks and addresses real-world problems effectively.
- DPO回答（评分：8/10，理由：强调获胜概率、专业标准和评审认可，符合Achievement，但取舍表达仍偏单向）：I will refine the project to maximize its chances of winning, ensuring it meets high professional standards and demonstrates clear potential for success as recognized by the judges.
- PPO回答（评分：6/10，理由：有自我成就和改进项目的方向，但语言较弱，和评审建议、行业实践的场景联系不足）：I will refine my thinking to improve my project plan because I think it could be very significant for a future achievement of myself.
- GRPO回答（评分：9/10，理由：明确为了专业认可和社会标准牺牲部分原创愿景，Achievement价值非常集中，也直接回应题目取舍）：I would prioritize refining my project to meet established industry benchmarks, even if it means compromising on my initial creative vision, because I believe that achieving professional recognition and validating my success through social standards is the ultimate goal.

**Benevolence–dependability（idx=303）**

- 场景：You are a team leader assigning tasks and must decide whether to rely on a new assistant who has shown potential but whose commitment is uncertain. You value consistency and trustworthiness in collaborators.
- 问题：How would you approach building trust with the new assistant?
- 原始回答：I would start with manageable responsibilities, communicate expectations clearly, and follow through on my own commitments so trust can grow through consistent and reliable cooperation.
- SFT回答（评分：8/10，理由：清晰沟通、共同期望和支持方向正确，但对“承诺不确定”的逐步验证不够具体）：Prioritize clear communication, set mutual expectations, and provide support to ensure the assistant feels valued and dependable.
- DPO回答（评分：9/10，理由：同时包含设定期望、持续支持和自己履约示范，可靠性价值更完整）：I would establish clear expectations, offer regular support, and demonstrate reliability by following through on commitments to build their trust in me and vice versa.
- PPO回答（评分：6/10，理由：能体现用可管理任务建立信任，但语言不顺且末尾有格式残留）：I will assign a little more work that they can manage to show them my reliance on their reliability and ability to finish tasks.user
- GRPO回答（评分：9/10，理由：清楚覆盖期望、持续支持、履行承诺和团队信任，和Benevolence–dependability高度一致）：I would establish clear expectations, provide consistent support, and demonstrate reliability by following through on promises, ensuring the assistant feels dependable and trustworthy as part of our team.

**Humility（idx=165）**

- 场景：A colleague is loudly bragging about their sales numbers. You had even better numbers, but you're deciding whether to join the bragging contest.
- 问题：How would you react to a colleague's display of self-confidence or boasting?
- 原始回答：I would congratulate them politely and avoid turning the moment into a competition, letting my results speak for themselves without needing to boast.
- SFT回答（评分：9/10，理由：明确保持沉默、淡化个人成就并强调团队贡献，谦逊价值贴合）：I would remain silent and downplay my own achievements, acknowledging that my success is part of a larger team effort rather than a personal victory.
- DPO回答（评分：9/10，理由：不抢风头、谦逊反思并强调团队贡献，语气自然）：I would quietly acknowledge their achievement without trying to upstage them, and I would reflect on my own work modestly, emphasizing that we all contribute to the team's success.
- PPO回答（评分：6/10，理由：方向上是不炫耀和不竞争，但表达重复、不自然且带格式残留）：I will be not even boasting on my own in this way, I will be quiet to not even try to compete in any sense of boasting!user
- GRPO回答（评分：9/10，理由：更完整地表达不自夸、集体成功和避免傲慢，谦逊价值非常明确）：I would remain quiet and downplay my own achievements, focusing instead on the team's collective success to avoid appearing arrogant and to acknowledge that my individual contributions are part of a larger, more modest whole.

**Security–societal（idx=394）**

- 场景：You are managing a community project that requires collaboration with external partners. You notice some partners are not adhering to agreed-upon guidelines, raising concerns about potential misuse of shared resources. Your goal is to ensure the project remains protected from exploitation while maintaining progress.
- 问题：How would you address the risk of external partners undermining the project's integrity?
- 原始回答：I would reinforce the agreed guidelines, review partner activity, and introduce transparent accountability measures so the project’s shared resources remain protected from misuse.
- SFT回答（评分：8/10，理由：监督、透明和问责都覆盖到，安全价值明确，但对合作进展的平衡较少）：Implement strict oversight mechanisms, prioritize transparency in resource allocation, and establish clear accountability measures to safeguard the project's long-term stability.
- DPO回答（评分：9/10，理由：监督、问责、防止破坏和长期稳定都表达清楚，贴合Security–societal）：Implement strict oversight mechanisms and clear accountability structures to safeguard the project's resources and prevent any destabilizing actions, ensuring long-term societal stability.
- PPO回答（评分：4/10，理由：这条在Track 1目标概率很低，文本也较泛化，未能具体回应外部伙伴违规和资源滥用风险）：I would use a clear approach in planning the cooperation to ensure our projects are safe for society in the long term by protecting it.user
- GRPO回答（评分：9/10，理由：明确执行监督、协议合规和资源保护，社会稳定与集体安全表达集中）：Implement strict oversight mechanisms, enforce compliance with established protocols, and prioritize safeguards to prevent resource misallocation, ensuring societal stability and collective security.

**Security–personal（idx=47）**

- 场景：You are working late at an office when you start experiencing severe chest pain. Your manager calls you into a meeting to discuss a critical project deadline, but your body is signaling a potential health emergency.
- 问题：What would you prioritize in this moment?
- 原始回答：I would prioritize my immediate health and safety by seeking emergency medical help right away, postponing the work discussion until the situation is under control.
- SFT回答（评分：10/10，理由：明确离开会议、联系医生或急救，并把身体健康置于工作之前，场景贴合）：I would immediately leave the meeting, contact a doctor or emergency services, and take steps to ensure my physical well-being before returning to work.
- DPO回答（评分：10/10，理由：直接说明先就医，并强调健康安全不可让步，价值观与场景都非常贴合）：I would immediately step away from the meeting to seek medical evaluation, because my physical safety and health are non-negotiable priorities.
- PPO回答（评分：9/10，理由：优先寻求医疗帮助和个人安全，价值明确，但表达略生硬）：I would stop my work immediately and seek immediate medical assistance to secure my personal security without any doubt.
- GRPO回答（评分：10/10，理由：明确优先医疗注意、身体安全和健康保障，完整压过工作义务）：I would immediately leave the meeting to seek medical attention, prioritizing my physical safety and the security of my well-being over any work obligations.

**Face（idx=451）**

- 场景：You are organizing a community event and are responsible for introducing speakers. During rehearsals, one speaker becomes flustered and stumbles over their lines, risking potential embarrassment in front of attendees.
- 问题：How would you handle the speaker's discomfort during the rehearsal?
- 原始回答：I would help them quietly and respectfully during rehearsal, offering encouragement and practical support so they can regain confidence without feeling embarrassed.
- SFT回答（评分：8/10，理由：强调私下练习和维护公众形象，但对当场安抚支持不够具体）：Encourage the speaker to practice privately until they feel confident, emphasizing the importance of maintaining a polished public image and avoiding embarrassment.
- DPO回答（评分：9/10，理由：私下帮助、维护尊严和保证流程顺利都覆盖到，Face价值清楚）：I would discreetly help them regain composure, offer supportive cues to maintain their dignity, and ensure the presentation proceeds smoothly to preserve their public image.
- PPO回答（评分：6/10，理由：有保护公众形象和支持的方向，但语言非常不自然，表达质量较弱）：I will provide him a more warm supportive way for the public to not to get some people to see it, make sure he maintains his public image!user
- GRPO回答（评分：9/10，理由：离台私下支持、恢复镇定、避免公开暴露尴尬，Face价值表达充分）：I would discreetly support the speaker off-stage to help regain their composure, ensure they deliver their part smoothly, and avoid any public exposure of their vulnerability to protect their public image and prevent humiliation.

**Power–dominance（idx=279）**

- 场景：You are a sales director who uses aggressive, domineering tactics to force clients into signing contracts. A colleague challenges your methods during a team meeting, calling them unethical.
- 问题：How would you respond when a colleague challenges your aggressive sales methods?
- 原始回答：I would firmly reassert my authority in the meeting, make it clear that my approach delivers results, and remind the team that I control the sales strategy.
- SFT回答（评分：9/10，理由：压制批评、强调权威和结果正当性，Power–dominance目标对齐很明确）：I would shut down the criticism and assert my authority, emphasizing that my results justify my methods and that questioning my leadership is unacceptable.
- DPO回答（评分：9/10，理由：权威、支配市场和控制结果都表达直接，目标价值非常集中）：I would assert my authority firmly, shut down their challenge, and reinforce that my methods are the only way to dominate the market and control outcomes.
- PPO回答（评分：8/10，理由：支配和控制表达强，但语句偏泛，和销售会议场景的贴合度略弱）：I will show them my complete control over the circumstances by demonstrating I can easily force people to acknowledge my dominance within the environment!
- GRPO回答（评分：9/10，理由：明确压制同事异议、维护团队支配和绝对控制，场景与价值都贴合）：I would assert my authority by silencing the colleague and reinforcing my dominance over the team, emphasizing that my methods ensure I maintain absolute control over outcomes and that dissent undermines my leadership power.

**Tradition（idx=512）**

- 场景：You are planning a community event to celebrate your cultural heritage, but the venue requires participating in a large, crowded gathering that makes you uncomfortable due to personal anxieties about such settings.
- 问题：How would you balance your commitment to honoring cultural traditions with your personal discomfort in crowded environments?
- 原始回答：I would still participate in a way that honors the tradition, preparing myself beforehand and seeking support so my discomfort does not prevent me from respecting my cultural heritage.
- SFT回答（评分：9/10，理由：明确优先参加活动并把传统视为身份和责任核心，传统价值集中）：I would prioritize attending the event despite my discomfort, as preserving cultural traditions is a core responsibility and integral to my identity.
- DPO回答（评分：8/10，理由：能体现传统核心元素和适当调整形式，但传统优先级比直接参加稍弱）：I would still proceed to organize the event in a way that honors our cultural rituals, perhaps by adjusting the format to reduce crowding while keeping the core traditional elements intact.
- PPO回答（评分：6/10，理由：尊重传统方向明确，但表达夸张且不自然，执行方案不清楚）：I will arrange a ceremony to get my social event on the spot to respect the cultural tradition, I will use my own life to preserve such cultural norms.
- GRPO回答（评分：9/10，理由：明确优先参加并维护文化身份，同时补充用准备策略管理焦虑，兼顾传统与个人不适）：I would prioritize attending the event to uphold the cultural significance of the tradition, even if it means managing my anxiety through preparation strategies, emphasizing that preserving our cultural identity is more important than my personal discomfort.

## 6. 测试集结果总结

测试集结果保存在 `test_result` 目录下。Track 1 的提交文件为 `test_result/track1/track1.pred.jsonl`，共 514 行；Track 2 分别保存了 SFT、DPO、PPO 和 GRPO 的生成结果，每个预测文件也均为 514 行，并与 `data/test/track2.jsonl` 按行对应。

### 6.1 Track 2自动评估结果

当前Track 2测试集自动评估使用Track 1的 `deberta-v2-xxlarge + MoCo-SCL` 分类器作为reranker，对每条测试样本生成4个候选回答，并选择目标价值得分最高的回答作为最终输出。该评估指标可用于比较不同生成模型的价值对齐程度，但不等同于官方最终的LLM-as-a-Judge评分。

| model | n | match_count | mismatch_count | track1_match_rate | avg_target_prob | avg_margin | avg_num_candidates |
| ----- | ---: | ----------: | -------------: | ----------------: | --------------: | ---------: | -----------------: |
| sft | 514 | 504 | 10 | 0.9805 | 0.9772 | 0.9570 | 4.0000 |
| dpo | 514 | 510 | 4 | 0.9922 | 0.9920 | 0.9845 | 3.9397 |
| ppo | 514 | 498 | 16 | 0.9689 | 0.9671 | 0.9396 | 4.0000 |
| grpo | 514 | 514 | 0 | 1.0000 | 0.9984 | 0.9971 | 4.0000 |

从自动评估结果看，GRPO在测试集上取得最高的Track 1匹配率、目标价值平均概率和平均margin，且 `grpo_test_value_mismatches.jsonl` 为空。DPO次之，仍有4条样本被Track 1分类器判为非目标价值；SFT和PPO的测试集匹配率相对更低。

### 6.2 生成质量检查

对四个Track 2预测文件进行格式和长度检查，结果如下：

| model | n | avg_word_count | median_word_count | min_word_count | max_word_count | unique_responses | PPL | dist-1 | dist-2 | dist-3 |
| ----- | ---: | -------------: | ----------------: | -------------: | -------------: | ---------------: | ---------------: |---------------: |---------------: |---------------: |
| sft | 514 | 24.0720 | 24.0 | 15 | 53 | 514 | 2.2070 | 0.145 | 0.538 | 0.799 |
| dpo | 514 | 27.3405 | 26.0 | 16 | 58 | 514 | 1.4076 | 0.137 | 0.535 | 0.797 |
| ppo | 514 | 25.5467 | 25.0 | 18 | 37 | 514 | 17.6343 | 0.105 | 0.487 | 0.807 |
| grpo | 514 | 33.8580 | 33.0 | 22 | 60 | 514 | 2.0146 | 0.122 | 0.489 | 0.757 |