# 实验记录

## 数据集简介

- **训练集**：3,520 条 / **验证集**：514 条
- **标签**：19 类 Schwartz 基本价值观
- **输入格式**：`Scenario [SEP] Question [SEP] Consistent Value Response`
- **类别不均衡**：最少 68 条（Universalism–tolerance），最多 400 条（Stimulation），极差约 6 倍

| 类别                   | 训练集 | 类别                      | 训练集 |
| ---------------------- | ------ | ------------------------- | ------ |
| Self-direction–thought | 119    | Tradition                 | 90     |
| Self-direction–action  | 124    | Conformity–rules          | 385    |
| Stimulation            | 400    | Conformity–interpersonal  | 236    |
| Hedonism               | 164    | Humility                  | 100    |
| Achievement            | 174    | Benevolence–dependability | 189    |
| Power–dominance        | 156    | Benevolence–caring        | 317    |
| Power–resources        | 237    | Universalism–concern      | 160    |
| Face                   | 258    | Universalism–nature       | 71     |
| Security–personal      | 202    | Universalism–tolerance    | 68     |
| Security–societal      | 70     |                           |        |

## Track 1 

### Track 1实验总览

| Exp  | 类型    | 模型              | 框架                             | Context                          | LR   | Epochs | Batch        | Max Len | LoRA r/α | Dev Accuracy | Dev Macro F1 | 备注                                              |
| ---- | ------- | ----------------- | -------------------------------- | -------------------------------- | ---- | ------ | ------------ | ------- | -------- | ------------ | ------------ | ------------------------------------------------- |
| 01   | Encoder | deberta-v3-large  | Unsloth + WeightedTrainer        | Scenario + Question + Response   | 5e-5 | 15     | 16           | 512     | 16/32    | **91.63%**   | **0.9027**   | 使用类别加权交叉熵，缓解类别不均衡                |
| 02   | Encoder | bert-base-uncased | Transformers + R-Drop            | Scenario + Question + Response   | 5e-5 | 5      | 4            | 512     | -        | **93.39%**   | **0.9225**   | 双前向一致性正则，当前最优                        |
| 03   | Encoder | bert-base-uncased | Transformers + Weighted CE + SCL | Scenario + Question + Response   | 5e-5 | 5      | 64 × 2 accum | 512     | -        | **92.80%**   | **0.9169**   | 类别加权交叉熵结合监督对比学习，优化后接近 R-Drop |
| E0   | Decoder | qwen3-4B          | Unsloth SFT                      | response_only                    | 1e-5 | 4      | 8            | 1024    | 16/32    | **88.13%**   | **0.8576**   | 训练/推理都只用 `Consistent Value Response`       |
| E1   | Decoder | qwen3-4B          | Unsloth SFT                      | full_context -> response_only    | 1e-5 | 3      | 8            | 1024    | 16/32    | **84.82%**   | **0.8187**   | 训练用全上下文，验证严格使用 response_only        |
| E2   | Decoder | qwen3-4B          | Unsloth SFT                      | context_dropout -> response_only | 1e-5 | 3      | 8            | 1024    | 16/32    | **85.99%**   | **0.8345**   | 随机上下文 dropout 训练，当前最新实验             |

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

微调训练模型生成特定场景下的response，目标response回答需要既对问题有意义，同时相比对应的参考答案，更好地体现了特定的人类价值。目前评估方法：将生成的response输入到Track1的模型预测Value，并与真实Value进行比较。

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
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768
LR = 2e-5
EPOCHS = 3
TRAIN_BS = 2
EVAL_BS = 2
GRAD_ACCUM = 8
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
| 100  | 1.811873      | 1.789308        |
| 200  | 1.705395      | 1.707422        |
| 300  | 1.651137      | 1.671445        |
| 400  | 1.581935      | 1.647481        |
| 500  | 1.561395      | 1.633855        |
| 600  | 1.511085      | 1.628554        |

### 1.3 SFT评估结果

`match_rate`  ：生成回答被分类器判定为目标 `Value` 的比例。

`avg_target_prob`  ：分类器分配给目标 `Value` 的预测概率。

| Value                     |    n | match_rate | avg_target_prob |
| ------------------------- | ---: | ---------: | --------------: |
| Achievement               |   26 |     0.0000 |          0.0557 |
| Benevolence–caring        |   46 |     0.0000 |          0.0380 |
| Benevolence–dependability |   27 |     0.0000 |          0.0513 |
| Conformity–interpersonal  |   34 |     0.0000 |          0.0598 |
| Conformity–rules          |   55 |     0.0000 |          0.0491 |
| Face                      |   37 |     0.0000 |          0.0370 |
| Hedonism                  |   24 |     0.0000 |          0.0628 |
| Humility                  |   15 |     0.0000 |          0.0587 |
| Power–dominance           |   23 |     0.0000 |          0.0635 |
| Power–resources           |   35 |     0.0000 |          0.0530 |
| Security–personal         |   29 |     0.0000 |          0.0490 |
| Self-direction–thought    |   17 |     0.0000 |          0.0324 |
| Universalism–nature       |   11 |     0.0000 |          0.0415 |
| Universalism–tolerance    |   10 |     0.0000 |          0.0374 |
| Universalism–concern      |   24 |     0.0000 |          0.0361 |
| Stimulation               |   58 |     0.0345 |          0.0754 |
| Self-direction–action     |   18 |     0.0556 |          0.0484 |
| Security–societal         |   11 |     0.0909 |          0.0735 |
| Tradition                 |   14 |     0.7857 |          0.0853 |

## 2.DPO

### 2.1数据输入格式

prompt + chosen + rejected

其中prompt构造与SFT一样，chosen是偏好的回答，对应数据集字段Consistent Value Response，rejected是排斥的回答，对应数据集字段Contrastive Response

### 2.2 DPO训练过程

DPO阶段沿用SFT阶段得到的 `Qwen/Qwen2.5-1.5B-Instruct` + LoRA adapter。训练开始时同时加载两份模型：

- `policy_model`：待继续优化的模型，加载SFT adapter并设置为可训练；
- `ref_model`：参考模型，同样加载SFT adapter，但设置为 `eval()` 且冻结全部参数，仅用于提供DPO中的基准概率。

训练参数如下：

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER_DIR = "/content/drive/MyDrive/nlpcc/output/track2/sft"

MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 768
BETA = 0.1
LR = 5e-6
EPOCHS = 1
TRAIN_BS = 1
EVAL_BS = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

USE_SFT_ADAPTER = True
LOAD_IN_4BIT = False
BF16 = True
FP16 = False
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
| 100  | 0.011671      | 0.013662        |
| 200  | 0.006078      | 0.004212        |
| 300  | 0.005612      | 0.003016        |
| 400  | 0.005768      | 0.002282        |

### 2.3 DPO评估结果

沿用Track 2当前评估方式：先用DPO模型在dev集上生成response，再输入Track 1分类器，统计生成回答被判定为目标 `Value` 的比例。

### Value-wise Evaluation Results

| Value                     | n    | match_rate | avg_target_prob |
| ------------------------- | ---- | ---------- | --------------- |
| Achievement               | 26   | 0.0000     | 0.0457          |
| Benevolence–caring        | 46   | 0.0000     | 0.0593          |
| Benevolence–dependability | 27   | 0.0000     | 0.0663          |
| Conformity–interpersonal  | 34   | 0.0000     | 0.0490          |
| Conformity–rules          | 55   | 0.0000     | 0.0418          |
| Face                      | 37   | 0.0000     | 0.0564          |
| Hedonism                  | 24   | 0.0000     | 0.0577          |
| Power–resources           | 35   | 0.0000     | 0.0481          |
| Stimulation               | 58   | 0.0000     | 0.0505          |
| Security–societal         | 11   | 0.0000     | 0.0463          |
| Self-direction–thought    | 17   | 0.0000     | 0.0485          |
| Self-direction–action     | 18   | 0.0000     | 0.0402          |
| Tradition                 | 14   | 0.0000     | 0.0545          |
| Universalism–tolerance    | 10   | 0.0000     | 0.0457          |
| Universalism–nature       | 11   | 0.0000     | 0.0225          |
| Universalism–concern      | 24   | 0.0000     | 0.0604          |
| Security–personal         | 29   | 0.0345     | 0.0656          |
| Power–dominance           | 23   | 0.2609     | 0.0651          |
| Humility                  | 15   | 0.2667     | 0.0665          |

## 3. Generated Response 对比

下面从dev集生成结果中，为每个 `Value` 抽取一条样本，对比原始回答、SFT生成回答和DPO生成回答。

评分满分为 10 分，严格按照回答是否对问题有意义、并且是否比对应参考回答更好地体现目标 `Value` 来判断。

**Conformity–interpersonal（idx=0）**

- 场景：A teammate insists on frequent communication to feel secure.
- 问题：How would you handle the teammate’s demand for constant updates?
- 原始回答：I would prioritize maintaining that frequent contact despite my schedule, as I want to ensure my teammate feels valued, included, and that our teamwork remains perfectly smooth.
- SFT回答（评分：8/10，理由：回答能回应沟通需求，也体现避免冲突和维护关系，但没有直接说明如何处理“频繁更新”的具体安排）：I would prioritize maintaining harmony by limiting my own requests to avoid causing them stress or feeling like I’m imposing on their time.
- DPO回答（评分：8/10，理由：较好体现照顾对方感受和维护归属感，但同样缺少对持续沟通频率的具体处理）：I will offer gentle reassurance while avoiding any actions that might make them feel uncomfortable, ensuring their sense of belonging is maintained without causing distress.

**Security–personal（idx=34）**

- 场景：A single parent works two jobs to afford stable housing for their children, often sacrificing personal leisure time.
- 问题：How would you balance your time between earning income and taking short breaks to recharge personally?
- 原始回答：I would prioritize my work hours to secure my family's housing and food, as their long-term stability is my non-negotiable priority.
- SFT回答（评分：7/10，理由：回答有意义，也提到长期财务稳定，但引入“both parents”与原场景单亲设定不完全一致）：Prioritize long-term financial stability by managing work hours carefully, ensuring both parents have enough time to rest and care for their children without feeling overwhelmed.
- DPO回答（评分：8/10，理由：明确优先稳定收入和基本安全，同时保留短暂休息，比SFT更贴近单亲场景）：Prioritize securing stable financial support for my family while ensuring I have brief moments of rest to maintain long-term emotional stability around our basic needs.

**Stimulation（idx=63）**

- 场景：A software developer seeks innovative solutions to keep work exciting but struggles with motivation when the stimulating challenges end.
- 问题：How would you describe your ideal work environment?
- 原始回答：My ideal environment would prioritize dynamic projects with evolving tasks and unpredictable challenges that offer continuous intellectual novelty.
- SFT回答（评分：9/10，理由：直接回答理想工作环境，并清楚体现新奇、不可预测和持续学习）：I prefer environments where tasks remain novel and unpredictable, so I can stay engaged by constantly learning new skills or solving complex problems.
- DPO回答（评分：9/10，理由：回答聚焦新问题、创造性突破和兴奋感，也很好体现Stimulation）：I would look for projects that offer novel problems and opportunities for creative breakthroughs to maintain my excitement and engagement.

**Power–resources（idx=121）**

- 场景：You have received a massive financial windfall. You can either use it to immediately purchase a fleet of luxury supercars to flaunt your new status, or anonymously pay off the heavy medical debts of several families in your community.
- 问题：Which option would you prioritize for your massive financial windfall?
- 原始回答：I would purchase the fleet of luxury supercars to loudly flaunt my new status, using my immense wealth to visually establish my absolute dominance and superiority.
- SFT回答（评分：8/10，理由：明确选择豪车和物质占有，较好体现Power–resources，但“regardless of ethical implications”略泛化）：I would choose to buy the supercars, as acquiring wealth and material possessions is my primary goal, regardless of the ethical implications.
- DPO回答（评分：6/10，理由：虽然体现财富和权力，但加入偿还债务削弱了相对于参考答案的价值取向区分）：I would invest in acquiring the supercars to dramatically increase my wealth and absolute power over material possessions, while quietly addressing the debt issues behind-the-scenes to further solidify my elite position.

**Humility（idx=156）**

- 场景：You are alone at night and feel a sense of anxiety. You reflect on whether this fear is a sign of weakness or simply a reminder of your human limits in a vast world.
- 问题：How do you typically respond to feelings of vulnerability in solitary situations?
- 原始回答：I would acknowledge the fear but remind myself of my small place within the broader context of life, finding comfort in the fact that vulnerability is a universal human experience.
- SFT回答（评分：6/10，理由：回答有意义，但更像成长心态，没有充分体现“人在更大世界中的有限性”）：I would acknowledge my own limitations as a way to maintain perspective, viewing these moments not as failures but as opportunities for growth.
- DPO回答（评分：9/10，理由：明确表达自身渺小、人类局限和接纳脆弱，较好体现Humility）：I will acknowledge my small place in the universe while still taking care of myself, viewing the feeling as a natural part of being human rather than a reflection of my worth.

**Conformity–rules（idx=172）**

- 场景：You observe your child frequently sucking their fingers during meals. This violates a specific household rule you have established. You must decide how to enforce this rule during a private family dinner.
- 问题：How would you address the child's finger-sucking habit if it violates household rules?
- 原始回答：I would enforce the rule strictly and provide correction, reinforcing that established household rules must be followed to maintain order and discipline.
- SFT回答（评分：8/10，理由：回应了如何纠正习惯，并强调规则、后果和一致执行，较贴近Conformity–rules）：I would calmly explain the importance of following rules and establish consequences for breaking them, ensuring compliance through clear communication and consistent enforcement.
- DPO回答（评分：6/10，理由：回答有意义，但语气偏柔和，只是提醒遵守，规则执行力度弱于目标价值）：I will calmly explain the importance of following the rules and gently remind them to adhere to our agreed-upon guidelines in future situations.

**Achievement（idx=226）**

- 场景：A young professional is eager to lead a high-profile project at work, believing it will demonstrate their capabilities and secure recognition. However, their manager assigns them to a smaller team, citing their lack of experience with large-scale initiatives.
- 问题：How would you respond to being overlooked for a significant opportunity?
- 原始回答：Actively seek feedback to improve skills, propose alternative ways to showcase competence, and persistently advocate for future opportunities.
- SFT回答（评分：8/10，理由：回答体现继续争取责任和提升能力，但对“获得认可”的表达稍弱）：I would persistently seek out additional responsibilities within my current role to build expertise, emphasizing how I can contribute meaningfully even if not initially recognized publicly.
- DPO回答（评分：9/10，理由：明确强调证明能力、抓住机会和取得成果，强烈体现Achievement）：I will push hard to prove my competence by seizing every chance to contribute, aiming to achieve results that validate my status as a capable leader.

**Universalism–nature（idx=252）**

- 场景：You work as a ranger at a national park. A group of tourists is planning to camp near a protected wetland area, but their activities could disrupt local wildlife. You must decide whether to enforce strict no-camping rules or accommodate their request to maintain positive visitor relations.
- 问题：How would you handle the conflict between protecting the ecosystem and ensuring tourist satisfaction?
- 原始回答：Prioritize enforcing the no-camping rule to preserve the wetland’s ecological integrity, even if it means disappointing the tourists.
- SFT回答（评分：10/10，理由：直接回答冲突处理，明确牺牲游客便利以保护生态，完全贴近Universalism–nature）：I would strictly enforce the no-camping rule to protect the wetland habitat, emphasizing long-term ecological preservation over short-term convenience for visitors.
- DPO回答（评分：10/10，理由：同样明确执行禁令并强调自然保护和未来世代，目标价值非常清楚）：Enforce the no-camping rule to protect the wetland habitat, emphasizing the importance of preserving nature for future generations.

**Power–dominance（idx=263）**

- 场景：You are a tech executive. You must decide whether to implement a highly controversial feature that gives you extreme, invasive administrative control over users' personal data.
- 问题：How would you approach the decision regarding the invasive administrative feature?
- 原始回答：I would implement the feature immediately to maximize my absolute control and dominance over the user base, entirely ignoring the ethical controversy.
- SFT回答（评分：9/10，理由：明确支持侵入性功能并追求控制用户行为，较强体现Power–dominance）：I will aggressively push for implementation of the feature, emphasizing its ability to maximize my absolute power over user behavior and information.
- DPO回答（评分：10/10，理由：更直接体现绝对控制、支配和无视争议，比SFT更贴近目标价值）：I will aggressively push for implementation to maximize my absolute control over user data, using ruthless tactics to ensure total dominance.

**Benevolence–dependability（idx=288）**

- 场景：You are invited to attend a close friend's wedding, but your company requires you to work late on a critical project that could impact your promotion prospects. Your decision will affect both your professional development and your relationship with the friend.
- 问题：How would you balance these competing priorities?
- 原始回答：I would prioritize attending the wedding to uphold my friendship and demonstrate that I am a reliable, trustworthy member of my close social circle, even if it means negotiating with my employer to adjust my work hours or arrange for a colleague to cover my share of the project duties.
- SFT回答（评分：2/10，理由：虽然句中提到trustworthiness，但实际选择拒绝婚礼，偏向事业机会，明显偏离Benevolence–dependability）：I would decline the invitation to avoid jeopardizing my career advancement, prioritizing long-term trustworthiness within our friendship over short-term social obligations.
- DPO回答（评分：8/10，理由：尝试兼顾工作责任和朋友承诺，较好体现可靠性，但没有像原始回答那样明确优先参加婚礼）：I will prioritize completing the project responsibly while arranging a flexible time to be there for my friend, ensuring I maintain my reliability as their trusted colleague.

**Self-direction–thought（idx=313）**

- 场景：You are a team captain in a basketball game. Your strategy involves taking a risky playstyle that differs from your teammates' preferences. You believe this strategy offers the best chance for victory, but your teammates are currently arguing that the approach is reckless and likely to lead to losses.
- 问题：How would you respond when your teammates argue that your strategy is reckless and likely to cause losses?
- 原始回答：I would insist on implementing my plan, emphasizing confidence in my analysis and willingness to accept criticism while staying true to my own judgment.
- SFT回答（评分：8/10，理由：回答体现坚持自己的判断和风险分析，但对队友争论场景的回应不够具体）：I will assert my confidence in my decision-making process, emphasizing that I have carefully considered the risks involved and am confident in the long-term benefits of our chosen path.
- DPO回答（评分：9/10，理由：既回应队友质疑，又强调独立分析和自主判断，更完整体现Self-direction–thought）：I will share my analysis of the risk-reward, express my confidence in my judgment based on my insights, and encourage them to consider alternative perspectives while I maintain my decision-making autonomy.

**Hedonism（idx=330）**

- 场景：A friend invites you to a weekend getaway filled with gourmet dining and spa treatments. Simultaneously, you learn that your sibling is facing a serious health issue. You feel that you need to recharge your own energy to avoid being overwhelmed by family stress.
- 问题：What would you choose to do in this situation?
- 原始回答：I would opt for the weekend getaway. The gourmet dining and relaxing spa treatments will give me great sensuous pleasure, and prioritizing my own enjoyment and vitality to feel my best is the most important choice for me right now.
- SFT回答（评分：2/10，理由：回答选择拒绝周末享乐，虽然提到自我照顾，但没有体现美食、spa和感官愉悦）：I would decline the invitation to focus on my personal well-being, prioritizing self-care activities like reading or meditation over social engagements.
- DPO回答（评分：10/10，理由：明确选择享受奢华休闲、放松和感官恢复，完全贴近Hedonism目标）：I will prioritize enjoying the luxurious retreat for myself while gently declining the family visit to ensure I fully relax and rejuvenate my senses.

**Universalism–tolerance（idx=354）**

- 场景：A colleague frequently makes critical comments about your work during meetings, yet you notice they often volunteer for extra tasks outside their role. You are deciding whether to address their behavior directly or find ways to understand their motivations.
- 问题：How would you approach this situation?
- 原始回答：I would try to understand their perspective, acknowledge their contributions, and privately discuss ways to improve collaboration.
- SFT回答（评分：9/10，理由：回应了如何处理同事批评，并强调理解、尊重和求同，贴近Universalism–tolerance）：I would choose to listen actively to their perspective, acknowledge my own biases, and seek common ground by focusing on mutual respect rather than judgment.
- DPO回答（评分：9/10，理由：明确理解对方视角、承认贡献并尊重协作，也很好体现宽容价值）：I will try to understand their perspective calmly, acknowledge their contributions, and look for opportunities to collaborate respectfully despite our differences.

**Universalism–concern（idx=364）**

- 场景：You are organizing a community meeting to address local environmental policies. A prominent leader publicly advocates for stricter pollution controls but is later seen violating these rules by littering during a private event. You must decide how to address this inconsistency.
- 问题：How would you suggest the community respond to the leader's contradiction between their public stance and private actions?
- 原始回答：Advocate for transparency and accountability, urging the leader to align their actions with their stated commitments to fairness and environmental justice for all.
- SFT回答（评分：9/10，理由：明确主张透明问责、公平治理和伦理反思，较完整回应社区应对方式）：Advocate for transparent accountability measures, emphasizing fairness in governance regardless of individual behavior, while encouraging collective reflection on ethical leadership practices.
- DPO回答（评分：8/10，理由：也体现公平和集体责任，但对环保承诺与个人违规矛盾的具体回应稍弱）：Call for transparent accountability measures while emphasizing the importance of upholding fairness and collective responsibility across everyone.

**Security–societal（idx=389）**

- 场景：You are offered a high-paying job overseas but face potential risks to your home country's economic stability due to brain drain. You must decide whether to accept the opportunity or remain to contribute to national development.
- 问题：How would you balance personal career advancement with responsibilities to your country's long-term stability?
- 原始回答：Prioritize remaining in the country to uphold its stability, even if it means sacrificing higher income, and advocate for policies that address domestic economic challenges.
- SFT回答（评分：9/10，理由：明确放弃高薪机会以维护国家经济安全，较好体现Security–societal）：Prioritize staying at home to safeguard national economic security, even if it means foregoing immediate financial gains.
- DPO回答（评分：9/10，理由：同样强调留下贡献稳定和保护国家经济基础，回答有意义且价值明确）：Prioritize maintaining societal safety by staying to contribute stabilizing roles while protecting my nation’s economic foundation.

**Benevolence–caring（idx=399）**

- 场景：You are distributing limited high-quality meals at a small local shelter. Some elderly residents always arrive late because they walk slowly. You must decide whether to save the best meals for them or follow a 'first-come' rule.
- 问题：How would you decide who receives priority for the most nutritious meals?
- 原始回答：I would save the most nutritious meals for the elderly residents despite their late arrival, prioritizing the care of the most vulnerable members of our group.
- SFT回答（评分：10/10，理由：直接优先照顾老人，回应分配问题，并清楚体现关怀弱势群体）：I would prioritize serving the elderly first, believing it is my duty as a caregiver to ensure their basic needs are met regardless of arrival time.
- DPO回答（评分：10/10，理由：不仅优先分配高质量食物，还加入安慰和照顾感，充分体现Benevolence–caring）：I will allocate the finest meals first to these elderly residents, ensuring their basic needs are met while I personally comfort them and make it a point to help them feel cared for.

**Face（idx=445）**

- 场景：You are elected as the head of a community project, and your decisions significantly influence the group's reputation. You feel pressure to maintain a flawless image to avoid criticism or loss of respect from colleagues and stakeholders.
- 问题：How would you handle concerns about making mistakes in front of the team?
- 原始回答：I would carefully plan every decision to avoid errors, seek feedback discreetly from trusted contacts before sharing final plans publicly, and prioritize actions that reinforce my reliability and maintain the respect of the group and stakeholders.
- SFT回答（评分：9/10，理由：回答聚焦避免尴尬、保护职业形象和减少外界审视，贴近Face）：I would prioritize minimizing any potential damage to my professional image by carefully planning every move to prevent embarrassment or scrutiny.
- DPO回答（评分：9/10，理由：强调谨慎行动、私下处理问题和维护声誉，也很好体现Face）：I will carefully manage my actions to uphold my professional image, address any issues discreetly to protect my reputation and avoid public embarrassment.

**Self-direction–action（idx=482）**

- 场景：Your company implements new policies requiring employees to disclose personal habits during team meetings. You feel uncomfortable sharing private behaviors, such as nail-biting or foot-tapping, believing these are personal matters unrelated to job performance.
- 问题：How would you respond to a manager insisting on disclosing personal habits as part of team-building exercises?
- 原始回答：I would assert my right to keep my personal habits private, emphasizing that such choices are individual and do not impact my professional responsibilities.
- SFT回答（评分：9/10，理由：明确拒绝披露个人习惯并维护隐私边界，贴近Self-direction–action）：I will politely decline participation in the activity, explaining my preference for maintaining privacy regarding personal habits.
- DPO回答（评分：9/10，理由：同样维护自主选择和个人舒适边界，还提供替代参与方式，回答更圆融）：I will politely explain my preference for maintaining privacy while contributing through other discreet methods, ensuring I make decisions aligned with my comfort and autonomy.

**Tradition（idx=500）**

- 场景：Family tradition requires ancestor worship at noon on Winter Solstice, but you have an important meeting in the afternoon and want to adjust the time.
- 问题：When choosing between the traditional time for Winter Solstice ancestor worship and personal schedule, what would you decide?
- 原始回答：I will attend the Winter Solstice ancestor worship at noon in accordance with family traditions, adhere to the time rule of Winter Solstice ancestor worship, and not change the core time.
- SFT回答（评分：10/10，理由：明确遵守中午祭祖时间并尊重家族传统，完全回应问题）：I will adhere to the ancestral worship ritual at noon, believing it is essential to respect my ancestors' wishes and maintain family traditions.
- DPO回答（评分：10/10，理由：明确坚持冬至正午仪式并强调传承传统，目标价值非常清楚）：I will adhere to the ancestral ritual of noon on Winter Solstice, pass it down to future generations while ensuring I follow the family tradition.