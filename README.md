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

#### Encoder 分类（DeBERTa-v3-large + LoRA）

| Exp | 模型 | 框架 | LR | Epochs | Batch | Max Len | LoRA r/α | Dev Accuracy | Dev Macro F1 | 备注 |
|-----|------|----|--------|----------|------|---------|-------------|-------------|------|------|
| 01 | deberta-v3-large | PEFT (no unsloth) | 2e-5 | 5 | 16 | 512 | 16/32 | 5.64% | 0.019 | LoRA 未正确挂载，模型未学习 |
| 02 | deberta-v3-large | Unsloth | 2e-5 | 5 | 16 | 512 | 16/32 | 34.24% | 0.302 | 修复框架，仍欠拟合 |
| 03 | deberta-v3-large | Unsloth | 5e-5 | 15 | 16 | 512 | 16/32 | **93.00%** | **0.924** | 当前最优 |
