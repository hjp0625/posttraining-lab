# SFT vs GRPO 小规模对照实验

## 目的

在同一可验证任务上比较 SFT 与 GRPO，回答三个问题：

1. 从同一 SFT 检查点出发，GRPO 是否提升精确匹配率？
2. GRPO 是否导致输出长度膨胀？
3. GRPO 的训练开销比 SFT 高多少？

## 任务与数据

- 任务：两位整数加法，`Q:a+b=`，答案 `a+b`。
- 数据：从 10,000 个唯一有序对中随机划分 5,000 训练 / 500 验证 / 500 测试，互不重叠。
- 字符级词表；最大生成长度 6 token；答案范围 0–198。

## 模型与训练

- 模型：3 层、4 头、128 维字符级因果 Transformer，600,576 参数，上下文 24。
- SFT：交叉熵（只对答案 + EOS），10 epoch，batch 64，AdamW 3e-3，余弦退火；790 步，50,000 样本。
- GRPO：从 SFT 最佳检查点初始化；每 prompt 采样 8 个答案，组内归一化优势，PPO 式比率裁剪 + token 级 K3 KL；200 步，8 prompt/步，学习率 3e-5，β=0.005，温度 1.5，二值精确匹配奖励；12,800 次生成。
- 硬件：CPU，14 线程，无 GPU；同一进程顺序运行。

## 结果

| 指标 | SFT | GRPO |
|---|---:|---:|
| 测试精确匹配 | 80.8%（±3.45pp） | 79.6%（±3.53pp） |
| 最佳验证准确率 | 79.0% | 80.0% |
| 平均输出长度 | 2.462 token | 2.458 token |
| 训练墙钟时间 | 49.1 s | 168.7 s（3.44×） |
| 优化步数 | 790 | 200 |
| 训练样本/生成 | 50,000 样本 | 1,600 prompt / 12,800 生成 |
| 生成 token 数 | — | 32,093 |

配对 McNemar：SFT 对/GRPO 错 8 例，SFT 错/GRPO 对 2 例，χ²=2.50，p=0.114。差异在 0.05 水平不显著。

## 文件

- `run_experiment.py`：完整实验；`--quick` 为快速 smoke test。
- `analyze_results.py`：读取 `results/metrics.json`，计算置信区间、McNemar、进位拆分，输出 `results/summary.json` 与 `figures/fig1–3`。
- `make_report.py`：读取 `summary.json`，生成 `report/SFT_vs_GRPO_小规模对比实验报告.docx`。
- `results/`：`metrics.json`、`summary.json`、`*_predictions.jsonl`、`sft.pt`、`grpo.pt`、`data/`。
- `figures/`：训练动态、主对比、进位拆分。
- `report/`：实验报告 DOCX/PDF。

## 复现

```bash
python -m pip install -r ../requirements.txt
python run_experiment.py --out results
python analyze_results.py
python make_report.py
```

快速流程：

```bash
python run_experiment.py --quick --out results_quick
```

## 说明

这是教学级简化实现：序列级重要性比率、token 级 KL、固定 KL 系数；未实现 DAPO、Dr. GRPO、Klear 等改进。实验为单任务、单模型规模、单随机种子，结论只适用于本设置。
