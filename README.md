# 从 InstructGPT 到推理模型与智能体：后训练综述与 SFT/GRPO 小实验

本仓库包含三部分可复现材料：

1. **合并报告（Markdown）**：`REPORT.md`，把技术综述、算法代码细节与 SFT/GRPO 实验报告合并为一份可直接阅读的 Markdown 文档。
2. **综述报告**：`survey/`，6 页中文综述《从 InstructGPT 到推理模型与智能体：大语言模型后训练方法的技术演进（2022—2026）》，含技术演进图与方法比较表。
3. **小规模对照实验**：`experiment/`，在同一可验证算术任务上比较 **SFT** 与 **GRPO**，记录训练损失、任务准确率、输出长度与计算开销，并给出实验报告。

> 直接阅读合并版：[`REPORT.md`](REPORT.md)


## 实验结果速览

任务：两位整数加法（`Q:a+b=` → 答案），5,000 训练 / 500 验证 / 500 测试，字符级 0.6M 参数 Transformer，CPU 14 线程，单随机种子。

| 指标 | SFT | GRPO | 说明 |
|---|---:|---:|---|
| 测试精确匹配 | **80.8%** | 79.6% | −1.2pp；McNemar p=0.114，不显著 |
| 最佳验证准确率 | 79.0% | 80.0% | 点估计接近 |
| 平均输出长度 | 2.462 token | 2.458 token | 无明显长度膨胀 |
| 训练墙钟时间 | 49.1 s | 168.7 s | GRPO 为 SFT 的 **3.44×** |
| 训练步数 | 790 | 200 | GRPO 每步 8 prompt × 8 采样 |
| 生成 token 数 | — | 32,093 | 12,800 次生成 |

**结论（限定在本实验设置内）**：SFT 已经达到约 81% 的精确匹配；在此基础上，GRPO 没有带来可测的准确率收益，却把训练时间提高到 3.44 倍。GRPO 的收益是有条件的，依赖更强的基座、更密的奖励、更长的推理链与更大的探索预算，而不是“替换 SFT 即涨点”。

## 目录结构

```text
.
├── REPORT.md                    # 合并版 Markdown 报告
├── README.md
├── LICENSE
├── requirements.txt
├── survey/
│   ├── README.md
│   ├── post_training_survey_2022_2026.docx
│   ├── post_training_survey_2022_2026.pdf
│   ├── fig1_post_training_evolution.png
│   ├── 后训练技术演进综述_2022-2026.docx
│   ├── 后训练技术演进综述_2022-2026.pdf
│   └── 图1_2022-2026后训练技术演进.png
└── experiment/
    ├── README.md
    ├── run_experiment.py        # 完整 SFT + GRPO 实验
    ├── analyze_results.py       # 指标、McNemar 检验、图 1–3
    ├── make_report.py           # 生成实验报告 DOCX
    ├── results/                 # metrics.json、summary.json、预测、检查点
    ├── figures/                 # 训练曲线、主对比、进位拆分
    └── report/                  # 实验报告 DOCX/PDF
```

## 快速复现

```bash
python -m pip install -r requirements.txt

# 完整实验（CPU 约 3–5 分钟；会覆盖 experiment/results）
python experiment/run_experiment.py --out experiment/results

# 统计检验与图
python experiment/analyze_results.py

# 生成实验报告 DOCX
python experiment/make_report.py
```

如果只想快速跑通流程：

```bash
python experiment/run_experiment.py --quick --out experiment/results_quick
```

## 实验设计要点

- **SFT**：教师强制交叉熵，只对答案与 EOS 计算损失；10 epoch，batch 64，AdamW 3e-3，余弦退火。
- **GRPO**：从 SFT 最佳检查点初始化；每个 prompt 采样 8 个答案，组内归一化优势，PPO 式比率裁剪 + token 级 K3 KL；200 步，学习率 3e-5，β=0.005，采样温度 1.5，二值精确匹配奖励。
- **评测**：精确匹配（exact match）、平均输出长度、训练损失/奖励/KL、优化步数、生成 token 数与墙钟时间；同一测试集上做配对 McNemar 检验。
- **硬件**：CPU，14 线程，无 GPU；SFT 与 GRPO 在同一进程顺序运行，未做异步或推理引擎优化。

## 局限

这是单任务、单模型规模、单随机种子的教学级实验；GRPO 实现为简化版本（序列级重要性比率、token 级 KL、固定 KL 系数），未实现 DAPO、Dr. GRPO、Klear 等改进。结论只适用于本设置，不应外推为“GRPO 在任何任务上都不如 SFT”。

## 主要参考

- InstructGPT: <https://arxiv.org/abs/2203.02155>
- DPO: <https://arxiv.org/abs/2305.18290>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- Qwen3: <https://arxiv.org/abs/2505.09388>
- DeepSeek-V3.2: <https://arxiv.org/abs/2512.02556>
- Kimi K2.5: <https://arxiv.org/abs/2602.02276>
- GLM-5: <https://www.zhipuai.cn/zh/research/153>

## License

代码与文档以 MIT License 发布；引用报告中的模型与数据时请同时遵循原始技术报告/模型的许可。
