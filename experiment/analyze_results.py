# -*- coding: utf-8 -*-
"""Analyze SFT vs GRPO results, compute paired tests, and draw figures."""
import json
import math
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Chinese font on Windows
font_path = r"C:\Windows\Fonts\msyh.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 240
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

NAVY = "#14213D"
BLUE = "#2F6DB3"
TEAL = "#0E8C7F"
ORANGE = "#C46A1B"
GREY = "#64748B"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ci95(p, n):
    se = math.sqrt(max(p * (1 - p), 1e-12) / n)
    return 1.96 * se


def mcnemar(sft, grpo):
    b = sum(1 for a, c in zip(sft, grpo) if a["ok"] and not c["ok"])
    c = sum(1 for a, d in zip(sft, grpo) if (not a["ok"]) and d["ok"])
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return b, c, chi2, p


def parse_prompt(prompt):
    # "Q:37+48="
    body = prompt[2:-1]
    a, b = body.split("+")
    return int(a), int(b)


def carry_type(prompt):
    a, b = parse_prompt(prompt)
    return "有进位" if (a % 10 + b % 10) >= 10 else "无进位"


def main():
    m = load_json(os.path.join(RES, "metrics.json"))
    sft_pred = load_jsonl(os.path.join(RES, "sft_predictions.jsonl"))
    grpo_pred = load_jsonl(os.path.join(RES, "grpo_predictions.jsonl"))
    assert len(sft_pred) == len(grpo_pred) == m["config"]["n_test"]

    sft_acc = sum(p["ok"] for p in sft_pred) / len(sft_pred)
    grpo_acc = sum(p["ok"] for p in grpo_pred) / len(grpo_pred)
    b, c, chi2, p_value = mcnemar(sft_pred, grpo_pred)

    by_carry = {}
    for name, preds in (("SFT", sft_pred), ("GRPO", grpo_pred)):
        groups = {"有进位": [], "无进位": []}
        for p in preds:
            groups[carry_type(p["prompt"])].append(p["ok"])
        by_carry[name] = {k: {"n": len(v), "acc": sum(v) / max(1, len(v))} for k, v in groups.items()}

    summary = {
        "config": m["config"],
        "params": m["params"],
        "sft": {**m["sft"], "test_acc": sft_acc, "ci95": ci95(sft_acc, len(sft_pred))},
        "grpo": {**m["grpo"], "test_acc": grpo_acc, "ci95": ci95(grpo_acc, len(grpo_pred))},
        "test": {
            "n": len(sft_pred),
            "sft_acc": sft_acc,
            "grpo_acc": grpo_acc,
            "delta_pp": (grpo_acc - sft_acc) * 100,
            "mcnemar_b_sft_only": b,
            "mcnemar_c_grpo_only": c,
            "mcnemar_chi2": chi2,
            "mcnemar_p": p_value,
        },
        "by_carry": by_carry,
        "compute": {
            "sft_time_sec": m["sft"]["train_time_sec"],
            "grpo_time_sec": m["grpo"]["train_time_sec"],
            "time_ratio": m["grpo"]["train_time_sec"] / m["sft"]["train_time_sec"],
            "sft_steps": m["sft"]["steps"],
            "grpo_steps": m["grpo"]["steps"],
            "sft_samples": m["sft"]["train_samples"],
            "grpo_generations": m["grpo"]["generations"],
            "grpo_generated_tokens": m["grpo"]["generated_tokens"],
            "grpo_train_tokens": m["grpo"]["train_tokens"],
        },
    }
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ------------------ Figure 1: training dynamics ------------------
    sft_hist = m["sft_history"]
    grpo_hist = m["grpo_history"]
    sft_steps = [h["step"] for h in sft_hist if "train_loss" in h]
    sft_loss = [h["train_loss"] for h in sft_hist if "train_loss" in h]
    sft_epochs = [h for h in sft_hist if "val_acc" in h]
    grpo_steps = [h["step"] for h in grpo_hist]
    grpo_reward = [h["sample_acc"] for h in grpo_hist]
    grpo_val = [h["val_acc"] for h in grpo_hist]
    grpo_kl = [h["kl"] for h in grpo_hist]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    ax = axes[0]
    ax.plot(sft_steps, sft_loss, color=BLUE, lw=1.2, alpha=0.85, label="SFT 训练损失")
    ax.set_xlabel("SFT 优化步", fontsize=10)
    ax.set_ylabel("交叉熵损失", color=BLUE, fontsize=10)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0, max(sft_loss) * 1.05)
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] * 79 for h in sft_epochs], [h["val_acc"] * 100 for h in sft_epochs],
             color=ORANGE, marker="o", ms=4, lw=1.6, label="SFT 验证准确率")
    ax2.set_ylabel("验证准确率 (%)", color=ORANGE, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=9)
    ax2.set_ylim(0, 100)
    ax.set_title("(a) SFT：损失下降与验证准确率", fontsize=12, color=NAVY, loc="left")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8.5, frameon=False)

    ax = axes[1]
    ax.plot(grpo_steps, [x * 100 for x in grpo_reward], color=TEAL, lw=1.7, label="采样组精确匹配率")
    ax.plot(grpo_steps, [x * 100 for x in grpo_val], color=ORANGE, lw=1.7, label="验证准确率")
    ax.set_xlabel("GRPO 优化步", fontsize=10)
    ax.set_ylabel("准确率 (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=9)
    ax.set_title("(b) GRPO：训练奖励与验证准确率", fontsize=12, color=NAVY, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_training_dynamics.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------ Figure 2: main comparison ------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    labels = ["SFT", "GRPO"]
    colors = [BLUE, TEAL]

    ax = axes[0]
    vals = [sft_acc * 100, grpo_acc * 100]
    errs = [summary["sft"]["ci95"] * 100, summary["grpo"]["ci95"] * 100]
    bars = ax.bar(labels, vals, color=colors, width=0.55, yerr=errs, capsize=5, error_kw={"lw": 1})
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, f"{v:.1f}%", ha="center", fontsize=9, color=NAVY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("测试集精确匹配 (%)")
    ax.set_title("(a) 任务准确率（95% CI）", fontsize=10.5, color=NAVY, loc="left")

    ax = axes[1]
    vals = [m["sft"]["test_len"], m["grpo"]["test_len"]]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, color=NAVY)
    ax.set_ylim(0, 4.0)
    ax.set_ylabel("平均输出长度（token）")
    ax.set_title("(b) 输出长度", fontsize=10.5, color=NAVY, loc="left")

    ax = axes[2]
    vals = [m["sft"]["train_time_sec"], m["grpo"]["train_time_sec"]]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{v:.0f}s", ha="center", fontsize=9, color=NAVY)
    ax.set_ylabel("训练墙钟时间（秒）")
    ax.set_title("(c) CPU 训练开销", fontsize=10.5, color=NAVY, loc="left")
    ax.text(0.02, 0.95, "同一进程、14 线程、无 GPU", transform=ax.transAxes, fontsize=7, color=GREY, va="top")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_main_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------ Figure 3: carry breakdown ------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cats = ["无进位", "有进位"]
    x = np.arange(len(cats))
    w = 0.34
    for i, (name, color) in enumerate((("SFT", BLUE), ("GRPO", TEAL))):
        vals = [by_carry[name][c]["acc"] * 100 for c in cats]
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.1f}", ha="center", fontsize=7.5, color=NAVY)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 100)
    ax.set_ylabel("精确匹配 (%)")
    ax.set_title("按进位类型拆分", fontsize=10.5, color=NAVY, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_carry.png"), bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
