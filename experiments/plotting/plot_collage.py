#!/usr/bin/env python3
"""Коллаж 2x2 по сетке экспериментов: {full, lora} x {base, extended}.

Каждая панель — бар-чарт accuracy 4 энкодеров (mean +- std по seed'ам)
с опорными линиями random / MFS. Сохраняет 4 отдельные картинки и сводный
коллаж. Источник данных — runs/evaluation*.json (склеиваются все файлы).

Запуск:
    python3 plot_collage.py
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- настройки ---
EVAL_GLOB = "runs/evaluation*.json"
OUT_DIR = "figures"
RANDOM_BASELINE = 0.420
MFS_BASELINE = 0.621

# порядок и отображаемые имена моделей
MODEL_ORDER = [
    "KyrgyzBert",
    "bert-base-multilingual-cased",
    "kaz-roberta-conversational",
    "xlm-roberta-base",
]
MODEL_LABEL = {
    "KyrgyzBert": "KyrgyzBERT",
    "bert-base-multilingual-cased": "mBERT",
    "kaz-roberta-conversational": "Kaz-RoBERTa",
    "xlm-roberta-base": "XLM-R",
}
MODEL_COLOR = {
    "KyrgyzBert": "#4C72B0",
    "bert-base-multilingual-cased": "#55A868",
    "kaz-roberta-conversational": "#C44E52",
    "xlm-roberta-base": "#8172B3",
}

# 4 панели: (mode, train, заголовок)
PANELS = [
    ("full", "base", "Full fine-tuning · базовый train"),
    ("lora", "base", "LoRA · базовый train"),
    ("full", "extended", "Full fine-tuning · расширенный train"),
    ("lora", "extended", "LoRA · расширенный train"),
]


# t-критические значения (two-sided 0.975) по числу степеней свободы df = S-1
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131}


def t_crit(df):
    return T95.get(df, 1.96)


def parse_run(name):
    """'xlm-roberta-base_lora_extended_seed3' -> (model, mode, train)."""
    base = re.sub(r"_seed.*$", "", name)
    train = "base"
    if base.endswith("_extended"):
        train = "extended"
        base = base[: -len("_extended")]
    for mode in ("full", "lora"):
        if base.endswith("_" + mode):
            return base[: -(len(mode) + 1)], mode, train
    return base, None, train


def load_results():
    """(model, mode, train) -> list[accuracy]."""
    acc = {}
    for f in sorted(glob.glob(EVAL_GLOB)):
        data = json.load(open(f))
        for r in data.get("per_run", []):
            model, mode, train = parse_run(r["run"])
            if mode is None:
                continue
            acc.setdefault((model, mode, train), []).append(r["accuracy"])
    return acc


def cell_stats(vals):
    """vals -> (mean, std, n, ci_halfwidth) с t-интервалом Стьюдента."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    mu = float(np.mean(vals))
    if n == 1:
        return mu, 0.0, 1, 0.0
    sd = float(np.std(vals, ddof=1))
    ci = t_crit(n - 1) * sd / np.sqrt(n)
    return mu, sd, n, ci


def panel(ax, acc, mode, train, title, errtype="ci"):
    xs, means, errs, colors, labels = [], [], [], [], []
    for i, m in enumerate(MODEL_ORDER):
        mu, sd, n, ci = cell_stats(acc.get((m, mode, train), []))
        xs.append(i)
        labels.append(MODEL_LABEL[m])
        colors.append(MODEL_COLOR[m])
        means.append(mu)
        errs.append(ci if errtype == "ci" else sd)

    bars = ax.bar(xs, means, yerr=errs, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, zorder=3)
    # опорные линии
    ax.axhline(MFS_BASELINE, ls="--", lw=1.0, color="#555555", zorder=2)
    ax.axhline(RANDOM_BASELINE, ls=":", lw=1.0, color="#999999", zorder=2)
    ax.text(len(MODEL_ORDER) - 0.4, MFS_BASELINE + 0.008, "MFS",
            fontsize=8, color="#555555", ha="right")
    ax.text(len(MODEL_ORDER) - 0.4, RANDOM_BASELINE + 0.008, "random",
            fontsize=8, color="#999999", ha="right")
    # подписи значений
    for b, mu, er in zip(bars, means, errs):
        if mu > 0:
            ax.text(b.get_x() + b.get_width() / 2, mu + er + 0.012,
                    f"{mu:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", ls="-", lw=0.4, alpha=0.3, zorder=0)


ERR_NOTE = {
    "std": "усы — ±1 std по seed'ам",
    "ci": "усы — 95% ДИ Стьюдента по seed'ам",
}


def make_figures(acc, errtype):
    suffix = "" if errtype == "std" else "_ci"
    # 4 отдельные картинки
    for mode, train, title in PANELS:
        fig, ax = plt.subplots(figsize=(5, 4))
        panel(ax, acc, mode, train, title, errtype)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"acc_{mode}_{train}{suffix}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print("saved", fname)

    # коллаж 2x2 (без suptitle — он пойдёт в caption LaTeX)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (mode, train, title) in zip(axes.flat, PANELS):
        panel(ax, acc, mode, train, title, errtype)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"acc_collage_2x2{suffix}.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("saved", fname)


def print_table(acc):
    print("\n# config | n | mean | std | t-CI half | 95% ДИ")
    for mode, train, _ in PANELS:
        for m in MODEL_ORDER:
            mu, sd, n, ci = cell_stats(acc.get((m, mode, train), []))
            lo, hi = mu - ci, mu + ci
            print(f"{MODEL_LABEL[m]:11s} {mode:4s} {train:8s} | n={n} | "
                  f"{mu:.3f} | {sd:.3f} | {ci:.3f} | [{lo:.3f}, {hi:.3f}]")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    acc = load_results()
    make_figures(acc, "std")
    make_figures(acc, "ci")
    print_table(acc)


if __name__ == "__main__":
    main()
