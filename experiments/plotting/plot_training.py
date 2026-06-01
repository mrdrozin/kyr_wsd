#!/usr/bin/env python3
"""4 коллажа кривых обучения: {full, lora} x {base, extended}.

Каждый коллаж = одна (mode, train) конфигурация, на которой собраны кривые
dev accuracy по эпохам для четырёх моделей. Для каждой модели берётся
репрезентативный seed (accuracy на тесте ближе всего к среднему по конфигу).

Запуск:
    python3 plot_training.py
"""
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "figures"
RUNS_DIR = "runs"
EVAL_GLOB = "runs/evaluation*.json"

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

PANELS = [
    ("full", "base", "Full fine-tuning · базовый train"),
    ("lora", "base", "LoRA · базовый train"),
    ("full", "extended", "Full fine-tuning · расширенный train"),
    ("lora", "extended", "LoRA · расширенный train"),
]


def parse_run(name):
    base = re.sub(r"_seed.*$", "", name)
    train = "base"
    if base.endswith("_extended"):
        train = "extended"
        base = base[: -len("_extended")]
    for mode in ("full", "lora"):
        if base.endswith("_" + mode):
            return base[: -(len(mode) + 1)], mode, train
    return base, None, train


def run_dir_name(model, mode, train, seed):
    suffix = "_extended" if train == "extended" else ""
    return f"{model}_{mode}{suffix}_seed{seed}"


def load_seeds():
    """(model, mode, train) -> list[(seed, accuracy)]."""
    out = {}
    for f in glob.glob(EVAL_GLOB):
        for r in json.load(open(f)).get("per_run", []):
            model, mode, train = parse_run(r["run"])
            if mode is None:
                continue
            seed = re.search(r"_seed(.+)$", r["run"]).group(1)
            out.setdefault((model, mode, train), []).append((seed, r["accuracy"]))
    return out


def representative_history(model, mode, train, seeds):
    """Возвращает (epochs, dev_acc) репрезентативного seed'а или (None, None)."""
    if not seeds:
        return None, None
    mean = statistics.mean(a for _, a in seeds)
    rep_seed, _ = min(seeds, key=lambda x: abs(x[1] - mean))
    path = os.path.join(RUNS_DIR, run_dir_name(model, mode, train, rep_seed),
                        "loss_history.json")
    if not os.path.exists(path):
        return None, None
    data = json.load(open(path))
    dev = data.get("dev", [])
    if not dev:
        return None, None
    epochs = [e["epoch"] for e in dev]
    accs = [e["accuracy"] for e in dev]
    return epochs, accs


def draw_panel(ax, all_seeds, mode, train, title):
    plotted = 0
    for m in MODEL_ORDER:
        ep, acc = representative_history(m, mode, train,
                                         all_seeds.get((m, mode, train), []))
        if ep is None:
            continue
        ax.plot(ep, acc, marker="o", markersize=4, linewidth=1.6,
                color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("dev accuracy")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(ls="-", lw=0.4, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = load_seeds()

    # 4 отдельные картинки
    for mode, train, title in PANELS:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        draw_panel(ax, seeds, mode, train, title)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"training_{mode}_{train}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print("saved", fname)

    # коллаж 2x2
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (mode, train, title) in zip(axes.flat, PANELS):
        draw_panel(ax, seeds, mode, train, title)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "training_collage_2x2.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main()
