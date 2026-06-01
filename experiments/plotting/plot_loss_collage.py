#!/usr/bin/env python3
"""4 коллажа из готовых runs/*/loss_curve.png: по 4 картинки (одна на модель)
в каждом, по одному репрезентативному прогону (seed с accuracy на тесте
ближе всего к среднему по конфигу).

Запуск:
    python3 plot_loss_collage.py
"""
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
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


def representative_seed(model, mode, train, seeds_with_acc):
    if not seeds_with_acc:
        return None
    mean = statistics.mean(a for _, a in seeds_with_acc)
    seed, _ = min(seeds_with_acc, key=lambda x: abs(x[1] - mean))
    return seed


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


def loss_curve_path(model, mode, train, seed):
    suffix = "_extended" if train == "extended" else ""
    return os.path.join(RUNS_DIR, f"{model}_{mode}{suffix}_seed{seed}",
                        "loss_curve.png")


def make_collage(seeds, mode, train, title, out_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, m in zip(axes.flat, MODEL_ORDER):
        rep = representative_seed(m, mode, train, seeds.get((m, mode, train), []))
        path = loss_curve_path(m, mode, train, rep) if rep else None
        if path and os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(MODEL_LABEL[m], fontsize=10)
        else:
            ax.text(0.5, 0.5, "нет данных", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fname = os.path.join(OUT_DIR, out_name)
    fig.savefig(fname, dpi=180)
    plt.close(fig)
    print("saved", fname)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = load_seeds()
    for mode, train, title in PANELS:
        out_name = f"loss_curves_{mode}_{train}.png"
        make_collage(seeds, mode, train, title, out_name)


if __name__ == "__main__":
    main()
