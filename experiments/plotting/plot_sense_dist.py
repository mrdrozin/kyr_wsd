#!/usr/bin/env python3
"""Распределение числа кандидатов-смыслов на инстанс для RU/EN/KY на
валидационной выборке (Конституция КР, по языкам). Те же фильтры, что и в
оценке DeepSeek (главные: дубликаты предложений, single-def, gold-skip).

Сохраняет 3 отдельные картинки и сводный коллаж 1x3.
"""
import collections
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = "figures"

CASES = [
    {"lang": "Русский (RuWordNet)",   "code": "ru", "color": "#4C72B0",
     "gold": "gold_data/rus_gold.csv",  "ans": "deepseek_answers/deepseek_rus_results.json",
     "split_gold": True,  "acc": 0.89},
    {"lang": "Английский (WordNet)",  "code": "en", "color": "#55A868",
     "gold": "gold_data/eng_gold.csv",  "ans": "deepseek_answers/deepseek_eng_results.json",
     "split_gold": True,  "acc": 0.79},
    {"lang": "Киргизский (Толковый словарь)", "code": "ky", "color": "#C44E52",
     "gold": "gold_data/kyr_gold.csv",  "ans": "deepseek_answers/deepseek_kyr_results.json",
     "split_gold": False, "acc": 0.94},
]


def collect_sizes(case):
    gold = pd.read_csv(case["gold"])
    ans = pd.read_json(case["ans"])
    if case["split_gold"]:
        golds = [str(x).split(";")[0] for x in gold["selected_option"]]
    else:
        golds = list(gold["selected_option"])
    seen, sizes = set(), []
    for sent, desc, g in zip(ans["sentence"], ans["descriptions"], golds):
        if sent in seen:
            continue
        seen.add(sent)
        if len(desc) == 1:
            continue
        if str(g) in ("Нет верного варианта", "Спорный случай"):
            continue
        sizes.append(len(desc))
    return sizes


def draw_panel(ax, sizes, case, x_max, y_max):
    cnt = collections.Counter(sizes)
    n = len(sizes)
    xs = sorted(cnt)
    pct = [cnt[x] / n * 100 for x in xs]
    ax.bar(xs, pct, color=case["color"], edgecolor="black", linewidth=0.5, zorder=3)
    mean = sum(sizes) / n
    ax.axvline(mean, ls="--", lw=1.0, color="#333333", zorder=2)
    ax.text(mean + 0.4, y_max * 0.92, f"среднее = {mean:.2f}",
            fontsize=9, color="#333333")
    ax.set_xlim(1.5, x_max + 0.5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("число кандидатов-смыслов на инстанс")
    ax.set_ylabel("доля инстансов, %")
    ax.set_title(f"{case['lang']}: $n={n}$, accuracy = {case['acc']:.2f}",
                 fontsize=10)
    ax.grid(axis="y", ls="-", lw=0.4, alpha=0.3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_sizes = [collect_sizes(c) for c in CASES]
    x_max = max(max(s) for s in all_sizes)
    y_max = max(max(collections.Counter(s).values()) / len(s) * 100
                for s in all_sizes) * 1.12

    # 3 отдельные картинки
    for sizes, case in zip(all_sizes, CASES):
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        draw_panel(ax, sizes, case, x_max, y_max)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"sense_dist_{case['code']}.png")
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print("saved", fname)

    # коллаж 1x3
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sizes, case in zip(axes, all_sizes, CASES):
        draw_panel(ax, sizes, case, x_max, y_max)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "sense_dist_collage.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    main()
