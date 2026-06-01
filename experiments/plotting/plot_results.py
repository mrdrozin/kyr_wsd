"""
Графики для отчёта ВКР.

  1. Кривые обучения — train/dev loss и dev accuracy по шагам. Заменяют таблицу
     лоссов (научный руководитель просил график вместо таблицы).
  2. Бар-чарт итоговых результатов на тесте с доверительными интервалами.

Запуск (после train.py / evaluate.py):
    python experiments/plot_results.py            # все графики из experiments/runs/
    python experiments/plot_results.py --loss-only
    python experiments/plot_results.py --results-only
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "experiments/runs"


def plot_loss_curve(run_dir):
    history = json.loads((run_dir / "loss_history.json").read_text(encoding="utf-8"))
    train, dev = history["train"], history["dev"]
    if not train and not dev:
        return None

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    if train:
        ax_loss.plot([p["step"] for p in train], [p["loss"] for p in train],
                     label="train loss", color="tab:blue", alpha=0.8)
    if dev:
        ax_loss.plot([p["step"] for p in dev], [p["loss"] for p in dev],
                     label="dev loss", color="tab:orange", marker="o")
    ax_loss.set_xlabel("шаг обучения")
    ax_loss.set_ylabel("loss")

    ax_acc = ax_loss.twinx()
    if dev:
        ax_acc.plot([p["step"] for p in dev], [p["accuracy"] for p in dev],
                    label="dev accuracy", color="tab:green", marker="s", linestyle="--")
    ax_acc.set_ylabel("dev accuracy")

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    ax_loss.legend(lines, [l.get_label() for l in lines], loc="best")
    ax_loss.set_title(f"Кривые обучения — {run_dir.name}")
    fig.tight_layout()
    out = run_dir / "loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def collect_by_config():
    """Сшиваем все evaluation*.json в runs/ в единый by_config-словарь.
    Это позволяет делать частичные оценки (несколько прогонов в разных
    файлах) и потом строить единый график."""
    files = sorted(RUNS.glob("evaluation*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"нет {RUNS}/evaluation*.json — сначала evaluate.py")
        return {}, []
    merged = {}
    sources = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! пропуск {path.name}: {e}", file=sys.stderr)
            continue
        by_cfg = data.get("by_config", {})
        for k, v in by_cfg.items():
            if k in merged and merged[k] != v:
                print(f"  ! конфиг {k!r} есть и в {merged.get('__src',{}).get(k,'?')} "
                      f"и в {path.name} — берём более поздний")
            merged[k] = v
        sources.append(path.name)
    if sources:
        print(f"склеено из: {', '.join(sources)}")
    return merged, sources


def plot_results_bar():
    by_config, sources = collect_by_config()
    if not by_config:
        return None

    names = sorted(by_config)
    means = [by_config[n]["mean"] for n in names]
    errs = [by_config[n]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 5))
    bars = ax.bar(names, means, yerr=errs, capsize=5, color="tab:blue", alpha=0.85)
    ax.set_ylabel("accuracy на тесте (Конституция)")
    ax.set_title("Сравнение моделей WSD (среднее ± std по seed'ам)")
    ax.set_ylim(0, 1)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.3f}",
                ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out = RUNS / "results_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()

    if not args.results_only:
        for run_dir in sorted(RUNS.glob("*_seed*")):
            if not (run_dir / "loss_history.json").exists():
                continue
            try:
                out = plot_loss_curve(run_dir)
                if out:
                    print(f"график: {out.relative_to(ROOT)}")
            except Exception as e:
                print(f"ОШИБКА на {run_dir.name}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    if not args.loss_only:
        try:
            out = plot_results_bar()
            if out:
                print(f"график: {out.relative_to(ROOT)}")
        except Exception as e:
            print(f"ОШИБКА в results_bar: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
