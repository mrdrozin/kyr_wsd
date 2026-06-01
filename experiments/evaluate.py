"""
Оценка обученных WSD-моделей на тесте (корпус Конституции).

Для каждого прогона считается accuracy и bootstrap 95% доверительный интервал
(ресэмплинг тестовых инстансов). Для нескольких seed'ов одного конфига —
агрегат: среднее, стандартное отклонение, разброс. Это закрывает претензии
«нет доверительных интервалов» и «сколько раз прогнали эксперимент».

Основная метрика — accuracy (доля верно выбранных смыслов). Macro-P/R/F
намеренно не считаются: при классах-синглтонах они малоинформативны
(см. thesis_review/REVIEW.md, пункт С2).

Запуск:
    python experiments/evaluate.py experiments/runs/KyrgyzBert_full_seed*
"""

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import GlossSelectionModel, GroupedWSDDataset, apply_lora, make_collate_fn, run_eval

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments/data"
RUNS = ROOT / "experiments/runs"


def bootstrap_ci(correct, n_boot=2000, seed=0):
    """95% доверительный интервал accuracy через bootstrap по инстансам."""
    if not correct:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(correct, dtype=float)
    n = len(arr)
    accs = [arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_run(run_dir, test_rows, device):
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    model = GlossSelectionModel(cfg["model_name"], dropout=cfg["dropout"])
    if cfg["lora"]:
        lp = cfg["lora_params"]
        # target_modules ОБЯЗАТЕЛЬНО брать из run_config — иначе структура LoRA
        # не совпадёт с чекпойнтом (у KyrgyzBert адаптер шире: query/key/value/dense)
        model = apply_lora(model, r=lp["r"], alpha=lp["alpha"], dropout=lp["dropout"],
                           target_modules=lp.get("target_modules", ["query", "value"]))
    model.load_state_dict(
        torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    collate = make_collate_fn(tokenizer, max_length=cfg["max_length"])
    # batch 8 — тест маленький (438 инстансов), а у XLM-R при больших группах
    # на T4 на 16 бывает OOM
    # тест маленький (438 инстансов) — воркеры не нужны и создают лишний риск
    # краша спавна; токенизация в основном потоке здесь занимает доли секунды
    loader = DataLoader(GroupedWSDDataset(test_rows), batch_size=8,
                        collate_fn=collate, num_workers=0)

    res = run_eval(model, loader, device)
    lo, hi = bootstrap_ci(res["correct"])
    result = {
        "run": run_dir.name,
        "accuracy": round(res["accuracy"], 4),
        "ci95": [lo, hi],
        "evaluated_instances": len(res["correct"]),
        "skipped_no_gold": res["skipped"],
    }
    # освободить VRAM перед следующим прогоном
    del model
    return result


def config_name(run_name):
    """KyrgyzBert_full_seed42 -> KyrgyzBert_full"""
    return run_name.rsplit("_seed", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="каталоги прогонов в experiments/runs/")
    parser.add_argument("--output", default=str(RUNS / "evaluation.json"),
                        help="куда писать результаты (default: runs/evaluation.json). "
                             "Для частичных прогонов используйте отдельные имена, "
                             "например runs/evaluation_extended.json — plot_results.py "
                             "склеит все файлы evaluation*.json автоматически.")
    args = parser.parse_args()

    test_rows = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
    device = pick_device()

    per_run, failed = [], []
    for path in args.runs:
        run_dir = Path(path)
        if not (run_dir / "model.pt").exists():
            print(f"пропуск {run_dir}: нет model.pt", file=sys.stderr)
            continue
        try:
            result = evaluate_run(run_dir, test_rows, device)
            per_run.append(result)
            print(f"{result['run']}: acc={result['accuracy']} CI95={result['ci95']}")
        except Exception as e:
            msg = f"ОШИБКА в {run_dir.name}: {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            failed.append({"run": run_dir.name, "error": str(e)})
        finally:
            # очистка не должна ронять скрипт, даже если CUDA-контекст повреждён
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  (cleanup warning: {e})", file=sys.stderr)

    # --- агрегат по seed'ам одного конфига
    by_config = defaultdict(list)
    for r in per_run:
        by_config[config_name(r["run"])].append(r["accuracy"])

    summary = {}
    for cfg_name, accs in by_config.items():
        a = np.array(accs)
        summary[cfg_name] = {
            "seeds": len(accs),
            "accuracies": [round(x, 4) for x in accs],
            "mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "min": round(float(a.min()), 4),
            "max": round(float(a.max()), 4),
        }

    out = {"per_run": per_run, "by_config": summary, "failed": failed}
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = RUNS / output_path.name if output_path.parent == Path('.') else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Агрегат по конфигам ===")
    for cfg_name, s in summary.items():
        print(f"{cfg_name}: mean={s['mean']} std={s['std']} "
              f"(seeds={s['seeds']}, min={s['min']}, max={s['max']})")
    rel = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    print(f"\nЗаписано: {rel}")


if __name__ == "__main__":
    main()
