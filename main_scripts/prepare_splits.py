"""
Формирование честных выборок для киргизского WSD.

Зачем: ранее число эпох подбиралось на корпусе Конституции, на нём же мерялся
итоговый результат — фактически отладка на тестовой выборке. Здесь dev
(для подбора гиперпараметров) выделяется из обучающих данных и НЕ пересекается
с тестом. Конституция используется только как финальный тест — один раз.

  train.json / dev.json — instance-level сплит пересобранного train (90/10).
                          Деление по instance_id: все строки одного инстанса
                          целиком уходят в одну часть (иначе утечка инстанса).
  test.json             — дословная копия корпуса Конституции (validation_set.json),
                          размеченного вручную. Не модифицируется.

Запуск из корня репозитория:  python3 experiments/prepare_splits.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REBUILT = ROOT / "experiments/data/train_rebuilt.json"
CONSTITUTION = ROOT / "kyr_wsd_dataset/validation_set.json"
DATA = ROOT / "experiments/data"

DEV_FRACTION = 0.10
SEED = 42


def group_by_instance(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["instance_id"]].append(r)
    return groups


def gloss_stats(groups):
    sizes = [len(v) for v in groups.values()]
    n = len(sizes)
    return {
        "instances": n,
        "rows": sum(sizes),
        "glosses_per_instance": {
            "min": min(sizes) if n else 0,
            "mean": round(sum(sizes) / n, 3) if n else 0,
            "max": max(sizes) if n else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_REBUILT),
                        help="путь к rebuilt-файлу для сплита "
                             "(default: experiments/data/train_rebuilt.json)")
    parser.add_argument("--prefix", default="",
                        help="суффикс для имён сплитов: train_PREFIX.json / dev_PREFIX.json "
                             "(default: без суффикса — train.json / dev.json)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute() and not source.exists():
        source = ROOT / source   # пробуем относительно корня репозитория
    rebuilt = json.loads(source.read_text(encoding="utf-8"))
    groups = group_by_instance(rebuilt)

    instance_ids = sorted(groups)              # детерминированный порядок до shuffle
    random.Random(SEED).shuffle(instance_ids)

    n_dev = round(len(instance_ids) * DEV_FRACTION)
    dev_ids = set(instance_ids[:n_dev])
    train_ids = set(instance_ids[n_dev:])
    assert not (dev_ids & train_ids)

    train_rows = [r for iid in train_ids for r in groups[iid]]
    dev_rows = [r for iid in dev_ids for r in groups[iid]]

    # тест — Конституция, не зависит от источника train; пишем только если ещё нет
    test_path = DATA / "test.json"
    if test_path.exists():
        test_rows = json.loads(test_path.read_text(encoding="utf-8"))
    else:
        test_rows = json.loads(CONSTITUTION.read_text(encoding="utf-8"))
        test_path.write_text(
            json.dumps(test_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    suffix = f"_{args.prefix}" if args.prefix else ""
    (DATA / f"train{suffix}.json").write_text(
        json.dumps(train_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / f"dev{suffix}.json").write_text(
        json.dumps(dev_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    report = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "prefix": args.prefix or None,
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "split_unit": "instance_id",
        "train": gloss_stats(group_by_instance(train_rows)),
        "dev": gloss_stats(group_by_instance(dev_rows)),
        "test": gloss_stats(group_by_instance(test_rows)),
        "test_source": "kyr_wsd_dataset/validation_set.json (корпус Конституции КР, ручная разметка)",
    }
    (DATA / f"splits{suffix}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Записано: experiments/data/train{suffix}.json")
    print(f"Записано: experiments/data/dev{suffix}.json")
    print(f"Тест:     experiments/data/test.json")


if __name__ == "__main__":
    main()
