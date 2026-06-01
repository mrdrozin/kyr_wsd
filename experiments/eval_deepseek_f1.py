#!/usr/bin/env python3
"""Пересчёт DeepSeek WSD: accuracy + честный micro-F1 (с учётом отказов).

Точно повторяет фильтры из main_notebooks/wsd_results_evaluation.ipynb:
  - удаляются дубликаты предложений (актуально для RU);
  - пропускаются gold ∈ {'Нет верного варианта', 'Спорный случай'};
  - пропускаются инстансы с единственным определением (skip_one_def=True).

Дальше: предсказание = первый элемент senseIDs, если он есть; иначе None
(отказ). Считаются accuracy (= correct/total, None трактуется как неверный
ответ), micro-precision (= correct/attempted), micro-recall (= correct/total)
и micro-F1.
"""
import json

import pandas as pd


CASES = [
    {
        "name": "Русский",
        "gold_csv": "../gold_data/rus_gold.csv",
        "answers": "../deepseek_answers/deepseek_rus_results.json",
        "pred_key": "senseIDs",
        "gold_col": "selected_option",
        "split_gold": True,        # 'a;b' -> 'a'
        "use_glosses": False,
    },
    {
        "name": "Английский",
        "gold_csv": "../gold_data/eng_gold.csv",
        "answers": "../deepseek_answers/deepseek_eng_results.json",
        "pred_key": "sense_IDs",
        "gold_col": "selected_option",
        "split_gold": True,
        "use_glosses": False,
    },
    {
        "name": "Киргизский",
        "gold_csv": "../gold_data/kyr_gold.csv",
        "answers": "../deepseek_answers/deepseek_kyr_results.json",
        "pred_key": "senseIDs",
        "gold_col": "selected_option",
        "split_gold": False,
        "use_glosses": True,
    },
]


def evaluate(case):
    # пути в CASES даны как из main_notebooks/ (с '../'); приводим к корню проекта
    gold_df = pd.read_csv(case["gold_csv"].replace("../", "", 1))
    ans = pd.read_json(case["answers"].replace("../", "", 1))

    sentences = list(ans["sentence"])
    descriptions = list(ans["descriptions"])
    preds = list(ans[case["pred_key"]])
    gold = list(gold_df[case["gold_col"]])
    if case["split_gold"]:
        gold = [str(x).split(";")[0] for x in gold]

    seen_sentences = set()
    total = 0
    attempted = 0
    correct = 0
    for sent, descs, g, p in zip(sentences, descriptions, gold, preds):
        if sent in seen_sentences:
            continue
        seen_sentences.add(sent)
        if len(descs) == 1:
            continue
        if g in ("Нет верного варианта", "Спорный случай"):
            continue

        total += 1
        # предсказание: первый sense из списка или None (отказ)
        p_list = p if isinstance(p, list) else [p]
        if len(p_list) == 0:
            pred = None
        else:
            pred = p_list[0]

        if case["use_glosses"]:
            try:
                gold_val = descs[int(g)] if isinstance(g, (int, str)) and str(g).lstrip("-").isdigit() else g
            except Exception:
                gold_val = g
            pred_val = descs[int(pred)] if (pred is not None) else None
        else:
            gold_val = g
            pred_val = pred

        if pred_val is None:
            continue  # отказ: attempted и correct не растут
        attempted += 1
        if pred_val == gold_val:
            correct += 1

    accuracy = correct / total if total else 0.0
    precision = correct / attempted if attempted else 0.0
    recall = correct / total if total else 0.0  # = accuracy
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    abstentions = total - attempted

    print(f"== {case['name']} ==")
    print(f"  total (после фильтров): {total}")
    print(f"  attempted (DeepSeek дал ответ): {attempted}")
    print(f"  abstentions (отказался): {abstentions}")
    print(f"  correct: {correct}")
    print(f"  accuracy (= correct/total):     {accuracy:.4f}")
    print(f"  micro-precision (correct/att.): {precision:.4f}")
    print(f"  micro-recall    (correct/total):{recall:.4f}")
    print(f"  micro-F1:                       {f1:.4f}")
    print()
    return dict(name=case["name"], total=total, attempted=attempted,
                correct=correct, accuracy=accuracy, precision=precision,
                recall=recall, f1=f1)


def main():
    rows = [evaluate(c) for c in CASES]
    print("# Сводка")
    print(f"{'Язык':12s} | total | attempted | acc    | P      | R      | F1")
    for r in rows:
        print(f"{r['name']:12s} | {r['total']:5d} | {r['attempted']:9d} | "
              f"{r['accuracy']:.4f} | {r['precision']:.4f} | "
              f"{r['recall']:.4f} | {r['f1']:.4f}")


if __name__ == "__main__":
    main()
