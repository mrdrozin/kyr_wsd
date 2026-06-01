"""
Пересборка обучающего датасета киргизского WSD.

Проблема старого train_set.json: дистракторы (ложные глоссы) подбирались случайно,
поэтому кандидаты инстанса не были реальными конкурирующими значениями целевого
слова — задача получалась нереалистично лёгкой.

Здесь кандидаты инстанса = ВСЕ значения (синсеты) целевого существительного в
WordNet, переведённые на киргизский (auxiliary_data/all_glosses_dct.json).
Верный синсет помечается label=1, остальные — label=0.

Этапы:
  1. Воспроизводится фильтр SemCor из auxiliary_notebooks/get_kyr_contexts.ipynb
     (NOUN, лемма не из тестовых, ровно один gold-sense, не более 8 на лемму) —
     получается список записей, выровненный по индексу с переведёнными
     контекстами auxiliary_data/kyr_train_sentences.json.
  2. Для каждой записи перечисляются все синсеты-существительные леммы; их
     английские определения переводятся через all_glosses_dct.json.
  3. Финальный отсев: инстансы, у которых после сборки осталось <= 1 кандидата
     (истинные моносеманты и слова, ужатые непереведёнными значениями до одного
     варианта), исключаются — разрешать в них нечего.

Запуск из корня репозитория:  python3 experiments/build_train_set.py
Вывод: experiments/data/train_rebuilt.json  (строки instance_id/context/gloss/label)
"""

import json
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parent.parent
SEMCOR = ROOT / "wsd-data/data/jsonl/SemCor.jsonl"
ENG_GOLD = ROOT / "gold_data/eng_gold.csv"
CONTEXTS = ROOT / "auxiliary_data/kyr_train_sentences.json"
GLOSSES = ROOT / "auxiliary_data/all_glosses_dct.json"
OUT = ROOT / "experiments/data/train_rebuilt.json"
REPORT = ROOT / "experiments/data/build_train_report.json"

MAX_PER_LEMMA = 8


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def lemmatize_noun(word):
    """Лемматизация существительного через WordNet (как в get_kyr_contexts.ipynb)."""
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer().lemmatize(word, "n")


def filter_semcor(test_nouns):
    """Воспроизводит фильтр обучающих записей из get_kyr_contexts.ipynb."""
    cnt = Counter()
    records = []
    for line in read_jsonl(SEMCOR):
        lemma = line["lemma"]
        if (cnt[lemma] < MAX_PER_LEMMA
                and line["pos"] == "NOUN"
                and lemma not in test_nouns
                and len(line["sense"].split(";")) == 1):
            records.append(line)
            cnt[lemma] += 1
    return records


def main():
    # --- тестовые существительные (исключаются из обучения, как в исходном пайплайне)
    import csv
    with open(ENG_GOLD, encoding="utf-8") as f:
        test_nouns = {lemmatize_noun(row["word"]) for row in csv.DictReader(f)}

    records = filter_semcor(test_nouns)
    contexts = json.loads(CONTEXTS.read_text(encoding="utf-8"))
    glosses_dct = json.loads(GLOSSES.read_text(encoding="utf-8"))

    if len(records) != len(contexts):
        raise SystemExit(
            f"Рассинхрон: записей SemCor {len(records)}, контекстов {len(contexts)}. "
            "Фильтр SemCor должен совпадать с get_kyr_contexts.ipynb."
        )

    rows = []
    stats = Counter()
    glosses_per_instance = []
    seen_ids = set()

    for rec, context in zip(records, contexts):
        stats["records_total"] += 1

        # --- контекст: пустой или без двух маркеров [TGT] — непригоден
        if not context or context.count("[TGT]") != 2:
            stats["dropped_bad_context"] += 1
            continue

        # --- верный синсет и каноничная лемма из sense-key
        try:
            correct_lemma = wn.lemma_from_key(rec["sense"])
            correct_synset = correct_lemma.synset()
        except Exception:
            stats["dropped_bad_sense_key"] += 1
            continue

        correct_def = correct_synset.definition()
        if correct_def not in glosses_dct:
            stats["dropped_untranslated_correct"] += 1
            continue

        # --- все значения целевого существительного
        candidates = wn.synsets(correct_lemma.name(), pos=wn.NOUN)
        if correct_synset not in candidates:
            candidates = [correct_synset] + candidates

        inst_rows = []
        seen_syn = set()
        for syn in candidates:
            if syn.name() in seen_syn:
                continue
            seen_syn.add(syn.name())
            definition = syn.definition()
            if definition not in glosses_dct:
                stats["distractors_untranslated"] += 1
                continue
            inst_rows.append({
                "instance_id": rec["id"],
                "context": context,
                "gloss": glosses_dct[definition],
                "label": 1 if syn == correct_synset else 0,
            })

        # --- финальный отсев: нужно >= 2 кандидатов и ровно один верный
        if len(inst_rows) <= 1:
            stats["dropped_monosemous"] += 1
            continue
        if sum(r["label"] for r in inst_rows) != 1:
            stats["dropped_label_anomaly"] += 1
            continue

        if rec["id"] in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(rec["id"])

        rows.extend(inst_rows)
        glosses_per_instance.append(len(inst_rows))
        stats["instances_kept"] += 1

    n = len(glosses_per_instance)
    report = {
        "records_total": stats["records_total"],
        "instances_kept": stats["instances_kept"],
        "rows_total": len(rows),
        "dropped": {
            "bad_context": stats["dropped_bad_context"],
            "bad_sense_key": stats["dropped_bad_sense_key"],
            "untranslated_correct_gloss": stats["dropped_untranslated_correct"],
            "monosemous_or_single_candidate": stats["dropped_monosemous"],
            "label_anomaly": stats["dropped_label_anomaly"],
            "duplicate_id": stats["dropped_duplicate_id"],
        },
        "distractors_skipped_untranslated": stats["distractors_untranslated"],
        "glosses_per_instance": {
            "min": min(glosses_per_instance) if n else 0,
            "mean": round(sum(glosses_per_instance) / n, 3) if n else 0,
            "max": max(glosses_per_instance) if n else 0,
        },
    }

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nЗаписано: {OUT.relative_to(ROOT)}")
    print(f"Отчёт:    {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
