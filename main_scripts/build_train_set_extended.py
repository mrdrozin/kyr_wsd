"""
Расширенная сборка обучающей выборки — абляция к build_train_set.py.

Отличается тем, что:
  • инстансы, ставшие моносемантными после фильтрации (всего 1 переведённый
    синсет или 1 синсет в WordNet), НЕ выкидываются: к ним добавляются от 3
    до 10 случайных киргизских глоссов как дистракторы;
  • полисемантичные инстансы ограничиваются сверху по числу кандидатов:
    максимум 11 = 1 верный + 10 неверных. Если в WordNet+переводах кандидатов
    больше, лишние неверные выбрасываются случайно (с фиксированным seed).

Зачем: проверить, помогает ли расширение train на ~12.5k инстансов с заведомо
случайными дистракторами, ценой частичного отката фикса К2 (см. REVIEW.md).
Это аблационный эксперимент, а не дефолт.

Запуск из корня репозитория:  python3 experiments/build_train_set_extended.py
Вывод:
  experiments/data/train_rebuilt_extended.json
  experiments/data/build_train_extended_report.json
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parent.parent
SEMCOR = ROOT / "wsd-data/data/jsonl/SemCor.jsonl"
ENG_GOLD = ROOT / "gold_data/eng_gold.csv"
CONTEXTS = ROOT / "auxiliary_data/kyr_train_sentences.json"
GLOSSES = ROOT / "auxiliary_data/all_glosses_dct.json"
OUT = ROOT / "experiments/data/train_rebuilt_extended.json"
REPORT = ROOT / "experiments/data/build_train_extended_report.json"

MAX_PER_LEMMA = 8
MAX_CANDIDATES = 11           # 1 верный + до 10 неверных
MIN_RANDOM_DISTRACTORS = 3
MAX_RANDOM_DISTRACTORS = 10
SEED = 42


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def lemmatize_noun(word):
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer().lemmatize(word, "n")


def filter_semcor(test_nouns):
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


def dedup_synsets(synsets):
    seen, out = set(), []
    for s in synsets:
        if s.name() not in seen:
            seen.add(s.name())
            out.append(s)
    return out


def main():
    with open(ENG_GOLD, encoding="utf-8") as f:
        test_nouns = {lemmatize_noun(row["word"]) for row in csv.DictReader(f)}

    records = filter_semcor(test_nouns)
    contexts = json.loads(CONTEXTS.read_text(encoding="utf-8"))
    glosses_dct = json.loads(GLOSSES.read_text(encoding="utf-8"))

    if len(records) != len(contexts):
        raise SystemExit(
            f"Рассинхрон: записей SemCor {len(records)}, контекстов {len(contexts)}")

    rng = random.Random(SEED)
    # пул киргизских глоссов для случайных дистракторов (только в моносемант-ветке)
    kyr_pool = list(set(glosses_dct.values()))

    rows = []
    stats = Counter()
    glosses_per_instance = []
    seen_ids = set()

    for rec, context in zip(records, contexts):
        stats["records_total"] += 1

        if not context or context.count("[TGT]") != 2:
            stats["dropped_bad_context"] += 1
            continue

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

        all_synsets = wn.synsets(correct_lemma.name(), pos=wn.NOUN)
        if correct_synset not in all_synsets:
            all_synsets = [correct_synset] + all_synsets
        all_synsets = dedup_synsets(all_synsets)

        # собираем переведённых кандидатов
        candidates = []  # list of (gloss_kyr, label)
        for syn in all_synsets:
            d = syn.definition()
            if d not in glosses_dct:
                stats["distractors_untranslated"] += 1
                continue
            candidates.append((glosses_dct[d], 1 if syn == correct_synset else 0))

        if sum(label for _, label in candidates) != 1:
            stats["dropped_label_anomaly"] += 1
            continue

        if rec["id"] in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(rec["id"])

        # ветка по числу реальных кандидатов
        n_real = len(candidates)
        if n_real <= 1:
            # моносемантный после фильтрации -> добавляем случайные дистракторы
            correct_gloss = candidates[0][0]
            n_rand = rng.randint(MIN_RANDOM_DISTRACTORS, MAX_RANDOM_DISTRACTORS)
            pool = [g for g in kyr_pool if g != correct_gloss]
            random_distractors = rng.sample(pool, min(n_rand, len(pool)))
            final = [(correct_gloss, 1)] + [(g, 0) for g in random_distractors]
            # тип моносеманта: true (1 синсет в WordNet) vs pseudo (синсетов >1, но
            # переведён только один)
            if len(all_synsets) == 1:
                stats["filled_true_monosemous"] += 1
            else:
                stats["filled_pseudo_monosemous"] += 1
        else:
            # полисемантный — ограничиваем сверху до MAX_CANDIDATES
            if n_real > MAX_CANDIDATES:
                correct = [c for c in candidates if c[1] == 1]
                wrongs = [c for c in candidates if c[1] == 0]
                kept_wrongs = rng.sample(wrongs, MAX_CANDIDATES - 1)
                final = correct + kept_wrongs
                stats["polysemous_capped"] += 1
            else:
                final = candidates
                stats["polysemous_kept"] += 1

        for gloss, label in final:
            rows.append({
                "instance_id": rec["id"], "context": context,
                "gloss": gloss, "label": label,
            })
        glosses_per_instance.append(len(final))
        stats["instances_kept"] += 1

    n = len(glosses_per_instance)
    bucket = Counter(glosses_per_instance)
    report = {
        "config": {
            "seed": SEED,
            "max_candidates": MAX_CANDIDATES,
            "random_distractors_range": [MIN_RANDOM_DISTRACTORS, MAX_RANDOM_DISTRACTORS],
        },
        "records_total": stats["records_total"],
        "instances_kept": stats["instances_kept"],
        "rows_total": len(rows),
        "by_branch": {
            "polysemous_kept": stats["polysemous_kept"],
            "polysemous_capped": stats["polysemous_capped"],
            "filled_true_monosemous": stats["filled_true_monosemous"],
            "filled_pseudo_monosemous": stats["filled_pseudo_monosemous"],
        },
        "dropped": {
            "bad_context": stats["dropped_bad_context"],
            "bad_sense_key": stats["dropped_bad_sense_key"],
            "untranslated_correct_gloss": stats["dropped_untranslated_correct"],
            "label_anomaly": stats["dropped_label_anomaly"],
            "duplicate_id": stats["dropped_duplicate_id"],
        },
        "distractors_skipped_untranslated": stats["distractors_untranslated"],
        "glosses_per_instance": {
            "min": min(glosses_per_instance) if n else 0,
            "mean": round(sum(glosses_per_instance) / n, 3) if n else 0,
            "max": max(glosses_per_instance) if n else 0,
            "histogram": dict(sorted(bucket.items())),
        },
    }

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nЗаписано: {OUT.relative_to(ROOT)}")
    print(f"Отчёт:    {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
