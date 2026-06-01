#!/usr/bin/env python3
"""
collect_missing_glosses.py
==========================

Собирает список английских определений (glosses) из WordNet, которые
понадобятся для пересборки обучающего корпуса с настоящими WSD-дистракторами
(все альтернативные значения той же леммы), но ещё не переведены на
киргизский в glosses_dct.json.

Логика
------
1. Загружает уже переведённые определения из glosses_dct.json.
2. Идёт по SemCor.jsonl с теми же фильтрами, что используются в
   prepare_train.ipynb: только существительные, лемма не входит в
   валидационный набор, sense_key однозначен, на одну лемму не более
   --lemma-limit вхождений.
3. Для каждого прошедшего фильтр instance:
      • определяет правильный synset по sense_key,
      • получает все synsets этой леммы из WordNet,
      • берёт у каждого synset его definition() — это и есть тот набор
        кандидатов, который должна различать модель.
4. Сравнивает множество требуемых definitions с уже переведёнными,
   печатает статистику и сохраняет недостающие в JSON-список,
   готовый к подаче в DeepSeek для пакетного перевода.

Использование
-------------
    python collect_missing_glosses.py \\
        --semcor      /path/to/SemCor.jsonl \\
        --glosses-dct auxiliary_data/glosses_dct.json \\
        --eng-gold    gold_data/eng_gold.csv \\
        --output      missing_glosses.json \\
        --lemma-limit 8

Опции
-----
    --lemma-limit N     максимум instance на лемму (по умолчанию 8,
                        соответствует текущему prepare_train.ipynb)
    --keep-monosemous   не отфильтровывать леммы с одним значением
                        (по умолчанию они исключаются как непригодные
                        для WSD)
    --dry-run           не писать выходной файл, только напечатать сводку
    --verbose-sample N  сколько примеров недостающих glosses показать в конце

Зависимости
-----------
    pip install pandas nltk
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# Чтение SemCor и валидации
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Лениво читает JSONL: по одному словарю на строку."""
    with open(path, 'r', encoding='utf-8') as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def build_test_nouns(eng_gold_path: str | Path) -> set[str]:
    """Возвращает множество лемм-существительных из валидационного eng_gold.csv.

    Эти леммы должны быть исключены из обучения, чтобы избежать утечки
    из валидации в train.
    """
    df = pd.read_csv(eng_gold_path)
    lemmatizer = WordNetLemmatizer()
    return {lemmatizer.lemmatize(word, 'n') for word in df['word']}


# ---------------------------------------------------------------------------
# Получение всех synsets целевой леммы
# ---------------------------------------------------------------------------

def collect_target_synsets(sense_key: str, lemma_hint: str) -> tuple[object | None, list]:
    """По sense_key и подсказке-лемме возвращает (правильный synset,
    список всех synsets-кандидатов).

    Логика выбора канонической леммы:
      • из правильного synset берутся имена лемм существительных;
      • если lemma_hint среди них — используется он;
      • иначе берётся первое имя из synset.

    Если правильный synset вдруг не попал в общий список (бывает для
    многословных лемм или несовпадения формы), он добавляется явно,
    чтобы не потерять правильную метку.
    """
    try:
        true_synset = wn.lemma_from_key(sense_key).synset()
    except Exception:
        return None, []

    if true_synset.pos() != 'n':
        return None, []

    lemma_names = [lm.name() for lm in true_synset.lemmas()]
    canonical = lemma_hint if lemma_hint in lemma_names else (
        lemma_names[0] if lemma_names else lemma_hint
    )

    all_synsets = wn.synsets(canonical, pos='n')

    if true_synset not in all_synsets:
        all_synsets = list(all_synsets) + [true_synset]

    return true_synset, all_synsets


# ---------------------------------------------------------------------------
# Основной обход
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--semcor', required=True,
                        help='Путь к SemCor.jsonl')
    parser.add_argument('--glosses-dct', default='auxiliary_data/glosses_dct.json',
                        help='JSON-словарь уже переведённых definitions (en→ky)')
    parser.add_argument('--eng-gold', default='gold_data/eng_gold.csv',
                        help='CSV с английским валидационным набором')
    parser.add_argument('--output', default='missing_glosses.json',
                        help='Куда сохранить список недостающих definitions')
    parser.add_argument('--lemma-limit', type=int, default=8,
                        help='Максимум instance на лемму (как в prepare_train.ipynb)')
    parser.add_argument('--keep-monosemous', action='store_true',
                        help='Не отфильтровывать однозначные леммы')
    parser.add_argument('--dry-run', action='store_true',
                        help='Не сохранять выходной файл')
    parser.add_argument('--verbose-sample', type=int, default=5,
                        help='Сколько примеров недостающих glosses показать')
    args = parser.parse_args()

    # ----- 1. Загружаем уже переведённые definitions ----------------------

    with open(args.glosses_dct, 'r', encoding='utf-8') as fh:
        glosses_dct = json.load(fh)
    translated = set(glosses_dct.keys())
    print(f'[1/4] Загружено уже переведённых glosses: {len(translated):,}')

    # ----- 2. Леммы валидации, которые исключаем из train -----------------

    test_nouns = build_test_nouns(args.eng_gold)
    print(f'[2/4] Лемм из валидации (исключаются из обучения): {len(test_nouns):,}')

    # ----- 3. Обход SemCor.jsonl с теми же фильтрами ----------------------

    print(f'[3/4] Читаем SemCor: {args.semcor}')

    cnt: Counter[str] = Counter()
    needed_glosses: set[str] = set()
    glosses_per_lemma: dict[str, set[str]] = defaultdict(set)
    instances_total = 0
    instances_passed = 0
    skipped = Counter()

    for entry in read_jsonl(args.semcor):
        instances_total += 1
        lemma = entry.get('lemma')
        pos = entry.get('pos')
        sense = entry.get('sense', '')

        # Те же фильтры, что в prepare_train.ipynb
        if pos != 'NOUN':
            skipped['not_noun'] += 1
            continue
        if lemma in test_nouns:
            skipped['in_validation'] += 1
            continue
        if len(sense.split(';')) != 1:
            skipped['ambiguous_sense'] += 1
            continue
        if cnt[lemma] >= args.lemma_limit:
            skipped['lemma_limit'] += 1
            continue

        true_synset, all_synsets = collect_target_synsets(sense, lemma)
        if true_synset is None:
            skipped['wordnet_lookup_failed'] += 1
            continue
        if not args.keep_monosemous and len(all_synsets) <= 1:
            skipped['monosemous'] += 1
            continue

        # Собираем все английские definitions, которые нам понадобятся
        for synset in all_synsets:
            gloss_en = synset.definition()
            needed_glosses.add(gloss_en)
            glosses_per_lemma[lemma].add(gloss_en)

        cnt[lemma] += 1
        instances_passed += 1

    print(f'   Просмотрено строк SemCor : {instances_total:,}')
    print(f'   Прошло фильтры instance  : {instances_passed:,}')
    print(f'   Уникальных лемм          : {len(cnt):,}')
    if skipped:
        print('   Отсев по причинам:')
        for reason, n in skipped.most_common():
            print(f'      {reason:25s}: {n:,}')

    # ----- 4. Сравнение и сохранение --------------------------------------

    print()
    print('=' * 72)
    print('СВОДКА')
    print('=' * 72)

    already_have = needed_glosses & translated
    missing = needed_glosses - translated

    print(f'  Всего требуемых glosses (правильные + дистракторы): {len(needed_glosses):>8,}')
    print(f'  Уже переведены                                    : {len(already_have):>8,}')
    print(f'  Надо доперевести                                  : {len(missing):>8,}')
    if needed_glosses:
        coverage = 100.0 * len(already_have) / len(needed_glosses)
        print(f'  Покрытие имеющимися переводами                    : {coverage:>7.1f}%')

    # Распределение synsets-кандидатов на лемму
    synset_counts = [len(s) for s in glosses_per_lemma.values()]
    if synset_counts:
        print()
        print('  Распределение числа значений у лемм:')
        print(f'    среднее: {statistics.mean(synset_counts):.2f}')
        print(f'    медиана: {statistics.median(synset_counts):.0f}')
        print(f'    минимум: {min(synset_counts)}')
        print(f'    максимум: {max(synset_counts)}')
        dist = Counter(min(c, 10) for c in synset_counts)
        print('    распределение (1…10+):')
        for k in range(1, 11):
            label = f'{k}+' if k == 10 else str(k)
            bar = '█' * (dist[k] * 40 // max(dist.values()))
            print(f'      {label:>3s}: {dist[k]:>5,}  {bar}')

    # Сохранение
    print()
    if args.dry_run:
        print(f'  [--dry-run]: файл {args.output} не записан')
    else:
        sorted_missing = sorted(missing)
        with open(args.output, 'w', encoding='utf-8') as fh:
            json.dump(sorted_missing, fh, ensure_ascii=False, indent=2)
        print(f'  Список недостающих glosses сохранён в {args.output}')

    # Примеры для визуального контроля
    if missing and args.verbose_sample > 0:
        print()
        print(f'  Примеры недостающих glosses (первые {args.verbose_sample}):')
        for gloss in sorted(missing)[:args.verbose_sample]:
            print(f'    • {gloss}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nПрервано пользователем', file=sys.stderr)
        sys.exit(130)