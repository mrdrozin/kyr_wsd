"""
Сборка валидационного (тестового) набора данных для WSD-модели (киргизский язык).

Алгоритм:
  1. Читает результаты Apertium-разбора Конституции (киргизские предложения).
  2. Читает аннотацию английских смыслов (eng_gold.csv) и сопоставление
     eng↔kyr смыслов (eng-kyr-matched_gold.csv).
  3. Для каждого запроса (deepseek_eng_results.json) извлекает леммy
     целевого слова через streamparser и строит контекст с маркерами [TGT].
  4. Берёт список киргизских толкований из словаря (glossary.csv) и
     правильный индекс из таблицы сопоставления.
  5. Пропускает случаи с единственным толкованием или без правильного ответа.
  6. Записывает итоговый список словарей в validation_set.json.
"""

import json
from collections import defaultdict
from math import isnan
from pathlib import Path

import pandas as pd
from streamparser import LexicalUnit

# ---------------------------------------------------------------------------
# Пути (относительно директории скрипта)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

KYR_SENTENCES_JSON     = BASE_DIR / "../auxiliary_data/apertium_constitution.json"
ENG_QUERIES_JSON       = BASE_DIR / "../deepseek_answers/deepseek_eng_results.json"
ENG_GOLD_CSV           = BASE_DIR / "../gold_data/eng_gold.csv"
MATCHED_SENSES_CSV     = BASE_DIR / "../gold_data/eng-kyr-matched_gold.csv"
GLOSSARY_CSV           = BASE_DIR / "../auxiliary_data/glossary.csv"
OUTPUT_FILE            = BASE_DIR / "validation_set.json"

# Количество возможных столбцов с толкованиями в glossary.csv
MAX_SENSES = 11

# Немногочисленные случаи, когда Apertium предлагает несколько лемм;
# правильный вариант выбран вручную.
HARD_LEMMAS = {
    frozenset(("ата",   "атан")):  "ата",
    frozenset(("бал",   "бала")):  "бала",
    frozenset(("жара",  "жаран")): "жаран",
    frozenset(("күн",   "күнү")):  "күн",
    frozenset(("уй",    "уюм")):   "уюм",
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_lex_units(units: list) -> list:
    """Преобразует список строк streamparser в объекты LexicalUnit."""
    return [LexicalUnit(unit) for unit in units]


def build_senses_dct(matched_senses: pd.DataFrame) -> dict:
    """
    Строит словарь {(eng_sense, kyr_lemma): индекс_kyр_толкования}.
    Нумерация в CSV начинается с 1; -1 означает «правильного ответа нет».
    """
    dct = defaultdict()
    for _, row in matched_senses.iterrows():
        selected = int(row.selected_option.split(".")[0])
        # переводим в 0-based; -1 остаётся -1 (нет правильного ответа)
        dct[(row.eng_sense, row.kyr_lemma)] = selected - 1 if selected > 0 else selected
    return dct


def build_glossary_dict(glossary_csv: pd.DataFrame) -> dict:
    """
    Читает CSV-словарь и строит {существительное: [список толкований]}.
    Первая строка CSV пропускается (заголовочная строка с индексом).
    """
    dct = defaultdict()
    for _, row in glossary_csv.iloc[1:].iterrows():
        senses = []
        for i in range(1, MAX_SENSES + 1):
            val = row.get(f"sense{i}")
            if isinstance(val, float) and isnan(val):
                break
            senses.append(val)
        dct[row.noun] = senses
    return dct


def get_sentence(sentence: list, kyr_token_id: int) -> tuple:
    """
    По токену целевого слова извлекает лемму и строит строку контекста
    с маркерами [TGT] вокруг целевого слова.

    Возвращает (lemma, context_string).
    """
    target_unit = sentence[kyr_token_id]
    nouns = set()

    # перебираем все варианты разбора и собираем существительные-леммы
    for reading in target_unit.readings:
        for sread in reading:
            if "n" in sread.tags:
                nouns.add(sread.baseform)

    if len(nouns) == 1:
        noun = nouns.pop()
    else:
        # неоднозначный случай — берём лемму из справочника
        noun = HARD_LEMMAS[frozenset(nouns)]

    # вставляем маркеры вокруг целевого слова
    words = [unit.wordform for unit in sentence]
    words.insert(kyr_token_id + 1, "[TGT]")
    words.insert(kyr_token_id, "[TGT]")

    return noun, " ".join(words)


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def build_validation_set(
    eng_queries: list,
    senses_gold: list,
    kyr_units: list,
    glossary_dict: dict,
    senses_dct: dict,
) -> list:
    """
    Для каждого запроса формирует набор словарей с контекстом и толкованиями.
    Пропускает примеры с единственным толкованием или без правильного ответа.
    """
    validation_set = []

    for query, eng_sense in zip(eng_queries, senses_gold):
        sentence_id   = query["sentence_id"]
        kyr_token_id  = query["kyr_token_id"]
        sentence      = kyr_units[sentence_id]

        lemma, tgt_sentence = get_sentence(sentence, kyr_token_id)

        kyr_senses = glossary_dict[lemma]
        sense_idx  = senses_dct[(eng_sense, lemma)]

        # пропускаем: одно толкование или нет правильного
        if len(kyr_senses) == 1 or sense_idx == -1:
            continue

        for ind, kyr_sense in enumerate(kyr_senses):
            validation_set.append({
                "instance_id": f"{sentence_id}_{kyr_token_id}",
                "context":     tgt_sentence,
                "gloss":       kyr_sense,
                "label":       int(ind == sense_idx),
            })

    return validation_set


def main():
    # Загрузка данных
    with open(KYR_SENTENCES_JSON, encoding="utf-8") as f:
        raw_sentences = json.load(f)

    with open(ENG_QUERIES_JSON, encoding="utf-8") as f:
        eng_queries = json.load(f)

    eng_gold      = pd.read_csv(ENG_GOLD_CSV)
    matched_senses = pd.read_csv(MATCHED_SENSES_CSV)
    glossary_csv  = pd.read_csv(GLOSSARY_CSV)

    # Формируем словари
    senses_dct    = build_senses_dct(matched_senses)
    glossary_dict = build_glossary_dict(glossary_csv)

    # Извлекаем правильные английские смыслы (первый элемент, если несколько)
    senses_gold = [opt.split(";")[0] for opt in eng_gold.selected_option]

    # Разбираем киргизские предложения через streamparser
    kyr_units = [get_lex_units(sent) for sent in raw_sentences]

    # Сборка набора
    validation_set = build_validation_set(
        eng_queries, senses_gold, kyr_units, glossary_dict, senses_dct
    )

    # Запись результата
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(validation_set, f, ensure_ascii=False, indent=4)

    print(f"Записано {len(validation_set)} примеров → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
