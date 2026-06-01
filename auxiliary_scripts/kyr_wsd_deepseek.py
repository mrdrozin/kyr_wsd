"""
Разрешение лексической многозначности (WSD) киргизских слов через DeepSeek-R1.

Скрипт:
  1. Загружает валидационный датасет и морфологический разбор Апертиума.
  2. Формирует запросы к модели deepseek-reasoner (JSON-режим).
  3. Сохраняет ответы с предсказанными senseID в JSON-файл.
"""

import json
import os

import numpy as np
import pandas as pd
import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from streamparser import LexicalUnit

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
VALIDATION_SET_PATH = "../kyr_wsd_dataset/validation_set.json"
APERTIUM_CORPUS_PATH = "../auxiliary_data/apertium_constitution.json"
GOLD_OUTPUT_PATH = "../gold_data/kyr_gold.csv"
RESULTS_OUTPUT_PATH = "../deepseek_answers/deepseek_kyr_results.json"

# Словарь для разрешения неоднозначных лемм, которые Апертиум разбирает двояко
HARD_DCT = {
    frozenset(("ата", "атан")): "ата",
    frozenset(("бал", "бала")): "бала",
    frozenset(("жара", "жаран")): "жаран",
    frozenset(("күн", "күнү")): "күн",
    frozenset(("уй", "уюм")): "уюм",
}


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_lex_units(units):
    """Преобразует список строковых записей Апертиума в объекты LexicalUnit."""
    return [LexicalUnit(unit) for unit in units]


def get_lemma(sentence, kyr_token_id):
    """
    Извлекает базовую форму целевого существительного из разбора Апертиума.
    При нескольких вариантах леммы использует HARD_DCT.
    """
    target_unit = sentence[kyr_token_id]
    nouns = set()
    for reading in target_unit.readings:
        for sread in reading:
            if "n" in sread.tags:
                nouns.add(sread.baseform)

    if len(nouns) == 1:
        return nouns.pop()
    return HARD_DCT[frozenset(nouns)]


def get_cot_prompt(word: str, sentence: str, synset_descriptions) -> str:
    """
    Формирует Chain-of-Thought промпт для DeepSeek на English (как в ноутбуке).
    Возвращает строку промпта.
    """
    parts = [
        "You are going to identify the corresponding sense tag of an ambiguity word "
        "in Kyrgyz sentences. Do the following tasks.",
        f"1. {word} has different meanings. Below are possible meanings. "
        "Comprehend the sensetags and meanings.",
    ]
    parts.extend(synset_descriptions)
    parts.append(
        "2. Now examine the sentence below. You are going to identify the most "
        "suitable meaning for ambiguity word."
    )
    parts.append(sentence)
    parts.extend([
        "3. Try to identify the meaning of the word in the above sentence which is "
        "enclosed with the <WSD>. You can think of the real meaning of sentence and "
        "decide the most suitable meaning for the word.",
        "4. Based on the identified meaning, try to find the most appropriate sense "
        "from the below. You are given definition of each sense tag too.",
    ])
    parts.extend(synset_descriptions)
    parts.extend([
        "5. If you have more than one senses identified after above steps, you can "
        "return the numbers in order of confidence level.",
        "6. Return JSON object that contains the ambiguity word and the finalized "
        "senseIDs. Use the following format for the output.",
        "<JSON Object with fields ambiguity_word and senseIDs >",
    ])
    return "\n".join(parts)


def load_data():
    """Загружает датасет и разбор Апертиума; возвращает DataFrame и список предложений."""
    with open(VALIDATION_SET_PATH, encoding="utf-8") as f:
        kyr_data = json.load(f)
    kyr_df = pd.DataFrame(kyr_data)

    with open(APERTIUM_CORPUS_PATH, encoding="utf-8") as f:
        kyr_sentences_raw = json.load(f)

    kyr_units = [get_lex_units(sent) for sent in kyr_sentences_raw]
    return kyr_df, kyr_units


def build_queries_and_gold(kyr_df, kyr_units):
    """
    Формирует список запросов к API (kyr_queries) и золотую разметку (kyr_gold).
    """
    kyr_gold = []
    kyr_queries = []

    grouped = kyr_df.groupby("instance_id")
    for instance_id, group in grouped:
        context_tgt = group["context"].tolist()[0]
        context = " <WSD> ".join(context_tgt.split("[TGT]"))
        labels = group["label"].tolist()

        sentence_id, kyr_token_id = [int(a) for a in instance_id.split("_")]
        wsd_word = get_lemma(kyr_units[sentence_id], kyr_token_id)

        glosses = group["gloss"].tolist()
        descriptions = [f"{i+1}) {gloss}" for i, gloss in enumerate(glosses)]

        kyr_gold.append({
            "item_index": instance_id,
            "word": wsd_word,
            "selected_option": int(np.argmax(labels)),
        })
        kyr_queries.append({
            "sentence_id": sentence_id,
            "kyr_token_id": kyr_token_id,
            "sentence": context,
            "word": wsd_word,
            "descriptions": descriptions,
        })

    return kyr_queries, kyr_gold


def call_deepseek(client, word, sentence, descriptions):
    """Выполняет один запрос к deepseek-reasoner и возвращает строку JSON-ответа."""
    prompt = get_cot_prompt(word, sentence, descriptions)
    messages = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def run_deepseek_inference(client, kyr_queries):
    """
    Проходит по всем запросам, получает ответы DeepSeek.
    При ошибке парсинга JSON делает повторный запрос.
    """
    answers = []
    for query in tqdm.tqdm(kyr_queries):
        content = call_deepseek(
            client, query["word"], query["sentence"], query["descriptions"]
        )
        answers.append(content)

    # Разбираем ответы; при неудаче повторяем запрос
    for query, answer in tqdm.tqdm(zip(kyr_queries, answers)):
        try:
            answer_json = json.loads(answer)
            query["senseIDs"] = answer_json["senseIDs"]
        except Exception:
            content = call_deepseek(
                client, query["word"], query["sentence"], query["descriptions"]
            )
            sense_ids = json.loads(content)
            query["senseIDs"] = sense_ids["senseIDs"]

    # Переводим senseIDs из 1-based в 0-based
    for query in kyr_queries:
        query["senseIDs"] = [int(sid) - 1 for sid in query["senseIDs"]]

    return kyr_queries


def main():
    # Загрузка API-ключа из .env
    load_dotenv()
    api_key = os.getenv("DEEPSEEK")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Загрузка данных
    kyr_df, kyr_units = load_data()

    # Формирование запросов и золотой разметки
    kyr_queries, kyr_gold = build_queries_and_gold(kyr_df, kyr_units)

    # Сохранение золотой разметки
    os.makedirs(os.path.dirname(GOLD_OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(kyr_gold).to_csv(GOLD_OUTPUT_PATH, index=False)
    print(f"Золотая разметка сохранена: {GOLD_OUTPUT_PATH}")

    # Инференс через DeepSeek
    kyr_queries_with_answers = run_deepseek_inference(client, kyr_queries)

    # Сохранение результатов
    os.makedirs(os.path.dirname(RESULTS_OUTPUT_PATH), exist_ok=True)
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kyr_queries_with_answers, f, indent=4, ensure_ascii=False)
    print(f"Результаты DeepSeek сохранены: {RESULTS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
