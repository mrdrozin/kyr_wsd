"""
Перевод контекстных предложений SemCor с английского на киргизский через DeepSeek API.

Логика:
1. Читает SemCor.jsonl, фильтрует существительные (≤8 вхождений на лемму,
   однозначные, вне валидационного множества из eng_gold.csv).
2. Формирует контексты с маркерами [TGT] вокруг целевого слова.
3. Переводит батчами по 20 через deepseek-chat (temperature=0),
   сохраняя маркеры [TGT] в переводе.
4. Проверяет каждое предложение: если [TGT] потеряны или строка пуста —
   повторяет одиночный запрос.
5. Сохраняет список переведённых предложений в kyr_sentences.json.
"""

import json
import os
from collections import Counter

import tqdm
from dotenv import load_dotenv
from nltk import WordNetLemmatizer as wnl
from nltk.corpus import wordnet as wn
from openai import OpenAI
from tqdm import trange
import pandas as pd

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
SEMCOR_PATH = "../wsd-data/data/jsonl/SemCor.jsonl"
ENG_GOLD_PATH = "../gold_data/eng_gold.csv"
KYR_SENTENCES_OUT_PATH = "kyr_sentences.json"

MAX_LEMMA_COUNT = 8
BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Утилиты чтения данных
# ---------------------------------------------------------------------------

def read_jsonl(file_path: str, cnt: int = 35_000):
    """Построчно читает JSONL-файл, отдаёт не более cnt записей."""
    with open(file_path, "r", encoding="utf-8") as f:
        read_lines = 0
        for line in f:
            line = line.strip()
            if line:
                read_lines += 1
                yield json.loads(line)
            if cnt == read_lines:
                break


def load_test_nouns(eng_gold_path: str) -> set:
    """Возвращает множество лемматизированных существительных из валидационного набора."""
    df = pd.read_csv(eng_gold_path)
    lemmatizer = wnl()
    test_nouns = set()
    for word in df.word:
        lemma = lemmatizer.lemmatize(word, "n")
        test_nouns.add(lemma)
    return test_nouns


def filter_train_set(lines: list, test_nouns: set) -> list:
    """
    Оставляет только однозначные существительные, не входящие в валидационное
    множество, с ограничением MAX_LEMMA_COUNT вхождений на лемму.
    """
    cnt = Counter()
    train_set = []
    for line in lines:
        if (
            cnt[line["lemma"]] < MAX_LEMMA_COUNT
            and line["pos"] == "NOUN"
            and line["lemma"] not in test_nouns
            and len(line["sense"].split(";")) == 1
        ):
            train_set.append(line)
            cnt[line["lemma"]] += 1
    return train_set


# ---------------------------------------------------------------------------
# Работа с предложениями
# ---------------------------------------------------------------------------

def mask_target(sentence: str, start: int, end: int) -> str:
    """Вставляет маркеры [TGT] вокруг целевого слова в токенизированном предложении."""
    tokens = sentence.split()
    tokens.insert(end, "[TGT]")
    tokens.insert(start, "[TGT]")
    return " ".join(tokens)


def get_gloss(sense_key: str) -> str:
    """Возвращает глоссу синсета по sense_key из WordNet."""
    lemma = wn.lemma_from_key(sense_key)
    return lemma.synset().definition()


def prepare_for_translation(records: list) -> tuple[list, list]:
    """
    Для каждой записи формирует замаскированный контекст и английскую глоссу.
    Возвращает (contexts_en, glosses_en).
    """
    contexts_en, glosses_en = [], []
    for rec in records:
        masked = mask_target(rec["sentence"], rec["start"], rec["end"])
        gloss = get_gloss(rec["sense"])
        contexts_en.append(masked)
        glosses_en.append(gloss)
    return contexts_en, glosses_en


# ---------------------------------------------------------------------------
# Перевод через DeepSeek
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """Создаёт OpenAI-совместимый клиент для DeepSeek."""
    load_dotenv()
    key = os.getenv("DEEPSEEK")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def translate_batch(client: OpenAI, texts: list[str]) -> list[str]:
    """
    Переводит батч предложений с маркерами [TGT] на киргизский.
    Маркеры [TGT] должны сохраниться вокруг целевого слова в переводе.
    Возбуждает ValueError, если количество переводов не совпадает с входом.
    """
    prompt = (
        "Translate each of the following English lines to Kyrgyz."
        "The tokens [TGT] are special markers that surround the target word."
        "They MUST remain in the translation exactly as they are, and they MUST surround the translated target word."
        "DO NOT translate the [TGT] markers themselves."
        "Return the translations separated by ' ||| ' in the same order, preserving all punctuation and the exact placement of [TGT] around the appropriate word."
    )
    batch_text = " ||| ".join(texts)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": batch_text},
    ]
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
    )
    translated = resp.choices[0].message.content.split(" ||| ")
    if len(translated) != len(texts):
        raise ValueError("Количество переводов не совпало с исходным")
    return [t.strip() for t in translated]


def translate_single(client: OpenAI, text: str) -> str:
    """Переводит одно предложение на киргизский, сохраняя маркеры [TGT]."""
    prompt = (
        "Translate the following English line to Kyrgyz."
        "The tokens [TGT] are special markers that surround the target word."
        "They MUST remain in the translation exactly as they are, and they MUST surround the translated target word."
        "DO NOT translate the [TGT] markers themselves."
        "Return the translation preserving all punctuation and the exact placement of [TGT] around the appropriate word."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content


def translate_contexts(client: OpenAI, contexts: list[str]) -> list[str]:
    """
    Переводит список контекстных предложений батчами BATCH_SIZE.
    При ошибке в батче подставляет пустые строки для сохранения индексации.
    Возвращает список переведённых предложений той же длины, что и contexts.
    """
    translated_contexts = []
    problem_index = []

    # Основной проход батчами
    for i in trange(0, len(contexts), BATCH_SIZE):
        try:
            res = translate_batch(client, contexts[i : i + BATCH_SIZE])
            translated_contexts.extend(res)
        except ValueError:
            # Сохраняем структуру: пустые строки на месте проблемного батча
            translated_contexts.extend([""] * len(contexts[i : i + BATCH_SIZE]))
            problem_index.append(i)

    return translated_contexts, problem_index


def fix_bad_translations(
    client: OpenAI, contexts: list[str], translated_contexts: list[str]
) -> list[str]:
    """
    Проверяет каждое переведённое предложение на корректность маркеров [TGT].
    Предложение считается некорректным, если оно пустое или [TGT] не делят его
    ровно на 3 части. Такие предложения переводятся повторно одиночным запросом.
    """
    for i in tqdm.trange(len(translated_contexts)):
        cur_sent = translated_contexts[i]
        splt = cur_sent.split("[TGT]")
        # Корректное предложение: ровно 2 маркера [TGT] → 3 части
        if len(splt) != 3 or cur_sent == "":
            translated = translate_single(client, contexts[i])
            translated_contexts[i] = translated.strip()
    return translated_contexts


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def main():
    # 1. Загрузка и фильтрация данных SemCor
    test_nouns = load_test_nouns(ENG_GOLD_PATH)
    lines = list(read_jsonl(SEMCOR_PATH, cnt=int(10e6)))
    train_set = filter_train_set(lines, test_nouns)

    # 2. Формирование контекстов с маркерами [TGT]
    contexts, _ = prepare_for_translation(train_set)

    # 3. Инициализация клиента
    client = build_client()

    # 4. Перевод батчами
    print("Перевод контекстов батчами...")
    translated_contexts, _ = translate_contexts(client, contexts)

    # 5. Исправление предложений с потерянными маркерами [TGT]
    print("Исправление некорректных переводов...")
    translated_contexts = fix_bad_translations(client, contexts, translated_contexts)

    # 6. Сохранение результата
    with open(KYR_SENTENCES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translated_contexts, f, ensure_ascii=False, indent=4)
    print(f"Переведённые контексты сохранены: {KYR_SENTENCES_OUT_PATH}")


if __name__ == "__main__":
    main()
