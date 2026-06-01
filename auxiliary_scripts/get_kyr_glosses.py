"""
Перевод глосс WordNet с английского на киргизский через DeepSeek API.

Логика:
1. Читает SemCor.jsonl, фильтрует существительные (≤8 вхождений на лемму,
   однозначные, вне валидационного множества из eng_gold.csv).
2. Извлекает уникальные глоссы WordNet для отобранных примеров.
3. Переводит батчами по 20 через deepseek-chat (temperature=0).
4. Повторяет одиночные вызовы для проблемных батчей.
5. Дополнительно переводит глоссы из auxiliary_data/missing_glosses.json
   (тем же способом), затем объединяет с existing auxiliary_data/glosses_dct.json
   и сохраняет в auxiliary_data/all_glosses_dct.json.
6. Основной словарь (только из SemCor) сохраняется в glosses_dct.json.
"""

import json
import os
from collections import Counter, defaultdict

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
MISSING_GLOSSES_PATH = "../auxiliary_data/missing_glosses.json"
EXISTING_GLOSSES_PATH = "../auxiliary_data/glosses_dct.json"
ALL_GLOSSES_OUT_PATH = "../auxiliary_data/all_glosses_dct.json"
GLOSSES_DCT_OUT_PATH = "glosses_dct.json"

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
# Работа с WordNet
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
    Переводит батч английских строк на киргизский через deepseek-chat.
    Возбуждает ValueError, если количество переводов не совпадает с входом.
    """
    prompt = (
        "Translate each of the following English lines to Kyrgyz."
        "Return the translations separated by ' ||| ' in the same order, preserving all punctuation."
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
    """Переводит одну строку на киргизский (используется для повтора проблемных)."""
    prompt = (
        "Translate each the following English line to Kyrgyz."
        "Return the translation preserving all punctuation."
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


def translate_glosses(client: OpenAI, glosses_list: list[str]) -> dict:
    """
    Переводит список глосс батчами BATCH_SIZE.
    При ошибке в батче повторяет одиночными вызовами.
    Возвращает словарь {eng_gloss: kyr_gloss}.
    """
    glosses_dct = defaultdict(str)
    problem_index = []

    # Основной проход батчами
    for i in trange(0, len(glosses_list), BATCH_SIZE):
        try:
            batch = glosses_list[i : i + BATCH_SIZE]
            res = translate_batch(client, batch)
            for eng_gloss, kyr_gloss in zip(batch, res):
                glosses_dct[eng_gloss] = kyr_gloss
        except ValueError:
            for eng_gloss in glosses_list[i : i + BATCH_SIZE]:
                glosses_dct[eng_gloss] = ""
            problem_index.append(i)

    # Повторный проход одиночными вызовами для проблемных батчей
    for problem_ind in tqdm.tqdm(problem_index):
        for i in range(problem_ind, problem_ind + BATCH_SIZE):
            if i >= len(glosses_list):
                break
            eng_gloss = glosses_list[i]
            kyr_gloss = translate_single(client, eng_gloss)
            glosses_dct[eng_gloss] = kyr_gloss

    return dict(glosses_dct)


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def main():
    # 1. Загрузка и фильтрация данных SemCor
    test_nouns = load_test_nouns(ENG_GOLD_PATH)
    lines = list(read_jsonl(SEMCOR_PATH, cnt=int(10e6)))
    train_set = filter_train_set(lines, test_nouns)

    # 2. Извлечение уникальных глосс
    _, glosses = prepare_for_translation(train_set)
    glosses_unique = list(set(glosses))

    # 3. Инициализация клиента
    client = build_client()

    # 4. Перевод основных глосс (из SemCor)
    print("Перевод основных глосс...")
    glosses_dct = translate_glosses(client, glosses_unique)

    # 5. Сохранение основного словаря
    with open(GLOSSES_DCT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(glosses_dct, f, indent=4, ensure_ascii=False)
    print(f"Основной словарь сохранён: {GLOSSES_DCT_OUT_PATH}")

    # 6. Перевод пропущенных глосс из missing_glosses.json
    with open(MISSING_GLOSSES_PATH, "r") as f:
        missing_glosses = json.load(f)

    print("Перевод пропущенных глосс...")
    missing_glosses_dct = translate_glosses(client, missing_glosses)

    # 7. Объединение с существующим словарём и сохранение
    with open(EXISTING_GLOSSES_PATH) as f:
        existing = json.load(f)
    existing.update(missing_glosses_dct)
    with open(ALL_GLOSSES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=4)
    print(f"Объединённый словарь сохранён: {ALL_GLOSSES_OUT_PATH}")


if __name__ == "__main__":
    main()
