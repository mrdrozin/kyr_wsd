"""
ru_pipeline.py — Пайплайн разметки данных для русского языка (WSD).

Вход:
  - data/kyrgyz-constitution-2021_corpus.csv  — корпус переводов Конституции
  - PanLex HuggingFace-датасет 'cointegrated/panlex-meanings' (kir + rus)
  - Apertium (on-the-fly анализ киргизского через пакет apertium)
  - RuWordNet

Выход:
  - deepseek_rus_results.json — WSD-разметка: каждое вхождение существительного
    с выбранными RuWordNet-синсетами.

Логика: для каждого киргизского существительного находит однозначное выравнивание
с русским существительным через PanLex (со снятием диакритики), собирает
RuWordNet-синсеты и отправляет запрос к DeepSeek-R1 с CoT-промптом.
"""

# --- stdlib ---
import json
import os
import re

# --- сторонние ---
import apertium
import pandas as pd
import pymorphy3
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from ruwordnet import RuWordNet

# --- Пути (относительно корня проекта) ---
CORPUS_PATH = "data/kyrgyz-constitution-2021_corpus.csv"
OUTPUT_PATH = "deepseek_rus_results.json"


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> pd.DataFrame:
    """Загружает корпус переводов Конституции."""
    return pd.read_csv(corpus_path)


def strip_accent(x) -> str | None:
    """Убирает знак ударения (U+0301) из строки; None/NaN пропускает."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    return re.sub(r"́", "", str(x))


def load_panlex_vocab() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает PanLex-словари kir↔rus.
    Из русского столбца удаляются знаки ударения.
    Возвращает kir_rus_pairs и rus_to_kir_pairs.
    """
    df_kir = load_dataset("cointegrated/panlex-meanings", name="kir", split="train").to_pandas()
    df_rus = load_dataset("cointegrated/panlex-meanings", name="rus", split="train").to_pandas()

    # снимаем ударение в русских токенах
    df_rus["txt"] = df_rus["txt"].apply(strip_accent)

    df_kir_rus = df_kir.merge(df_rus, on="meaning", suffixes=["_kir", "_rus"])
    kir_rus_pairs = df_kir_rus[["txt_kir", "txt_rus"]].drop_duplicates()

    df_rus_to_kir = df_rus.merge(df_kir, on="meaning", suffixes=["_rus", "_kir"])
    rus_to_kir_pairs = df_rus_to_kir[["txt_rus", "txt_kir"]].drop_duplicates()

    return kir_rus_pairs, rus_to_kir_pairs


# ---------------------------------------------------------------------------
# Морфологический анализ (Apertium + pymorphy3)
# ---------------------------------------------------------------------------

def parse_unit(unit) -> dict:
    """
    Извлекает именные леммы из Apertium-анализа одного слова.
    Возвращает словарь с поверхностной формой, разборами и леммами-существительными.
    """
    analyses = []
    nouns = []
    for reading in unit.readings:
        morphemes = []
        for sread in reading:
            if "n" in sread.tags:
                morphemes.append({"lemma": sread.baseform, "tags": sread.tags})
                nouns.append(sread.baseform)
        if morphemes:
            analyses.append(morphemes)

    return {
        "surface": unit.wordform,
        "analyses": analyses,
        "nouns": list(set(nouns)),
    }


def get_nouns_from_sentence(sentence: str, analyzer: apertium.Analyzer) -> list[dict]:
    """
    Разбирает киргизское предложение через Apertium и возвращает только
    токены, у которых есть именные леммы, с их позицией (token_id).
    """
    units = analyzer.analyze(sentence)
    noun_lemmas = []
    for i, unit in enumerate(units):
        analysis_result = parse_unit(unit)
        analysis_result["token_id"] = i
        if analysis_result["analyses"]:
            noun_lemmas.append(analysis_result)
    return noun_lemmas


# ---------------------------------------------------------------------------
# Перевод и выравнивание
# ---------------------------------------------------------------------------

def translate(words: list[str], vocab: pd.DataFrame,
              lang_source: str = "kir", lang_target: str = "rus") -> list[str]:
    """
    Переводит список лемм через PanLex-словарь.
    Знак ударения (U+0301) убирается из результатов.
    """
    matches_arrays = [
        vocab.loc[vocab[f"txt_{lang_source}"] == word, f"txt_{lang_target}"].unique()
        for word in words
    ]
    matches = {re.sub("́", "", token) for arr in matches_arrays for token in arr}
    return list(matches)


def add_translations(token_dct: list[dict], vocab: pd.DataFrame) -> list[dict]:
    """Добавляет поле 'translations' к каждому существительному в списке."""
    for unit in token_dct:
        unit["translations"] = translate(unit["nouns"], vocab)
    return token_dct


def rus_align(nouns: list[str], sentence_split: list[tuple]) -> list[tuple]:
    """
    Ищет русские существительные, чья нормальная форма совпадает с переводами.
    sentence_split — список (index, token_text, pymorphy3_parse).
    Возвращает список (index, text, normal_form).
    """
    matches = []
    for index, rus_noun, rus_parse in sentence_split:
        if rus_parse.tag.POS == "NOUN":
            if rus_parse.normal_form in nouns:
                matches.append((index, rus_noun, rus_parse.normal_form))
    return matches


# ---------------------------------------------------------------------------
# Основной алгоритм выравнивания
# ---------------------------------------------------------------------------

def build_wsd_queries(kyr_sentences, rus_sentences, kir_rus_pairs: pd.DataFrame,
                      analyzer: apertium.Analyzer,
                      morph: pymorphy3.MorphAnalyzer) -> tuple[list, list]:
    """
    Для каждого предложения строит однозначные пары (киргизское сущ. → русское сущ.).
    Возвращает:
      - wsd_queries:  однозначные выравнивания (1:1)
      - hard_cases:   случаи, когда одно киргизское слово соответствует нескольким русским
    """
    hard_cases = []
    wsd_queries = []

    for sentence_id in tqdm.tqdm(range(len(kyr_sentences)), desc="Aligning sentences"):
        sentence_src = kyr_sentences[sentence_id]
        sentence_tgt = rus_sentences[sentence_id]

        nouns = get_nouns_from_sentence(sentence_src, analyzer)
        nouns = add_translations(nouns, kir_rus_pairs)

        sentence_split = [
            (index, token, morph.parse(token)[0])
            for index, token in enumerate(sentence_tgt.split())
        ]

        for unit in nouns:
            res = rus_align(unit["translations"], sentence_split)
            if res:
                lemmas = list({(ind, lemma) for ind, _, lemma in res})

                # один киргизский токен соответствует нескольким русским — сложный случай
                if len(lemmas) > 1:
                    hard_cases.append({
                        "sentence_id": sentence_id,
                        "kyr_token_id": unit["token_id"],
                        "wordform": unit["surface"],
                        "lemmas": [lemma for _, lemma in lemmas],
                        "lemmas_ids": [ind for ind, _ in lemmas],
                    })
                    continue

                # однозначное выравнивание (проверено вручную: len == 1)
                wsd_queries.append({
                    "sentence_id": sentence_id,
                    "kyr_token_id": unit["token_id"],
                    "wordform": unit["surface"],
                    "lemma": lemmas[0][1],
                    "lemma_id": lemmas[0][0],
                })

    print(f"wsd_queries (1:1): {len(wsd_queries)}")
    print(f"hard_cases: {len(hard_cases)}")
    return wsd_queries, hard_cases


# ---------------------------------------------------------------------------
# Подготовка данных для DeepSeek
# ---------------------------------------------------------------------------

def prepare_final_dicts(wsd_queries: list, rus_sentences,
                        wn: RuWordNet) -> list[dict]:
    """
    Для каждого выравнивания достаёт RuWordNet-синсеты и формирует запрос:
    предложение с маркерами <WSD>, слово и список описаний синсетов.
    """
    final_dicts = []
    for query in wsd_queries:
        sentence_id = query["sentence_id"]
        lemma = query["lemma"]
        lemma_id = query["lemma_id"]

        sentence_tokens = rus_sentences[sentence_id].split()
        sentence_tokens.insert(lemma_id + 1, "<WSD>")
        sentence_tokens.insert(lemma_id, "<WSD>")

        synsets = wn.get_synsets(lemma)
        synset_descriptions = [
            "; ".join([str(sns.id), str(sns.title), str(sns.definition)])
            for sns in synsets
        ]

        final_dicts.append({
            "sentence": " ".join(sentence_tokens),
            "word": lemma,
            "descriptions": synset_descriptions,
        })
    return final_dicts


# ---------------------------------------------------------------------------
# Промпт и вызов DeepSeek
# ---------------------------------------------------------------------------

def get_cot_prompt(word: str, sentence: str, synset_descriptions: list[str]) -> str:
    """
    Формирует CoT-промпт для WSD по методологии из статьи об оценке LLM для WSD.
    Целевое слово выделено тегами <WSD>.
    """
    parts = [
        "You are going to identify the corresponding sense tag of an ambiguity word in Russian sentences. Do the following tasks.",
        f"1. {word} has different meanings. Below are possible meanings. Comprehend the sensetags and meanings.",
    ]
    parts.extend(synset_descriptions)
    parts.append("2. Now examine the sentence below. You are going to identify the most suitable meaning for ambiguity word.")
    parts.append(sentence)
    parts.extend([
        "3. Try to identify the meaning of the word in the above sentence which is enclosed with the <WSD>. You can think of the real meaning of "
        "sentence and decide the most suitable meaning for the word.",
        "4. Based on the identified meaning, try to find the most appropriate senseIDs from the below. You are given definition of each sense tag too.",
    ])
    parts.extend(synset_descriptions)
    parts.extend([
        "5. If you have more than one senseIDs identified after above steps, you can return the senseIDs in order of confidence level.",
        "6. Return JSON object that contains the ambiguity word and the finalized senseIDs. Use the following format for the output.",
        "<JSON Object with fields ambiguity_word and senseIDs >",
    ])
    return "\n".join(parts)


def annotate_with_deepseek(final_dicts: list[dict], client: OpenAI) -> list[dict]:
    """
    Вызывает DeepSeek-R1 для каждого запроса и добавляет поле 'annotation'
    с результатом WSD (ambiguity_word + senseIDs).
    """
    for dct in tqdm.tqdm(final_dicts, desc="DeepSeek annotation"):
        prompt = get_cot_prompt(dct["word"], dct["sentence"], dct["descriptions"])
        messages = [{"role": "system", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            response_format={"type": "json_object"},
        )
        dct["annotation"] = json.loads(response.choices[0].message.content)
    return final_dicts


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    # Загрузка корпуса
    data = load_corpus(CORPUS_PATH)
    kyr_sentences = data["kyrgyz_tokenized13a"]
    rus_sentences = data["russian_tokenized13a"]

    # Загрузка PanLex-словарей (с нормализацией ударений)
    kir_rus_pairs, _ = load_panlex_vocab()

    # Инициализация Apertium-анализатора для киргизского
    analyzer = apertium.Analyzer("kir")

    # Инициализация морфоанализатора для русского
    morph = pymorphy3.MorphAnalyzer()

    # Построение WSD-запросов через выравнивание
    wsd_queries, _ = build_wsd_queries(
        kyr_sentences, rus_sentences, kir_rus_pairs, analyzer, morph
    )

    # Загрузка RuWordNet
    wn = RuWordNet()

    # Подготовка данных для API
    final_dicts = prepare_final_dicts(wsd_queries, rus_sentences, wn)

    # Инициализация DeepSeek-клиента
    load_dotenv()
    key = os.getenv("DEEPSEEK")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    # Аннотирование через DeepSeek-R1
    final_dicts = annotate_with_deepseek(final_dicts, client)

    # Сохранение результатов
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_dicts, f, indent=4, ensure_ascii=False)
    print(f"Результаты сохранены в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
