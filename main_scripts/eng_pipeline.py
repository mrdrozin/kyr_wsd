"""
eng_pipeline.py — Пайплайн разметки данных для английского языка (WSD).

Вход:
  - auxiliary_data/kyrgyz-constitution-2021_corpus.csv   — корпус переводов Конституции
  - auxiliary_data/apertium_constitution.json            — морфоанализ киргизских предложений от Apertium
  - PanLex HuggingFace-датасет 'cointegrated/panlex-meanings' (kir + eng)
  - WordNet (NLTK)

Выход:
  - deepseek_eng_results.json — WSD-разметка: каждое вхождение существительного с выбранными WordNet-синсетами.

Логика: для каждого киргизского существительного находит однозначное выравнивание
с английским существительным через PanLex, собирает WordNet-синсеты и отправляет
запрос к DeepSeek-R1 с CoT-промптом, чтобы выбрать нужный синсет в контексте.
"""

# --- stdlib ---
import json
import os
import re

# --- сторонние ---
import pandas as pd
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from openai import OpenAI
from streamparser import LexicalUnit
import stanza

# --- Пути (относительно корня проекта) ---
CORPUS_PATH = "../auxiliary_data/kyrgyz-constitution-2021_corpus.csv"
APERTIUM_PATH = "../auxiliary_data/apertium_constitution.json"
OUTPUT_PATH = "deepseek_eng_results.json"


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> pd.DataFrame:
    """Загружает корпус переводов Конституции."""
    return pd.read_csv(corpus_path)


def load_panlex_vocab() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает PanLex-словари kir↔eng и возвращает два DataFrame:
    kir_eng_pairs (txt_kir, txt_eng) и eng_to_kir_pairs (txt_eng, txt_kir).
    """
    df_kir = load_dataset("cointegrated/panlex-meanings", name="kir", split="train").to_pandas()
    df_eng = load_dataset("cointegrated/panlex-meanings", name="eng", split="train").to_pandas()

    df_kir_eng = df_kir.merge(df_eng, on="meaning", suffixes=["_kir", "_eng"])
    kir_eng_pairs = df_kir_eng[["txt_kir", "txt_eng"]].drop_duplicates()

    df_eng_to_kir = df_eng.merge(df_kir, on="meaning", suffixes=["_eng", "_kir"])
    eng_to_kir_pairs = df_eng_to_kir[["txt_eng", "txt_kir"]].drop_duplicates()

    return kir_eng_pairs, eng_to_kir_pairs


def load_apertium_data(apertium_path: str) -> list:
    """Загружает предварительно разобранные Apertium-данные из JSON."""
    with open(apertium_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Морфологический анализ (Apertium + Stanza)
# ---------------------------------------------------------------------------

def get_lex_units(units: list) -> list[LexicalUnit]:
    """Преобразует строки из Apertium-вывода в объекты LexicalUnit."""
    return [LexicalUnit(unit) for unit in units]


def parse_unit(unit: LexicalUnit) -> dict:
    """
    Извлекает все именные леммы для одного слова из Apertium-анализа.
    Возвращает словарь с поверхностной формой и списком лемм-существительных.
    """
    nouns = []
    for reading in unit.readings:
        for sread in reading:
            if "n" in sread.tags:
                nouns.append(sread.baseform)
    return {
        "surface": unit.wordform,
        "nouns": list(set(nouns)),
    }


def get_nouns_from_sentence(units: list[LexicalUnit]) -> list[dict]:
    """
    Проходит по всем юнитам предложения и возвращает только существительные
    с их token_id в предложении.
    """
    noun_lemmas = []
    for i, unit in enumerate(units):
        analysis_result = parse_unit(unit)
        analysis_result["token_id"] = i
        if analysis_result["nouns"]:
            noun_lemmas.append(analysis_result)
    return noun_lemmas


# ---------------------------------------------------------------------------
# Перевод и выравнивание
# ---------------------------------------------------------------------------

def translate(words: list[str], vocab: pd.DataFrame,
              lang_source: str = "kir", lang_target: str = "eng") -> list[str]:
    """
    Переводит список лемм через PanLex-словарь.
    Знак ударения (U+0301) убирается — используется в русском пайплайне,
    здесь оставлен для единообразия сигнатуры.
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


def eng_align(noun_translations: list[str], sentence_split: list[tuple]) -> list[tuple]:
    """
    Ищет английские существительные, чья лемма совпадает с переводами киргизского слова.
    sentence_split — список (index, text, stanza_word).
    Возвращает список (index, text, lemma).
    """
    matches = []
    for index, eng_noun, eng_parse in sentence_split:
        if eng_parse.upos == "NOUN":
            if eng_parse.lemma in noun_translations:
                matches.append((index, eng_noun, eng_parse.lemma))
    return matches


# ---------------------------------------------------------------------------
# Основной алгоритм выравнивания
# ---------------------------------------------------------------------------

def build_wsd_queries(kyr_nouns: list, eng_sentences, kir_eng_pairs: pd.DataFrame,
                      nlp) -> tuple[list, list, list]:
    """
    Для каждого предложения строит однозначные пары (киргизское сущ. → английское сущ.).
    Возвращает:
      - wsd_queries:     однозначные выравнивания (1:1)
      - all_wsd_queries: все выравнивания, включая неоднозначные по английской стороне
      - hard_cases:      случаи, когда одно киргизское слово соответствует нескольким английским
    """
    hard_cases = []
    wsd_queries = []
    all_wsd_queries = []
    counts_total = 0

    for sentence_id in tqdm.tqdm(range(len(kyr_nouns)), desc="Aligning sentences"):
        sentence_src = kyr_nouns[sentence_id]
        sentence_tgt = eng_sentences[sentence_id]

        nouns = get_nouns_from_sentence(sentence_src)
        nouns = add_translations(nouns, kir_eng_pairs)

        sent = nlp(sentence_tgt).sentences[0]
        sentence_split = [
            (index, analysis.text, analysis)
            for index, analysis in enumerate(sent.words)
        ]

        counts = [0] * len(sentence_split)
        answers = []

        for unit in nouns:
            res = eng_align(unit["translations"], sentence_split)
            if res:
                counts_total += 1
                eng_matched_lemmas = list({(ind, lemma) for ind, _, lemma in res})

                # один киргизский токен соответствует нескольким английским — сложный случай
                if len(eng_matched_lemmas) > 1:
                    hard_cases.append({
                        "sentence_id": sentence_id,
                        "kyr_token_id": unit["token_id"],
                        "wordform": unit["surface"],
                        "lemmas": [lemma for _, lemma in eng_matched_lemmas],
                        "lemmas_ids": [ind for ind, _ in eng_matched_lemmas],
                    })
                    continue

                matched_lemma = eng_matched_lemmas[0]
                dct = {
                    "sentence_id": sentence_id,
                    "kyr_token_id": unit["token_id"],
                    "wordform": unit["surface"],
                    "kyr_lemmas": unit["nouns"],
                    "lemma": matched_lemma[1],
                    "lemma_id": matched_lemma[0],
                }
                counts[matched_lemma[0]] += 1
                answers.append(dct)
                all_wsd_queries.append(dct)

        for dct in answers:
            # оставляем только однозначные выравнивания (1:1)
            if counts[dct["lemma_id"]] == 1:
                wsd_queries.append(dct)

    print(f"Всего совпадений: {counts_total}")
    print(f"all_wsd_queries: {len(all_wsd_queries)}")
    print(f"wsd_queries (1:1): {len(wsd_queries)}")
    print(f"hard_cases: {len(hard_cases)}")

    return wsd_queries, all_wsd_queries, hard_cases


# ---------------------------------------------------------------------------
# Подготовка данных для DeepSeek
# ---------------------------------------------------------------------------

def prepare_final_dicts(wsd_queries: list, eng_sentences) -> list[dict]:
    """
    Для каждого выравнивания достаёт WordNet-синсеты и формирует запрос:
    предложение с маркерами <WSD>, слово и список описаний синсетов.
    """
    final_dicts = []
    for query in wsd_queries:
        sentence_id = query["sentence_id"]
        lemma = query["lemma"]
        lemma_id = query["lemma_id"]

        sentence_tokens = eng_sentences[sentence_id].split()
        sentence_tokens.insert(lemma_id + 1, "<WSD>")
        sentence_tokens.insert(lemma_id, "<WSD>")

        synsets = wn.synsets(lemma)
        synset_descriptions = [
            "; ".join([str(sns.name()), lemma, str(sns.definition())])
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
        "You are going to identify the corresponding sense tag of an ambiguity word in English sentences. Do the following tasks.",
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
    eng_sentences = data["english_tokenized13a"]

    # Загрузка PanLex-словарей
    kir_eng_pairs, _ = load_panlex_vocab()

    # Загрузка Apertium-анализа
    kyr_json = load_apertium_data(APERTIUM_PATH)
    kyr_nouns = [get_lex_units(units) for units in kyr_json]

    # Инициализация Stanza для английского
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma")

    # Построение WSD-запросов через выравнивание
    wsd_queries, _, _ = build_wsd_queries(kyr_nouns, eng_sentences, kir_eng_pairs, nlp)

    # Подготовка данных для API
    final_dicts = prepare_final_dicts(wsd_queries, eng_sentences)

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
