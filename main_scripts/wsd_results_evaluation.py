"""
Оценка качества WSD-системы DeepSeek и BERT-модели по трём языкам.

Читает gold-аннотации (CSV) и ответы DeepSeek (JSON) для русского, английского
и киргизского языков, а также предсказания BERT-модели для киргизского.
Считает Macro-P / Macro-R / Macro-F1 и Accuracy для каждого набора.

Использование:
    python wsd_results_evaluation.py
"""

import json

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Пути к данным
# ---------------------------------------------------------------------------
# Золотые аннотации
RUS_GOLD_CSV = "../gold_data/rus_gold.csv"
ENG_GOLD_CSV = "../gold_data/eng_gold.csv"
KYR_GOLD_CSV = "../gold_data/kyr_gold.csv"

# Ответы DeepSeek
RUS_DEEPSEEK_JSON = "../deepseek_answers/deepseek_rus_results.json"
ENG_DEEPSEEK_JSON = "../deepseek_answers/deepseek_eng_results.json"
KYR_DEEPSEEK_JSON = "../deepseek_answers/deepseek_kyr_results.json"


def evaluate_results(
    sentences: list,
    descriptions_list: list,
    gold: list,
    deepseek_answers: list,
    use_glosses: bool = False,
    skip_one_def: bool = True,
) -> None:
    """Вычисляет и печатает метрики WSD для одного языка/системы.

    Параметры
    ----------
    sentences : list
        Контекстные предложения (нужны для дедупликации).
    descriptions_list : list
        Список словарей/списков глоссов для каждого экземпляра.
    gold : list
        Золотые метки (идентификаторы смыслов или индексы).
    deepseek_answers : list
        Предсказанные ответы системы (список или скаляр).
    use_glosses : bool
        Если True — сравниваем текстовые глоссы, иначе — идентификаторы смыслов.
    skip_one_def : bool
        Если True — пропускаем слова с единственной дефиницией.
    """
    all_ans = 0
    sentence_set: set = set()
    y_true, y_pred = [], []

    for sentence, descriptions, right_ans, predicted_answers in zip(
        sentences, descriptions_list, gold, deepseek_answers
    ):
        # Пропускаем повторяющиеся предложения
        if sentence in sentence_set:
            continue
        sentence_set.add(sentence)

        # Пропускаем слова с одной дефиницией и спорные случаи
        if (
            (skip_one_def and len(descriptions) == 1)
            or right_ans == "Нет верного варианта"
            or right_ans == "Спорный случай"
        ):
            continue

        all_ans += 1

        predicted_ans = (
            predicted_answers if isinstance(predicted_answers, list) else [predicted_answers]
        )

        if use_glosses:
            y_true.append(descriptions[right_ans])
            y_pred.append(descriptions[predicted_ans[0]] if len(predicted_ans) > 0 else None)
        else:
            y_true.append(right_ans)
            y_pred.append(predicted_ans[0] if len(predicted_ans) > 0 else None)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"Macro-P: {macro_prec:.4f}, Macro-R: {macro_rec:.4f}, Macro-F1: {macro_f1:.4f}")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


def load_language_data(
    gold_csv: str,
    deepseek_json: str,
    gold_col_split: bool = True,
    sense_id_col: str = "senseIDs",
) -> tuple[list, list, list, list]:
    """Загружает данные одного языка из CSV и JSON.

    Параметры
    ----------
    gold_csv : str
        Путь к CSV с золотыми аннотациями (колонка selected_option).
    deepseek_json : str
        Путь к JSON с ответами DeepSeek.
    gold_col_split : bool
        Если True — берём первый элемент после split(';') из selected_option.
    sense_id_col : str
        Имя колонки в JSON с предсказанными идентификаторами смыслов.

    Возвращает
    ----------
    sentences, descriptions, gold, deepseek_answers
    """
    annotations = pd.read_csv(gold_csv)
    queries = pd.read_json(deepseek_json)

    sentences = list(queries["sentence"])
    descriptions = list(queries["descriptions"])

    if gold_col_split:
        gold = list(annotations["selected_option"].str.split(";").str[0])
    else:
        gold = list(annotations["selected_option"])

    deepseek_answers = list(queries[sense_id_col])
    return sentences, descriptions, gold, deepseek_answers


def main() -> None:
    """Загружает данные и запускает оценку для каждого языка и системы."""

    # --- Русский: DeepSeek ---
    print("=== Русский (DeepSeek) ===")
    rus_sentences, rus_descriptions, rus_gold, rus_deepseek_answers = load_language_data(
        RUS_GOLD_CSV, RUS_DEEPSEEK_JSON, gold_col_split=True, sense_id_col="senseIDs"
    )
    evaluate_results(rus_sentences, rus_descriptions, rus_gold, rus_deepseek_answers)

    # --- Английский: DeepSeek ---
    print("\n=== Английский (DeepSeek) ===")
    eng_sentences, eng_descriptions, eng_gold, eng_deepseek_answers = load_language_data(
        ENG_GOLD_CSV, ENG_DEEPSEEK_JSON, gold_col_split=True, sense_id_col="sense_IDs"
    )
    evaluate_results(eng_sentences, eng_descriptions, eng_gold, eng_deepseek_answers)

    # --- Киргизский: DeepSeek ---
    print("\n=== Киргизский (DeepSeek) ===")
    kyr_sentences, kyr_descriptions, kyr_gold, kyr_deepseek_answers = load_language_data(
        KYR_GOLD_CSV, KYR_DEEPSEEK_JSON, gold_col_split=False, sense_id_col="senseIDs"
    )
    evaluate_results(
        kyr_sentences, kyr_descriptions, kyr_gold, kyr_deepseek_answers, use_glosses=True
    )


if __name__ == "__main__":
    main()
