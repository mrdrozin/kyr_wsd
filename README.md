# Разрешение лексической многозначности для киргизского языка

Код и данные к выпускной квалификационной работе по Word Sense Disambiguation (WSD)
для киргизского языка. Работа состоит из двух частей:

1. **Инструмент переноса меток.** Существительные в киргизских предложениях
   выравниваются с английскими/русскими через лексическую базу данных PanLex,
   а смысл выбирается большой языковой моделью DeepSeek-R1. Так строятся
   размеченные данные без ручной аннотации.
2. **Supervised-модели.** На автоматически построенном корпусе обучаются
   и сравниваются энкодер-модели (KyrgyzBERT, mBERT, Kaz-RoBERTa, XLM-R)
   в архитектуре gloss-selection, в режимах полного дообучения и LoRA.

Тестовая выборка размечена вручную на основе Конституции Кыргызской Республики.

## Структура репозитория

```
main_scripts/        основной пайплайн (исполняемые .py)
  ├─ eng_pipeline.py              перенос меток (EN) + DeepSeek WSD
  ├─ ru_pipeline.py               то же для RU
  ├─ wsd_results_evaluation.py    сведение результатов DeepSeek по языкам
  ├─ build_train_set.py           построение обучающего корпуса из SemCor
  ├─ build_train_set_extended.py  расширенная версия (со случайными дистракторами)
  └─ prepare_splits.py            разбиение train / dev / test
main_notebooks/      те же пайплайны в формате .ipynb

auxiliary_scripts/   вспомогательные скрипты построения киргизских данных
  ├─ get_kyr_glosses.py           перевод определений (глосс) на киргизский
  ├─ get_kyr_contexts.py          перевод контекстов с маркерами [TGT]
  ├─ collect_missing_glosses.py   добор недостающих определений
  ├─ prepare_validation.py        сборка тестовой выборки по Конституции
  └─ kyr_wsd_deepseek.py          разрешение многозначности (KY) через DeepSeek-R1
auxiliary_notebooks/ те же скрипты в формате .ipynb

experiments/         обучение и оценка энкодер-моделей
  ├─ model.py                     модель gloss-selection (BERT-энкодер)
  ├─ train.py                     обучение
  ├─ evaluate.py                  оценка обученной модели
  ├─ eval_deepseek_f1.py          замер DeepSeek + бейзлайны (random, MFS)
  ├─ configs/                     гиперпараметры по моделям (YAML)
  ├─ colab/                       лаунчеры для Google Colab
  ├─ plotting/                    скрипты построения графиков
  ├─ figures/                     итоговые графики работы
  └─ runs/                        логи обучения (loss-кривые, метрики) по сидам

gold_data/           ручная разметка (EN / RU / KY), CSV
deepseek_answers/    ответы DeepSeek-R1 по трём языкам, JSON
auxiliary_data/      словарь глосс и глоссарий (лёгкие артефакты)
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/requirements-colab.txt
cp .env.example .env        # вставьте свой ключ DeepSeek API
```

## Данные

Лёгкие артефакты (ручная разметка, ответы DeepSeek, словарь глосс) лежат в
репозитории. Крупные файлы вынесены во внешние хранилища:

- **Обучающие корпуса и тяжёлые артефакты** (`experiments/data/*`,
  `kyr_wsd_dataset/`, исходные контексты Конституции, словарь Удахина и др.) —
  Google Drive: **https://drive.google.com/file/d/12PoQ6Ro2Qkg0KcKe44E1EFcM55afx0_6/view?usp=sharing**. Скачайте и разложите по путям,
  указанным в архиве.

Обучающие выборки также можно воссоздать из SemCor скриптами
`main_scripts/build_train_set.py` и `build_train_set_extended.py`.

## Воспроизведение

Построение данных (требует ключ DeepSeek и данные из внешних хранилищ):

```bash
python main_scripts/eng_pipeline.py          # разметка EN через DeepSeek
python main_scripts/ru_pipeline.py           # разметка RU
python auxiliary_scripts/kyr_wsd_deepseek.py # разметка KY
python main_scripts/build_train_set.py       # обучающий корпус из SemCor
python main_scripts/prepare_splits.py        # train / dev / test
```

Обучение и оценка:

```bash
python experiments/train.py --config experiments/configs/xlmr.yaml
python experiments/evaluate.py
python experiments/eval_deepseek_f1.py       # DeepSeek + бейзлайны
python experiments/plotting/plot_results.py  # графики
```

> Примечание. Ключ DeepSeek читается из `.env` (`DEEPSEEK=...`).
> API-эндпойнт — OpenAI-совместимый, `https://api.deepseek.com`.
> WSD-разметка использует модель `deepseek-reasoner` (DeepSeek-R1),
> перевод корпуса — `deepseek-chat` (DeepSeek-V3).
