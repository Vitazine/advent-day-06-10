# Day 7 — фраза → запись расхода

**Обучение v3 завершено.** Выбран iteration-4 по valid (113/137). На фиксированном final-eval база получила **61/140 (43,6%)**, адаптер — **105/140 (75%)**: 47 улучшений и 3 регрессии. Это результат на синтетике, без независимой человеческой оценки и тестирования на телефоне.

- [Итоговый отчёт и все 140 пар ответов](training-v3/final-report.md)
- [Итоговые метрики](training-v3/final-summary.json)
- [Выбор до открытия final](training-v3/selection.json)
- [Контракт и исторические заметки о подготовке датасета](dataset-notes.md)
- [Ограничения мобильного применения](MOBILE_REVIEW.md)

Train/valid/eval: 420/137/140. Данные полностью синтетические. Веса и адаптеры не публикуются. Упоминания «обучение не запускалось» в dataset-notes.md и старых материалах относятся к моменту подготовки корпуса 1 ноября, до запуска обучения.

## Проверка без модели

Из корня репозитория, Python 3.12+:

```sh
python3 -m unittest discover -s experiments/day-07 -p 'test_*.py'
python3 experiments/day-07/score_candidates.py --references experiments/day-07/data/eval_provenance.jsonl --predictions experiments/day-07/training-v3/evaluation/iteration-4-eval.jsonl --baseline experiments/day-07/training-v3/evaluation/base-eval.jsonl
```

Проверка датасета `verify_dataset.py` перезаписывает validation_report.json; запускайте её в рабочей копии. Без `--tokenizer` MLX не нужен. Для генерации v3 включён минимальный неизменный набор файлов v2, который читает build_dataset.py; архив v1 и дубли остальных данных исключены. Генератор отказывается перезаписывать зафиксированный корпус.

## Повторное обучение

Нужны Apple Silicon/Metal, Python 3.12 и отдельно полученная модель `mlx-community/Qwen3-4B-Instruct-2507-4bit`. Укажите её каталог через `MODEL_PATH`. Измеренные версии: MLX 0.32.4, MLX-LM 0.32.2, Transformers 5.22.0. Публичный пакет проверяет сохранённые результаты, но не содержит готового адаптера. Существующий training-v3 — архив результатов: повторные запуски выполняйте в отдельной копии после сохранения архива.

Сведения об отборе файлов и хешах: [PUBLICATION.md](../../PUBLICATION.md).
