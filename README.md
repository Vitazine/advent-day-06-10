# Advent Challenge — Day 06–09

Эксперименты с QLoRA и Qwen3-4B-Instruct-2507-4bit на Apple Silicon.

| Эксперимент | База → адаптер | Ограничения |
|---|---|---|
| [Day 06: нейтральное перефразирование](experiments/day-06/README.md) | 26/40 → 33/40 | Четыре регрессии; синтетический final, AI-разметка |
| [Day 07: фраза → запись расхода](experiments/day-07/README.md) | 61/140 → 105/140 | Три регрессии; синтетический eval, без проверки на телефоне |

Включены исходный код, параметры четырёх основных прогонов каждого дня, ответы моделей, оценки, ошибки и синтетические данные day-07. Веса и адаптеры нужно получать/обучать отдельно. Train/valid стороннего ru_paradetox не перепубликуются.

[Что включено и исключено, проверки и ограничения воспроизведения](PUBLICATION.md).

## Day 08 и Day 09

- [Day 08 — границы дообучения на Средиземье](experiments/day-08/README.md): знания 0/86 → 13/86, JSON-схема 2/17 → 17/17; общие задачи 32/40 → 23/40. Oracle со справкой — 25/86. Надёжный справочник не получился.
- [Day 09 — дистилляция расходов](experiments/day-09/README.md): 54/79 → 60/79, шесть улучшений, без exact-match регрессий; остаются четыре невалидных ответа и четыре ложных расхода. Один seed, синтетический final.
- [Проверки публикации day-08/09](publication-day-08-09-verification.json), [контрольные суммы](publication-day-08-09-manifest.json).

Неудачные проверки короткого промпта сохранены. Веса и ZIP с адаптерами не включены. Day 10 пока не публикуется.

## Проверки без модели

Python 3.12+, из корня репозитория:

```sh
python3 -m unittest discover -s experiments/day-06 -p 'test_*.py'
python3 -m unittest discover -s experiments/day-07 -p 'test_*.py'
python3 experiments/day-07/score_candidates.py --references experiments/day-07/data/eval_provenance.jsonl --predictions experiments/day-07/training-v3/evaluation/iteration-4-eval.jsonl --baseline experiments/day-07/training-v3/evaluation/base-eval.jsonl
```

Один архивный тест v1 намеренно пропущен: эта версия не входит в публикацию. GPU-обучение при подготовке публикации не повторялось. Контрольные суммы публичных файлов находятся в publication-manifest.json; исторические хеши весов и локальных исходников описаны отдельно в PUBLICATION.md.

