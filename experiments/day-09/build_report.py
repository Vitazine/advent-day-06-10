"""Build a factual report only after the complete pipeline has produced final scores."""
import json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
def obj(path):return json.loads(path.read_text())
def timing(label):
    rows=read(ROOT/f'evaluation/{label}-final/responses.jsonl');v=sorted(r['seconds'] for r in rows)
    return f'{statistics.median(v):.2f} / {v[math.ceil(.95*len(v))-1]:.2f}'

def main():
    status=obj(ROOT/'status.json')
    if status['state']!='complete':raise RuntimeError('Pipeline is not complete')
    selected=obj(ROOT/'selection.json');rec=obj(ROOT/'recommendation.json')
    scores={k:obj(ROOT/f'evaluation/{k}-final-scores.json') for k in ('before','after','teacher')}
    before,after=scores['before'],scores['after'];n=before['n']
    wins=[i for i,v in after['outcomes'].items() if v and not before['outcomes'][i]]
    losses=[i for i,v in before['outcomes'].items() if v and not after['outcomes'][i]]
    lines=['# Day 9 — результаты дистилляции через субагента','',
      f"Рекомендация по этому эксперименту: **{'предпочесть дообученный адаптер для режима с подтверждением записи' if rec['recommended']=='after' else 'сохранить исходный адаптер Day7'}**.",
      f"Выбранный по valid запуск: {selected['selected_after']}. Подтверждение повторным seed: {'да' if selected['improvement_repeated_across_seeds'] else 'нет; выполнен один seed'}.",'',
      '| Метрика final | До (Day7) | После | Учитель |','|---|---:|---:|---:|']
    for label,key in [('Точные JSON','exact_correct'),('Невалидные ответы','invalid_responses'),('Ложные расходы на ignored','false_expenses_on_ignored'),('Ложные расходы при missing_context','false_expenses_on_missing_context')]:
      lines.append('| '+label+' | '+' | '.join(str(scores[k][key])+(f'/{n}' if key=='exact_correct' else '') for k in ('before','after','teacher'))+' |')
    lines+=['',f'Улучшений: {len(wins)}; регрессий: {len(losses)}. Новые опасные final-ошибки: '+str(rec['new_final_false_expense_ids'])+'.','',
      '## Скорость и затраты','',f'Локальная задержка p50 / p95, секунд: до **{timing("before")}**, после **{timing("after")}**. Измерения последовательные на том же устройстве; фоновые условия не полностью контролировались. Cold load сохранён отдельно в summary.json.','']
    for c in selected['candidates']:
      s=obj(ROOT/f"runs/{c['name']}/summary.json")
      lines.append(f"- {c['name']}: {s['wall_seconds']/60:.1f} мин, пик MLX {s['peak_memory_gb']:.2f} ГБ, {s['optimizer_updates']} updates; valid {c['exact_correct']}/63.")
    lines+=['','У учителя измерялось время работы субагента с инструментами, а не latency отдельного API-запроса. Долларовая стоимость и расход лимитов недоступны; процент денежной экономии не вычислялся. Использование субагента не считается бесплатным.','',
      '## Данные и ограничения','',
      '396 train,63 valid,79 final. Train: новые240 + replay180 до фильтрации, все ответы учителя сформированы заново. Независимые AI-аудиторы исключили спорные примеры и семьи; человеческого аудита не было. Все данные синтетические. Точный backend model ID субагентов не предоставлен, raw ответы и инструкции сохранены.',
      'Checkpoint выбран до раскрытия final. Final влияет только на рекомендацию применения, не на выбор весов или повторное обучение. Старый Day7 eval не переоценивался: новое сравнение использует независимые входы Day9.',
      'Пилоты batch1/2 проверяли производительность, их loss не использован для выбора качества. Последний встроенный val loss MLX измеряется до последнего update; оценки в таблице получены генерацией сохранённых весов.','',
      '## Регрессии и ошибки','',f'ID регрессий: {", ".join(losses) or "нет"}.',f'ID ошибок после: {", ".join(i for i,v in after["outcomes"].items() if not v) or "нет"}.',
      'Все исходные ответы: evaluation/before-final/responses.jsonl, evaluation/after-final/responses.jsonl, evaluation/teacher-final.jsonl. Метрики по полям/категориям/статусам — *-final-scores.json.']
    (ROOT/'final-report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'final-summary.json').write_text(json.dumps({'scores':scores,'improved':wins,'regressed':losses,'selection':selected,'recommendation':rec},ensure_ascii=False,indent=2))
    readme=ROOT/'README.md'
    if readme.exists():
        import re
        readme.write_text(re.sub(r'^Статус:.*$', 'Статус: обучение и итоговое сравнение завершены. Результаты — final-report.md; рекомендация — recommendation.json.', readme.read_text(), count=1, flags=re.MULTILINE))
    print(ROOT/'final-report.md')
if __name__=='__main__':main()
