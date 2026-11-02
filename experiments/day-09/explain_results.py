"""Append observed cases and limits to the completed final report."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'day-07'))
from contract import parse_model_response

def read(path):return [json.loads(s) for s in path.read_text().splitlines()]
def clean(s):return str(s).replace('|','\\|').replace('\n',' ')
def main():
 if json.loads((ROOT/'status.json').read_text())['state']!='complete':raise RuntimeError('Final not complete')
 refs=read(ROOT/'data/final-accepted.jsonl')
 models={k:{r['id']:r['response'] for r in read(ROOT/filename)} for k,filename in [('before','evaluation/before-final/responses.jsonl'),('after','evaluation/after-final/responses.jsonl'),('teacher','evaluation/teacher-final.jsonl')]}
 records=[]
 for r in refs:
  parsed={};invalid={}
  for k,pred in models.items():
   try:parsed[k]=parse_model_response(pred[r['id']],r['input']['current_date'])
   except (ValueError,TypeError) as e:parsed[k]=None;invalid[k]=str(e)
  records.append({'id':r['id'],'text':r['input']['text'],'current_date':r['input']['current_date'],'target':r['target'],'parsed':parsed,'invalid':invalid,'correct':{k:v==r['target'] for k,v in parsed.items()}})
 fixes=[r for r in records if r['correct']['after'] and not r['correct']['before']]
 regressions=[r for r in records if r['correct']['before'] and not r['correct']['after']]
 remaining=[r for r in records if not r['correct']['after']]
 lines=['## Где ученик справляется, а где нужен контроль','',
 'Ни один небольшой синтетический тест не доказывает безопасную автономную замену учителя. Ниже — наблюдаемое поведение на зафиксированных примерах, а не гарантия для реальных расходов.','',
 '| Тип входа | После обучения | Учитель |','|---|---:|---:|']
 for status in ('ok','needs_clarification','ignored'):
  group=[r for r in records if r['target']['status']==status]
  lines.append(f"| {status} | {sum(r['correct']['after'] for r in group)}/{len(group)} | {sum(r['correct']['teacher'] for r in group)}/{len(group)} |")
 for title,group in [('Исправленные ошибки',fixes),('Регрессии',regressions),('Оставшиеся ошибки',remaining)]:
  lines+=['',f'### {title}','']
  if not group:lines+=['На этом наборе не обнаружены.'];continue
  lines+=['| ID и вход | Ожидалось | Получено после |','|---|---|---|']
  for r in group[:8]:
   out=r['parsed']['after'];expected=r['target']
   if out is None:diff='Нарушение контракта: '+r['invalid']['after'];exp=json.dumps(expected,ensure_ascii=False)
   elif r['correct']['after']:
    exp=json.dumps(expected,ensure_ascii=False);diff='Все поля совпали с эталоном'
   else:
    keys=[k for k,v in expected.items() if out[k]!=v]
    exp=json.dumps({k:expected[k] for k in keys},ensure_ascii=False)
    diff=json.dumps({k:out[k] for k in keys},ensure_ascii=False)
   lines.append('| '+clean(r['id']+': '+r['text']+' (дата отсчёта '+r['current_date']+')')+' | '+clean(exp)+' | '+clean(diff)+' |')
 lines+=['','Автоматическую запись расходов следует оценивать отдельно от режима подсказки с подтверждением: ложный расход важнее небольшой прибавки общего exact match. Денежная экономия не измерена; время субагента включает инструменты и не сопоставимо напрямую с локальной задержкой.','',
 'Проверка коротких промптов: [отдельный отчёт](short-prompt/report.md). Микропилот английского варианта показал −36% времени обучения/сохранения, но прямая замена инструкции ухудшила качество. Полноценного обучения с коротким промптом не было.']
 report=ROOT/'final-report.md';text=report.read_text();marker='## Где ученик справляется, а где нужен контроль'
 if marker in text:raise FileExistsError('Qualitative analysis already appended')
 report.write_text(text+'\n'+'\n'.join(lines)+'\n')
 (ROOT/'final-cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 print(json.dumps({'fixes':len(fixes),'regressions':len(regressions),'remaining_errors':len(remaining)}))
if __name__=='__main__':main()
