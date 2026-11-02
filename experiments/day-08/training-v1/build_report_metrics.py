import sys,json,time
from pathlib import Path
sys.path.insert(0,'experiments/day-08/training-v1')
from core import *
E=ROOT/'evaluation'
# Human-readable answers were inspected in this session. These are manual AI annotations,
# not substring matching, and not an independent human evaluation.
correct={'tuned':[5,26,27,33,35,40,42,48,56,66,69,75,78],
         'base':[], 'prompt':[5],
         'oracle':[4,5,9,11,12,13,28,29,32,33,35,39,48,59,60,62,64,65,67,68,69,75,79,83,84]}
ambiguous={'tuned':[38],'oracle':[42]}
review=[]
refs=references('knowledge_final');qs={r['id']:r['messages'][-1]['content'] for r in rows_for('knowledge_final')}
for mode,indices in correct.items():
 for i,r in enumerate(read(E/f'{mode}-final-knowledge_final.jsonl')):
  verdict='correct' if i in indices else ('ambiguous' if i in ambiguous.get(mode,[]) else ('truncated' if r['finish_reason']=='length' else 'wrong'))
  reason={'correct':'Полный ответ по смыслу вопроса; допустимы краткая формулировка и склонение.',
          'wrong':'Неверная сущность/отношение/дата, неполнота или противоречащие эталону дополнения.',
          'ambiguous':'Основной факт верен, но дополнительное утверждение требует отдельной проверки; не включено в успехи.',
          'truncated':'Достигнут лимит 128 токенов; не засчитывается как полный корректный ответ. Отдельный статус, не доказательство отсутствия знания.'}[verdict]
  review.append({'mode':mode,'id':r['id'],'question':qs[r['id']],'target':refs[r['id']]['target'],**r,'verdict':verdict,'reason':reason})
put(ROOT/'final_knowledge_review.json',{'reviewer':'Codex AI; no independent human validation','metric':'Confirmed complete correct responses; truncated and ambiguous separate, excluded from successes','rows':review})
# Four criteria: neutral tone, no filler, <=2 sentences/no introduction, preserved facts/no additions.
style_bad={0:([0,1,1,0],'Ножны заменены ножами; разговорные уменьшительные.'),1:([1,0,1,1],'Лишний метакомментарий о соответствии даты заметке.'),5:([0,1,1,1],'Оставлены разговорные уменьшительные цветочки/листочки.'),7:([1,1,1,0],'Добавлена Вторая эпоха.'),8:([1,1,1,0],'Добавлены выдуманные эпоха и год.'),11:([1,1,1,0],'Потерян факт: Бергил — сын Берегонда.'),15:([1,1,1,0],'Добавлена Вторая эпоха.')}
key=json.loads((ROOT/'style_blind_key.json').read_text());sr=[]
for i,r in enumerate(key):
 criteria,reason=style_bad.get(i,([1,1,1,1],'Факты исходной заметки сохранены, нейтрально, кратко, без новых утверждений.'))
 sr.append({**r,'blind_index':i,'criteria':dict(zip(['neutral','no_filler','brief','faithful'],criteria)),'pass':all(criteria),'reason':reason})
put(ROOT/'final_style_review.json',{'method':'Modes shuffled with seed134; criteria assigned before revealing mode mapping; same AI reviewer, not independent double blind review','rows':sr})
# Preserve every base-pass/tuned-fail, including punctuation-only changes.
b=read(E/'base-dev-general.jsonl');t=read(E/'tuned-final-general.jsonl');bs=score('general',b);ts=score('general',t)
refs=references('general');qs={r['id']:r['messages'][-1]['content'] for r in rows_for('general')}
reg=[]
for x,y in zip(b,t):
 if bs['outcomes'][x['id']] and not ts['outcomes'][y['id']]:
  reg.append({'id':x['id'],'question':qs[x['id']],'target':refs[x['id']]['target'],'base':x['answer'],'tuned':y['answer'],'kind':'content_error' if x['id']=='general-014' else 'punctuation_or_case'})
put(ROOT/'general_regressions.json',{'base':bs['normalized_exact'],'tuned':ts['normalized_exact'],'regressions':reg,'improvements':[x['id'] for x in t if ts['outcomes'][x['id']] and not bs['outcomes'][x['id']]]})
# Schema denominator always comes from references, including invalid JSON outputs.
formats={}
for mode in ['base','prompt','tuned']:
 records=read(E/f'{mode}-final-format_final.jsonl');s=score('format_final',records)
 formats[mode]={'schema_valid':s['schema_valid'],'denominator':len(references('format_final')),'exact_content':s['normalized_exact']}
put(ROOT/'final_format_metrics.json',formats)
# Validate every prediction/meta pair before reporting.
checked=0
for f in E.glob('*.jsonl'):
 label,split=None,None
 candidates=['baseline_screen','knowledge_dev','robustness_dev','format_valid','style_valid','valid','general','knowledge_final','recall_final','robustness_final','format_final','style_final','brevity_final','final']
 for s in sorted(candidates,key=len,reverse=True):
  if f.stem.endswith('-'+s):split=s;break
 if split is None:raise ValueError(f)
 meta=json.loads(f.with_suffix('.meta.json').read_text());rows=read(f)
 assert meta['dataset_manifest_sha256']==verify()
 assert meta['input_sha256']==sha(DAY/'data'/f'{split}.jsonl')
 assert len(rows)==meta['rows'];score(split,rows);checked+=1
seconds=sum(json.loads(f.read_text())['total_seconds'] for f in E.glob('*-timing.json'))
seconds+=sum(json.loads((ROOT/f'runs/{s}/summary.json').read_text())['seconds'] for s in ['pilot','main'])
summary={'checked_prediction_files':checked,'measured_process_seconds_total':seconds,
         'knowledge_confirmed_complete':{m:len(v) for m,v in correct.items()},'knowledge_denominator':86,
         'style_all_criteria':{m:sum(r['pass'] for r in sr if r['mode']==m) for m in ['base','prompt','tuned']},
         'format':formats,'general':{'base':32,'tuned':23,'regressions':10,'content_regressions':1,'format_regressions':9,'improvements':1},
         'review_limits':'AI judgments only; ambiguous excluded; truncated incomplete; no new tuning after final.'}
put(ROOT/'final_metrics.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
