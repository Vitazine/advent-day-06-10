"""Sequential four-run experiment; fails closed and records progress atomically."""
import json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from training_common import *
from score_candidates import score,predictions

def now():return datetime.now(timezone.utc).isoformat()
def state(stage,status='running',**extra):
    write(RUN/'status.json',dict(status=status,stage=stage,updated_utc=now(),pid=os.getpid(),**extra))
    with (RUN/'events.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,status=status,time=now(),**extra))+'\n')
def stage_log_name(args):return '-'.join(args).replace('--','')
def stage(args):
    label=stage_log_name(args);state(' '.join(args))
    with (RUN/'logs'/f'{label}.log').open('x') as log:
        subprocess.run([sys.executable,str(ROOT/'train_v3.py'),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
def evaluate(name,split='valid'):
    result=score(read(ROOT/f'data/{split}_provenance.jsonl'),predictions(RUN/f'evaluation/{name}-{split}.jsonl'))
    write(RUN/f'evaluation/{name}-{split}-scores.json',result);return result
def main():
    verify_data();RUN.mkdir(exist_ok=False);(RUN/'logs').mkdir()
    state('preflight')
    try:
        launch=dict(started_utc=now(),pid=os.getpid(),manifest_sha256=sha(ROOT/'data/manifest.json'),base_weights_sha256=sha(MODEL/'model.safetensors'),base_config_sha256=sha(MODEL/'config.json'),python=sys.executable,epoch_rows=420,main_runs=4,selection='valid exact match, fewer false expenses, fewer regressions, simpler projections, lower LR; stable first on tie',sources={n:sha(ROOT/n) for n in ('pipeline_v3.py','train_v3.py','training_common.py','training_loss.py','score_candidates.py')})
        write(RUN/'launch.json',launch)
        stage(['train','--name','pilot-long','--pilot','--wide'])
        stage(['infer','--name','base-plain','--split','valid']);evaluate('base-plain')
        stage(['infer','--name','base','--split','valid']);base=evaluate('base')
        configs={};scores={}
        def run(name,lr,wide=False,seed=42):
            configs[name]=dict(lr=lr,wide=wide,seed=seed)
            args=['train','--name',name,'--lr',str(lr),'--seed',str(seed)]
            if wide:args.append('--wide')
            stage(args);stage(['infer','--name',name,'--split','valid']);scores[name]=evaluate(name)
            write(RUN/'dev-results.json',dict(configs=configs,scores=scores))
        def best(names):return max(names,key=lambda name:selection_key(scores[name],base['outcomes'],configs[name]['wide'],configs[name]['lr']))
        run('iteration-1',5e-5);run('iteration-2',1e-4)
        lr=configs[best(['iteration-1','iteration-2'])]['lr'];run('iteration-3',lr,True)
        repeated=best(list(scores));c=configs[repeated];run('iteration-4',c['lr'],c['wide'],43)
        selected=best(list(scores))
        selection=dict(selected=selected,selected_utc=now(),repeated_configuration=repeated,adapter_sha256=sha(RUN/f'runs/{selected}/adapter/adapters.safetensors'),eval_sha256=sha(ROOT/'data/eval.jsonl'),valid_results_sha256=sha(RUN/'dev-results.json'),final_outputs_exist=False)
        assert not list((RUN/'evaluation').glob('*-eval.jsonl'))
        write(RUN/'selection.json',selection)
        stage(['infer','--name','base','--split','eval']);b=evaluate('base','eval')
        stage(['infer','--name',selected,'--split','eval']);t=evaluate(selected,'eval')
        wins=[i for i,v in t['outcomes'].items() if v and not b['outcomes'][i]]
        losses=[i for i,v in t['outcomes'].items() if not v and b['outcomes'][i]]
        assert sha(MODEL/'model.safetensors')==launch['base_weights_sha256'];verify_data()
        result=dict(selected=selected,base_exact=b['exact_correct'],tuned_exact=t['exact_correct'],n=b['n'],improved=wins,regressed=losses,base_scores=b,tuned_scores=t,training_seconds=sum(json.loads((RUN/f'runs/{name}/summary.json').read_text())['seconds'] for name in configs),model_weights_unchanged=True)
        write(RUN/'final-summary.json',result)
        report=['# Day 7 — обучение v3 по схеме Day 6','',f"Выбран {selected} по valid. Final: база {b['exact_correct']}/{b['n']}, адаптер {t['exact_correct']}/{t['n']}. Улучшений {len(wins)}, ухудшений {len(losses)}.",'','| Запуск | LR | Проекции | Seed | Valid exact | Время, мин |','|---|---:|---|---:|---:|---:|']
        for name,c in configs.items():
            duration=json.loads((RUN/f'runs/{name}/summary.json').read_text())['seconds']/60
            report.append(f"| {name} | {c['lr']} | {'wide' if c['wide'] else 'q/v'} | {c['seed']} | {scores[name]['exact_correct']}/137 | {duration:.1f} |")
        report+=['','## Ошибки и ограничения',f'ID регрессий: {", ".join(losses) or "нет"}.',f"Ложные расходы на ignored: {b['false_expenses_on_ignored']} → {t['false_expenses_on_ignored']}; на missing_context: {b['false_expenses_on_missing_context']} → {t['false_expenses_on_missing_context']}.",'Одна эпоха: 420 микрошагов/105 optimizer updates; все четыре запуска от базы, одинаковые данные. Синтетический eval, без человеческой оценки или телефона. Улучшение не гарантируется; см. исходные ответы и срезы в evaluation/. Для сравнения качества v3 с v2 отдельного обучения v2 не было.','', '## Все финальные ответы','| ID | Вход | База | Адаптер |','|---|---|---|---|']
        bp=predictions(RUN/'evaluation/base-eval.jsonl');tp=predictions(RUN/f'evaluation/{selected}-eval.jsonl')
        def cell(s):return str(s).replace('|','\\|').replace('\n','<br>')
        for r in read(ROOT/'data/eval_provenance.jsonl'):report.append('| '+' | '.join(cell(v) for v in (r['id'],r['input']['text'],bp[r['id']],tp[r['id']]))+' |')
        (RUN/'final-report.md').write_text('\n'.join(report)+'\n')
        state('complete','complete',selected=selected,base_exact=b['exact_correct'],tuned_exact=t['exact_correct'],n=b['n'])
    except BaseException as exc:
        write(RUN/'failure.json',dict(time=now(),error=repr(exc),traceback=traceback.format_exc()))
        state('failed','failed',error=repr(exc));raise
if __name__=='__main__':main()
