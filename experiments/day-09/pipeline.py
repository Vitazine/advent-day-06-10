"""Sequential Day9 continuation; no final scoring until selection is frozen."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from runner_config import ROOT,START,DAY7,sha,verify_manifest
from evaluate import evaluate
from dataset_tools import read


def store(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)

def run(stage,args):
    store(ROOT/'status.json',{'stage':stage,'state':'running','updated_unix':time.time()})
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    with (logs/f'{stage}.log').open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'runner.py'),*args],stdout=f,stderr=subprocess.STDOUT,check=True)

def summarize(name,split):
    result=evaluate(ROOT/f'data/{split}-accepted.jsonl',ROOT/f'evaluation/{name}-{split}/responses.jsonl')
    store(ROOT/f'evaluation/{name}-{split}-scores.json',result)
    return result

def dangers(pred,refs):
    source={r['id']:r for r in read(refs)};out=set()
    for r in read(pred):
        if r['id'] not in source:continue
        try:a=json.loads(r['response'])
        except (ValueError,TypeError):continue
        if not isinstance(a,dict):continue
        target=source[r['id']]['target']
        if a.get('status')=='ok' and (target['status']=='ignored' or target['clarification_reason']=='missing_context'):out.add(r['id'])
    return out

def recommend_after(before_exact,after_exact,before_dangers,after_dangers):
    return after_exact>before_exact and not (set(after_dangers)-set(before_dangers))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,required=True,choices=[1,2]);a=ap.parse_args()
    verify_manifest();start_hash=sha(START)
    meta=json.loads((ROOT/'evaluation/before-valid/summary.json').read_text())
    expected={'input_sha256':sha(ROOT/'data/valid-inference.jsonl'),'adapter_sha256':start_hash,'system_sha256':sha(ROOT/'system.txt'),'template_sha256':sha(DAY7/'chat_template.jinja'),'temperature':0,'max_tokens':256}
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Baseline provenance mismatch')
    base=summarize('before','valid')
    refs=ROOT/'data/valid-accepted.jsonl'
    base_dangers=dangers(ROOT/'evaluation/before-valid/responses.jsonl',refs)
    candidates=[]
    for index,seed in enumerate([42,43],1):
        name=f'iteration-{index}'
        run(name,['train','--batch-size',str(a.batch_size),'--seed',str(seed),'--resume-adapter-file',str(START),'--output',str(ROOT/f'runs/{name}')])
        run(name+'-valid',['infer','--inputs',str(ROOT/'data/valid-accepted-inputs.jsonl'),'--adapter',str(ROOT/f'runs/{name}/adapter'),'--output',str(ROOT/f'evaluation/{name}-valid')])
        s=summarize(name,'valid');danger=dangers(ROOT/f'evaluation/{name}-valid/responses.jsonl',refs)
        regressions=[i for i,ok in base['outcomes'].items() if ok and not s['outcomes'][i]]
        eligible=not (danger-base_dangers) and s['exact_correct']>base['exact_correct']
        c={'name':name,'exact_correct':s['exact_correct'],'eligible':eligible,'dangerous_ids':sorted(danger),'regressions':regressions,'adapter_sha256':sha(ROOT/f'runs/{name}/adapter/adapters.safetensors')}
        candidates.append(c);store(ROOT/'candidates.json',candidates)
        if index==1 and not eligible:break
    eligible=[c for c in candidates if c['eligible']]
    # Even if training fails to improve, evaluate actual trained weights honestly.
    chosen=max(eligible or candidates,key=lambda c:(c['exact_correct'],-len(c['dangerous_ids']),-len(c['regressions'])))
    selection={'improvement_repeated_across_seeds':len(eligible)>=2,'selected_after':chosen['name'],'provisional_recommendation':'after' if eligible else 'before','candidates':candidates,'frozen_before_final':True,'manifest_sha256':verify_manifest(),'start_adapter_sha256':start_hash,'selected_adapter_sha256':chosen['adapter_sha256'],'created_unix':time.time()}
    with (ROOT/'selection.json').open('x') as f:json.dump(selection,f,ensure_ascii=False,indent=2)
    for label,adapter in [('before',START.parent),('after',ROOT/f"runs/{chosen['name']}/adapter")]:
        run(label+'-final',['infer','--inputs',str(ROOT/'data/final-accepted-inputs.jsonl'),'--adapter',str(adapter),'--output',str(ROOT/f'evaluation/{label}-final')])
        summarize(label,'final')
    teacher=evaluate(ROOT/'data/final-accepted.jsonl',ROOT/'evaluation/teacher-final.jsonl');store(ROOT/'evaluation/teacher-final-scores.json',teacher)
    before=json.loads((ROOT/'evaluation/before-final-scores.json').read_text())
    after=json.loads((ROOT/'evaluation/after-final-scores.json').read_text())
    finalrefs=ROOT/'data/final-accepted.jsonl'
    bd=dangers(ROOT/'evaluation/before-final/responses.jsonl',finalrefs)
    ad=dangers(ROOT/'evaluation/after-final/responses.jsonl',finalrefs)
    safe=bool(eligible) and recommend_after(before['exact_correct'],after['exact_correct'],bd,ad)
    store(ROOT/'recommendation.json',{'recommended':'after' if safe else 'before','new_final_false_expense_ids':sorted(ad-bd),'before_final_exact':before['exact_correct'],'after_final_exact':after['exact_correct'],'note':'Final affects deployment recommendation only, never checkpoint selection or retraining.'})
    verify_manifest()
    if sha(START)!=start_hash:raise ValueError('Original adapter changed')
    store(ROOT/'status.json',{'stage':'complete','state':'complete','updated_unix':time.time(),'selected':chosen['name']})

if __name__=='__main__':
    try:main()
    except Exception as e:
        store(ROOT/'status.json',{'state':'failed','error':repr(e),'updated_unix':time.time()});raise
