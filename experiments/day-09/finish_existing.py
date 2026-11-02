"""Finish the existing trained checkpoint after user-requested prompt experiments."""
import json,time
from runner_config import ROOT,START,sha,verify_manifest
from pipeline import store,run,summarize,dangers,recommend_after
from evaluate import evaluate
from build_report import main as report

def main():
 verify_manifest();source_hash=sha(START)
 before=summarize('before','valid');after=summarize('iteration-1','valid');refs=ROOT/'data/valid-accepted.jsonl'
 bd=dangers(ROOT/'evaluation/before-valid/responses.jsonl',refs);ad=dangers(ROOT/'evaluation/iteration-1-valid/responses.jsonl',refs)
 eligible=recommend_after(before['exact_correct'],after['exact_correct'],bd,ad)
 candidate={'name':'iteration-1','exact_correct':after['exact_correct'],'eligible':eligible,'dangerous_ids':sorted(ad),'regressions':[i for i,v in before['outcomes'].items() if v and not after['outcomes'][i]],'adapter_sha256':sha(ROOT/'runs/iteration-1/adapter/adapters.safetensors')}
 selection={'selected_after':'iteration-1','improvement_repeated_across_seeds':False,'provisional_recommendation':'after' if eligible else 'before','candidates':[candidate],'frozen_before_final':True,'manifest_sha256':verify_manifest(),'start_adapter_sha256':source_hash,'selected_adapter_sha256':candidate['adapter_sha256'],'created_unix':time.time(),'ruling':'No second long seed: user prioritized shortening time; compressed prompt drop-in candidates failed valid smoke. Single-seed limitation retained. No short-prompt training quality claim.'}
 with (ROOT/'selection.json').open('x') as f:json.dump(selection,f,ensure_ascii=False,indent=2)
 for label,adapter in [('before',START.parent),('after',ROOT/'runs/iteration-1/adapter')]:
  run(label+'-final',['infer','--inputs',str(ROOT/'data/final-accepted-inputs.jsonl'),'--adapter',str(adapter),'--output',str(ROOT/f'evaluation/{label}-final')]);summarize(label,'final')
 teacher=evaluate(ROOT/'data/final-accepted.jsonl',ROOT/'evaluation/teacher-final.jsonl');store(ROOT/'evaluation/teacher-final-scores.json',teacher)
 b=json.loads((ROOT/'evaluation/before-final-scores.json').read_text());a=json.loads((ROOT/'evaluation/after-final-scores.json').read_text());refs=ROOT/'data/final-accepted.jsonl'
 bd=dangers(ROOT/'evaluation/before-final/responses.jsonl',refs);ad=dangers(ROOT/'evaluation/after-final/responses.jsonl',refs)
 safe=eligible and recommend_after(b['exact_correct'],a['exact_correct'],bd,ad)
 store(ROOT/'recommendation.json',{'recommended':'after' if safe else 'before','new_final_false_expense_ids':sorted(ad-bd),'before_final_exact':b['exact_correct'],'after_final_exact':a['exact_correct'],'note':'Single seed. Final gates recommendation, never checkpoint selection or further tuning.'})
 verify_manifest()
 if sha(START)!=source_hash:raise ValueError('Original adapter changed')
 store(ROOT/'status.json',{'state':'complete','stage':'complete','selected':'iteration-1','updated_unix':time.time()});report()
if __name__=='__main__':
 try:main()
 except Exception as e:
  store(ROOT/'status.json',{'state':'failed','error':repr(e),'updated_unix':time.time()});raise
