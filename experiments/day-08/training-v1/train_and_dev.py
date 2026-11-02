"""Sequential pilot/train/dev only. Final stays locked for reviewed selection."""
import json,subprocess,sys,time,shutil
from core import *

def stage(name,argv,timeout=6000):
    put(ROOT/'status.json',{'state':'running','stage':name,'updated_unix':time.time()})
    log=ROOT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
    with log.open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'run.py'),*argv],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def main():
    verify()
    if json.loads((DAY.parent/'day-09/status.json').read_text()).get('state')!='complete':
        raise RuntimeError('Day9 is not complete; refusing GPU workload')
    for split in ('baseline_screen','knowledge_dev','robustness_dev','format_valid','style_valid','general'):
        path=ROOT/'evaluation'/f'base-dev-{split}.jsonl'
        score(split,read(path))
        meta=json.loads(path.with_suffix('.meta.json').read_text())
        assert meta['dataset_manifest_sha256']==verify() and meta['adapter_sha256'] is None
    stage('pilot',['train','--name','pilot','--steps','60'],timeout=600)
    pilot=json.loads((ROOT/'runs/pilot/summary.json').read_text())
    metrics=read(ROOT/'runs/pilot/metrics.jsonl')
    measured=[r for r in metrics if r['kind']=='train' and r['iteration']>10]
    seconds_per_micro=sum(10/r['iterations_per_second'] for r in measured)/50
    validation_overhead=sum(r['val_time'] for r in metrics if r['kind']=='valid')/2
    forecast=704*seconds_per_micro+5*validation_overhead+30
    steps=704 if forecast<6000 else 352
    if steps*seconds_per_micro+3*validation_overhead+30>=6000:
        raise RuntimeError('Even one epoch exceeds training budget; inspect pilot')
    put(ROOT/'pilot_decision.json',{'seconds_per_microstep_after_warmup':seconds_per_micro,
        'forecast_two_epochs_seconds':forecast,'chosen_steps':steps,'chosen_epochs':steps/352,
        'pilot_peak_memory_gb':pilot['peak_memory_gb'],'pilot_seconds':pilot['seconds'],
        'reset_to_base':True,'note':'Estimate includes validation; no pilot adapter is resumed.'})
    stage('main',['train','--name','main','--steps',str(steps)])
    evaluation_started=time.monotonic()
    def remaining_eval():
        remaining=3600-(time.monotonic()-evaluation_started)
        if remaining<60:raise TimeoutError('Dev evaluation budget exhausted; trained weights preserved, final locked')
        return remaining
    for n in range(176,steps+1,176):
        name=f'checkpoint-{n}';adapter=ROOT/'checkpoints'/name;adapter.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(ROOT/f'runs/main/adapter/{n:07d}_adapters.safetensors',adapter/'adapters.safetensors')
        shutil.copyfile(ROOT/'runs/main/adapter/adapter_config.json',adapter/'adapter_config.json')
        stage(name,['infer','--name',name,'--adapter',str(adapter),'--splits','knowledge_dev,robustness_dev,format_valid,style_valid,valid'],timeout=remaining_eval())
    stage('prompt-dev',['infer','--name','prompt-dev','--mode','prompt','--splits','knowledge_dev,format_valid,style_valid'],timeout=remaining_eval())
    put(ROOT/'status.json',{'state':'awaiting_review','stage':'dev_selection','updated_unix':time.time(),
        'steps':steps,'note':'Final is locked. Review semantic dev answers and freeze selection.'})

if __name__=='__main__':
    try:main()
    except Exception as e:
        put(ROOT/'status.json',{'state':'failed','error':repr(e),'updated_unix':time.time()});raise
