import subprocess,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base=ROOT/'short-prompt/benchmark'
for index,variant in enumerate(['long','short','short','long'],1):
 name=f'{index}-{variant}'
 (base/'status.json').write_text(json.dumps({'stage':name,'state':'running'}))
 with (base/f'{name}.log').open('x') as f:
  subprocess.run([sys.executable,str(ROOT/'benchmark_runner.py'),variant,'train','--output',str(base/name),'--resume-adapter-file',str(ROOT.parent/'day-07/training-v3/runs/iteration-4/adapter/adapters.safetensors')],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name+' complete',flush=True)
(base/'status.json').write_text(json.dumps({'state':'complete'}))
