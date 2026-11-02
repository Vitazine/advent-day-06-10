"""Run the frozen final comparisons sequentially; never change training data or weights."""
import subprocess,sys,time
from core import *

def main():
    selection=__import__('json').loads((ROOT/'selection.json').read_text())
    adapter=ROOT/'checkpoints/checkpoint-704'
    guard_final('final',selection,sha(adapter/'adapters.safetensors'),verify())
    common='knowledge_final,recall_final,final,robustness_final,format_final,style_final,brevity_final'
    jobs=[('tuned-final',common+',general',['--adapter',str(adapter)]),
          ('base-final',common,[]),
          ('prompt-final','knowledge_final,format_final,style_final',['--mode','prompt']),
          ('oracle-final','knowledge_final',['--mode','oracle'])]
    for label,splits,args in jobs:
        put(ROOT/'status.json',{'state':'running','stage':label,'updated_unix':time.time(),'selection_sha256':sha(ROOT/'selection.json')})
        with (ROOT/'logs'/f'{label}.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'run.py'),'infer','--name',label,'--splits',splits,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    put(ROOT/'status.json',{'state':'awaiting_review','stage':'final_analysis','updated_unix':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        put(ROOT/'status.json',{'state':'failed','stage':'final_evaluation','error':repr(e),'updated_unix':time.time()})
        raise
