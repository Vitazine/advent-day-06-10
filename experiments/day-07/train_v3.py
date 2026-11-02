"""Train or infer one stage in a fresh process. Original weights remain read-only."""
import os
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
import argparse,importlib.metadata,json,math,platform,time
from functools import partial
from types import SimpleNamespace
from training_common import *
import mlx.core as mx
import numpy as np
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.lora import CONFIG_DEFAULTS,train_model
import mlx_lm.lora as lora_module
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.callbacks import TrainingCallback
from training_loss import completion_loss
class Metrics(TrainingCallback):
    def __init__(self,path):self.path=path
    def record(self,kind,v):
        assert all(not isinstance(x,float) or math.isfinite(x) for x in v.values())
        with self.path.open('a') as f:f.write(json.dumps(dict(kind=kind,**v))+'\n')
        print(json.dumps(dict(kind=kind,**v)),flush=True)
    def on_train_loss_report(self,v):self.record('train',v)
    def on_val_loss_report(self,v):self.record('valid',v)
def loss_check():
    class Uniform:
        def __call__(self,x):return mx.zeros((*x.shape,16))
    batch=mx.array([[1,2,3,4,5,6,7,8,9,10,0,0],[1,2,3,4,5,6,0,0,0,0,0,0]])
    loss,n=completion_loss(Uniform(),batch,mx.array([[6,10],[3,6]]))
    assert n.item()==7 and abs(loss.item()-2.7725887)<1e-5
    return dict(target_tokens=int(n.item()),loss=float(loss.item()))
def train(a):
    manifest=verify_data();loss_evidence=loss_check();out=RUN/'runs'/a.name
    out.mkdir(parents=True,exist_ok=False)
    config={**CONFIG_DEFAULTS,**configuration(a.name,a.lr,a.seed,a.wide,a.pilot)}
    started=time.perf_counter();np.random.seed(a.seed);mx.random.seed(a.seed)
    model,tokenizer=load(str(MODEL),tokenizer_config={'trust_remote_code':False})
    tokenizer.chat_template=(ROOT/'chat_template.jinja').read_text()
    if a.pilot:
        data=out/'data';data.mkdir()
        rows=read(ROOT/'data/train.jsonl')
        rows.sort(key=lambda r:len(tokenizer.apply_chat_template(r['messages'],tokenize=True)),reverse=True)
        for split,pool in [('train',rows[:16]),('valid',read(ROOT/'data/valid.jsonl')[:4])]:
            (data/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pool))
        config['data']=str(data)
    write(out/'config.json',config)
    write(out/'provenance.json',dict(manifest_sha256=sha(ROOT/'data/manifest.json'),base_weights_sha256=json.loads((RUN/'launch.json').read_text())['base_weights_sha256'],resume=None,nominal_epochs=config['iters']/420 if not a.pilot else None,optimizer_updates=config['iters']//4,device=mx.device_info(),python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers')},loss_check=loss_evidence,training_files_sha256={n:sha(Path(config['data'])/n) for n in ('train.jsonl','valid.jsonl')}))
    for name in ('train_v3.py','training_common.py','training_loss.py'):(out/name).write_bytes((ROOT/name).read_bytes())
    cfg=SimpleNamespace(**config);trainset,validset,_=load_dataset(cfg,tokenizer)
    original=lora_module.train;lora_module.train=partial(original,loss=completion_loss)
    try:train_model(cfg,model,trainset,validset,Metrics(out/'metrics.jsonl'))
    finally:lora_module.train=original
    adapter=out/'adapter/adapters.safetensors';tensors=mx.load(str(adapter))
    assert tensors and all(bool(mx.all(mx.isfinite(t)).item()) for t in tensors.values())
    assert any(bool(mx.any(t!=0).item()) for k,t in tensors.items() if 'lora_b' in k.lower())
    write(out/'summary.json',dict(seconds=time.perf_counter()-started,peak_memory_gb=mx.get_peak_memory()/1e9,steps=config['iters'],optimizer_updates=config['iters']//4,adapter_sha256=sha(adapter),tensor_count=len(tensors),all_finite=True))
def infer(a):
    verify_data()
    if a.split=='eval' and not (RUN/'selection.json').exists():raise RuntimeError('Final locked until selection')
    out=RUN/'evaluation'/f'{a.name}-{a.split}.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    adapter=None if a.name in ('base','base-plain') else RUN/'runs'/a.name/'adapter'
    model,tokenizer=load(str(MODEL),adapter_path=str(adapter) if adapter else None,tokenizer_config={'trust_remote_code':False})
    tokenizer.chat_template=(ROOT/'chat_template.jinja').read_text()
    rows=read(ROOT/f'data/{a.split}_provenance.jsonl');started=time.perf_counter()
    with out.open('x') as f:
        for i,r in enumerate(rows):
            msgs=[dict(role='system',content=(ROOT/'system.txt').read_text().strip()),dict(role='user',content=json.dumps(r['input'],ensure_ascii=False))]
            if a.name=='base-plain':msgs=msgs[1:]
            prompt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            mx.random.seed(42)
            answer=generate(model,tokenizer,prompt=prompt,max_tokens=256,sampler=make_sampler(temp=0.),verbose=False)
            f.write(json.dumps(dict(id=r['id'],response=answer),ensure_ascii=False)+'\n');f.flush()
            print(f'{a.name}/{a.split}: {i+1}/{len(rows)}',flush=True)
    write(out.with_suffix('.meta.json'),dict(rows=len(rows),seconds=time.perf_counter()-started,max_tokens=256,temperature=0,source_sha256=sha(ROOT/f'data/{a.split}_provenance.jsonl'),adapter_sha256=sha(adapter/'adapters.safetensors') if adapter else None))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','infer']);p.add_argument('--name',required=True);p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--seed',type=int,default=42);p.add_argument('--wide',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--split',choices=['valid','eval'],default='valid');a=p.parse_args();{'train':train,'infer':infer}[a.action](a)
