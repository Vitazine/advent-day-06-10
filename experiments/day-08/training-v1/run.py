"""Offline Day8 runner: separate process per training/inference stage."""
import argparse,importlib.metadata,json,os,platform,shutil,sys,time
from functools import partial
from types import SimpleNamespace
from core import *
os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')

def load_runtime():
    import mlx.core as mx
    from mlx_lm import load
    sys.path.insert(0,str(DAY.parent/'day-06'))
    from loss import completion_loss
    return mx,load,completion_loss

def train(name,steps):
    verify();out=ROOT/'runs'/name;out.mkdir(parents=True,exist_ok=False)
    mx,load,loss=load_runtime()
    import mlx_lm.lora as lm
    import numpy as np
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.datasets import load_dataset
    class Metrics(TrainingCallback):
        def record(self,kind,values):
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'kind':kind,**values,'wall_elapsed':time.perf_counter()-started})+'\n')
            print(kind,json.dumps(values),flush=True)
        def on_train_loss_report(self,v):self.record('train',v)
        def on_val_loss_report(self,v):self.record('valid',v)
    config={**lm.CONFIG_DEFAULTS,**make_config(name,steps)}
    put(out/'config.json',config)
    put(out/'provenance.json',{'dataset_sha256':verify(),'template_sha256':sha(DAY/'chat_template.jinja'),
        'loss_sha256':sha(DAY.parent/'day-06/loss.py'),'model_config_sha256':sha(MODEL/'config.json'),
        'versions':{n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers')},
        'device':mx.device_info(),'resume':None,'seed':42,'optimizer_updates':steps//4,
        'source_sha256':{n:sha(ROOT/n) for n in ('core.py','run.py')}})
    for n in ('core.py','run.py'):shutil.copyfile(ROOT/n,out/('source-'+n))
    np.random.seed(42);mx.random.seed(42);started=time.perf_counter()
    model,tok=load(str(MODEL),tokenizer_config={'trust_remote_code':False})
    tok.chat_template=(DAY/'chat_template.jinja').read_text()
    cfg=SimpleNamespace(**config);tr,va,_=load_dataset(cfg,tok)
    lm.train=partial(lm.train,loss=loss)
    lm.train_model(cfg,model,tr,va,Metrics())
    tensors=mx.load(str(out/'adapter/adapters.safetensors'))
    assert all(bool(mx.all(mx.isfinite(t)).item()) for t in tensors.values())
    verify()
    put(out/'summary.json',{'seconds':time.perf_counter()-started,'steps':steps,'optimizer_updates':steps//4,
        'epochs':steps/352,'peak_memory_gb':mx.get_peak_memory()/1e9,
        'adapter_sha256':sha(out/'adapter/adapters.safetensors'),'all_finite':True})

def infer(label,splits,adapter,mode):
    manifest_hash=verify()
    selection=json.loads((ROOT/'selection.json').read_text()) if (ROOT/'selection.json').exists() else None
    adapter_hash=sha(Path(adapter)/'adapters.safetensors') if adapter else None
    for split in splits:guard_final(split,selection,adapter_hash,manifest_hash)
    mx,load,loss=load_runtime()
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.trainer import evaluate
    from mlx_lm.tuner.datasets import ChatDataset,CacheDataset
    started=time.perf_counter()
    model,tok=load(str(MODEL),adapter_path=adapter,tokenizer_config={'trust_remote_code':False})
    tok.chat_template=(DAY/'chat_template.jinja').read_text()
    out=ROOT/'evaluation';out.mkdir(exist_ok=True)
    if adapter and mode=='standard':
        probes={}
        for split,rs in [('valid',read(DAY/'data/valid.jsonl')),('train_probe',read(DAY/'data/train.jsonl')[:16])]:
            ds=CacheDataset(ChatDataset(rs,tok,mask_prompt=True))
            probes[split]=float(evaluate(model,ds,1,-1,max_seq_length=256,loss=loss))
        put(out/(label+'-loss.json'),{'loss':probes,'adapter_sha256':sha(Path(adapter)/'adapters.safetensors'),'note':'Exact saved weights; fixed first16 train rows and full24 valid rows.'})
    for split in splits:
        guard_final(split,selection,adapter_hash,manifest_hash)
        path=out/(label+'-'+split+'.jsonl')
        if path.exists():raise FileExistsError(path)
        rows=rows_for(split); refs=references(split);records=[];tick=time.perf_counter()
        for i,r in enumerate(rows):
            msgs=[dict(m) for m in r['messages']]
            if mode=='prompt':
                msgs[0]['content']+=' Сначала определи, кого или что спрашивают. Дай только ответ на этот вопрос, без дополнительных сведений. Различай владельца, создателя и материал; не путай Вторую и Третью эпохи. Не заменяй книжные факты сценами из фильмов.'
            elif mode=='oracle':
                if 'context' not in refs[r['id']]:raise ValueError('Missing oracle context')
                msgs[-1]['content']+='\nПроверенная справка: '+refs[r['id']]['context']
            prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            mx.random.seed(42);parts=[];last=None
            for part in stream_generate(model,tok,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0.0)):
                parts.append(part.text);last=part
            record={'id':r['id'],'answer':''.join(parts),'finish_reason':last.finish_reason,
                    'generation_tokens':last.generation_tokens}
            with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            records.append(record)
            if (i+1)%10==0 or i+1==len(rows):print(f'{label} {split}: {i+1}/{len(rows)}',flush=True)
        put(path.with_suffix('.meta.json'),{'rows':len(rows),'seconds':time.perf_counter()-tick,
            'temperature':0,'max_tokens':128,'mode':mode,'input_sha256':sha(DAY/'data'/f'{split}.jsonl'),
            'adapter_sha256':sha(Path(adapter)/'adapters.safetensors') if adapter else None,
            'dataset_manifest_sha256':verify(),'template_sha256':sha(DAY/'chat_template.jinja')})
        put(out/(label+'-'+split+'-scores.json'),score(split,records))
    put(out/(label+'-timing.json'),{'total_seconds':time.perf_counter()-started,'peak_memory_gb':mx.get_peak_memory()/1e9})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','infer']);p.add_argument('--name',required=True)
    p.add_argument('--steps',type=int,default=704);p.add_argument('--splits',default='knowledge_dev');p.add_argument('--adapter')
    p.add_argument('--mode',choices=['standard','prompt','oracle'],default='standard');a=p.parse_args()
    if a.action=='train':train(a.name,a.steps)
    else:infer(a.name,a.splits.split(','),a.adapter,a.mode)
