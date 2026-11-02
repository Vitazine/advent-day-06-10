"""Frozen-data training protocol adapted from Day 6, CPU-safe helpers."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'training-v3'
MODEL=Path(__import__('os').environ.get('MODEL_PATH', 'models/Qwen3-4B-Instruct-2507-4bit'))
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def verify_data():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    assert manifest['version']=='3'
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,name
    return manifest

def configuration(name,lr,seed,wide,pilot=False):
    steps=4 if pilot else 420
    updates=steps//4;warmup=max(1,round(updates*.03))
    keys=['self_attn.q_proj','self_attn.v_proj'] if not wide else ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    return dict(model=str(MODEL),train=True,fine_tune_type='lora',optimizer='adamw',optimizer_config={'adamw':{'weight_decay':.01}},data=str(ROOT/'data'),seed=seed,num_layers=16,batch_size=1,iters=steps,val_batches=-1,learning_rate=lr,steps_per_report=1 if pilot else 20,steps_per_eval=4 if pilot else 210,save_every=4 if pilot else 420,resume_adapter_file=None,adapter_path=str(RUN/'runs'/name/'adapter'),max_seq_length=2048,grad_checkpoint=False,grad_accumulation_steps=4,mask_prompt=True,lora_parameters=dict(keys=keys,rank=16,scale=2.0,dropout=0.0),lr_schedule=dict(name='cosine_decay',arguments=[lr,max(1,updates-warmup),lr*.1],warmup=warmup,warmup_init=lr*.1))
def selection_key(result,baseline,wide,lr):
    regressions=sum(bool(baseline.get(i)) and not ok for i,ok in result['outcomes'].items())
    dangerous=result['false_expenses_on_ignored']+result['false_expenses_on_missing_context']
    return (result['exact_correct'],-dangerous,-regressions,-int(wide),-lr)
