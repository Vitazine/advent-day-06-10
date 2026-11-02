"""Offline train/evaluate runner. Each training invocation loads the untouched base."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
from types import SimpleNamespace
from functools import partial
from common import *

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import mlx.core as mx
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
import mlx_lm.lora as lora_module
from loss import completion_loss
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

class Metrics(TrainingCallback):
    def __init__(self, path):
        self.path = path
    def record(self, kind, values):
        assert all(not isinstance(v,float) or math.isfinite(v) for v in values.values())
        with self.path.open('a') as f:
            f.write(json.dumps({'kind':kind, **values}) + '\n')
        print(json.dumps({'kind':kind, **values}), flush=True)
    def on_train_loss_report(self, values): self.record('train', values)
    def on_val_loss_report(self, values): self.record('valid', values)

def verify_data():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(ROOT / 'data' / name) == digest, name
    assert sha(ROOT / 'chat_template.jinja') == manifest['template_sha256']
    return manifest

def train(args):
    manifest = verify_data()
    out = ROOT / 'runs' / args.name
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    data = ROOT / 'data'
    train_rows = manifest['train_rows']
    if args.train_limit:
        data = out / 'data'
        rows = read_jsonl(ROOT / 'data/train.jsonl')[:args.train_limit]
        train_rows = len(rows)
        write_jsonl(data / 'train.jsonl', rows)
        write_jsonl(data / 'valid.jsonl', read_jsonl(ROOT / 'data/valid.jsonl'))
    if args.name.startswith('pilot'):
        data = out / 'data'
        pool = read_jsonl(ROOT / 'data/train_provenance.jsonl')
        rows = (sorted(pool, key=lambda r:abs(r['tokens']-86)) if args.mean_length else sorted(pool,key=lambda r:r['tokens'],reverse=True))[:16]
        write_jsonl(data / 'train.jsonl', [{'messages':messages(r['source'],r['target'])} for r in rows])
        write_jsonl(data / 'valid.jsonl', read_jsonl(ROOT / 'data/valid.jsonl')[:8])
    steps = args.steps or math.ceil(train_rows / args.batch)
    if steps % args.accumulation:
        raise ValueError('Microsteps must be divisible by gradient accumulation')
    keys = ['self_attn.q_proj', 'self_attn.v_proj']
    if args.wide:
        keys = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    config = {**CONFIG_DEFAULTS, 'model':str(MODEL), 'data':str(data), 'train':True,
              'optimizer':'adamw', 'optimizer_config':{'adamw':{'weight_decay':0.01}},
              'seed':args.seed, 'num_layers':16, 'batch_size':args.batch, 'iters':steps,
              'val_batches':-1, 'learning_rate':args.lr, 'steps_per_report':25 if steps>25 else 1,
              'steps_per_eval':max(1,steps//2), 'save_every':max(1,steps//2),
              'adapter_path':str(out / 'adapter'), 'max_seq_length':256,
              'grad_checkpoint':args.checkpoint, 'grad_accumulation_steps':args.accumulation, 'mask_prompt':True,
              'lora_parameters':{'keys':keys,'rank':16,'scale':2.0,'dropout':0.0}}
    updates = steps // args.accumulation
    warmup = max(1,round(updates*0.03))
    config['lr_schedule'] = {'name':'cosine_decay','arguments':[args.lr,max(1,updates-warmup),args.lr*0.1], 'warmup':warmup,'warmup_init':args.lr*0.1}
    write_json(out / 'config.json', config)
    write_json(out / 'provenance.json', {'data_manifest_sha256':sha(ROOT/'data/manifest.json'),
        'script_sha256':sha(Path(__file__)), 'device':mx.device_info(), 'python':platform.python_version(),
        'versions':{n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers','numpy')},
        'base_model':str(MODEL),'resume':None,'nominal_epochs':steps*args.batch/train_rows,
        'train_rows':train_rows,'optimizer_updates':updates,'effective_batch':args.batch*args.accumulation,
        'training_files_sha256':{n:sha(data/n) for n in ('train.jsonl','valid.jsonl')},
        'loss_name':'completion_loss: target position < actual sequence length',
        'source_sha256':{n:sha(ROOT/n) for n in ('run.py','common.py','loss.py','prepare.py','eval_cases.py')}})
    for name in ('run.py','common.py','loss.py'):
        (out / ('source-' + name)).write_bytes((ROOT/name).read_bytes())
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    started = time.perf_counter()
    model, tokenizer = load(str(MODEL), tokenizer_config={'trust_remote_code':False})
    configure(tokenizer)
    cfg = SimpleNamespace(**config)
    train_set, valid_set, _ = load_dataset(cfg, tokenizer)
    # Local override: upstream 0.32.2 default_loss includes first padding target.
    original_train = lora_module.train
    lora_module.train = partial(original_train, loss=completion_loss)
    try:
        train_model(cfg, model, train_set, valid_set, Metrics(out / 'metrics.jsonl'))
    finally:
        lora_module.train = original_train
    tensors = mx.load(str(out / 'adapter/adapters.safetensors'))
    assert all(bool(mx.all(mx.isfinite(t)).item()) for t in tensors.values())
    summary = {'seconds':time.perf_counter()-started,'peak_memory_gb':mx.get_peak_memory()/1e9,
               'adapter_sha256':sha(out/'adapter/adapters.safetensors'),'tensor_count':len(tensors),'all_finite':True,'steps':steps}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)

def infer(args):
    verify_data()
    if args.split == 'final' and not (ROOT / 'selection.json').exists():
        raise RuntimeError('Final locked until selection is recorded')
    out = ROOT / 'evaluation' / f'{args.name}-{args.split}.jsonl'
    if out.exists(): raise FileExistsError(out)
    adapter = None if args.name == 'base' else str(ROOT/'runs'/args.name/'adapter')
    model, tokenizer = load(str(MODEL),adapter_path=adapter,tokenizer_config={'trust_remote_code':False})
    configure(tokenizer)
    rows = read_jsonl(ROOT/'data'/f'{args.split}_eval.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for r in rows:
        prompt = tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
        mx.random.seed(42)
        answer = generate(model,tokenizer,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0.0),verbose=False)
        with out.open('a') as f: f.write(json.dumps({**r,'answer':answer},ensure_ascii=False)+'\n')
        print(f"{r['id']}: {answer}",flush=True)
    write_json(out.with_suffix('.meta.json'),{'seconds':time.perf_counter()-started,'rows':len(rows),
        'temperature':0,'max_tokens':128,'data_sha256':sha(ROOT/'data'/f'{args.split}_eval.jsonl'),
        'adapter_sha256':None if adapter is None else sha(Path(adapter)/'adapters.safetensors')})

def chat(args):
    if not args.text:
        raise ValueError('Specify --text')
    adapter = None if args.name == 'base' else str(ROOT/'runs'/args.name/'adapter')
    model, tokenizer = load(str(MODEL),adapter_path=adapter,tokenizer_config={'trust_remote_code':False})
    configure(tokenizer)
    prompt = tokenizer.apply_chat_template(messages(args.text),tokenize=False,add_generation_prompt=True)
    print(generate(model,tokenizer,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0.0),verbose=False))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','infer','chat'])
    p.add_argument('--text')
    p.add_argument('--name',required=True)
    p.add_argument('--split',choices=['dev','final','general'],default='dev')
    p.add_argument('--steps',type=int)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--wide',action='store_true')
    p.add_argument('--checkpoint',action='store_true')
    p.add_argument('--mean-length',action='store_true')
    p.add_argument('--train-limit',type=int)
    p.add_argument('--accumulation',type=int,default=1)
    args=p.parse_args()
    {'train':train,'infer':infer,'chat':chat}[args.action](args)
