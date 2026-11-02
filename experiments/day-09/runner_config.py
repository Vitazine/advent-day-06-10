"""CPU-only Day 9 configuration and fail-closed input/output checks."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DAY7 = ROOT.parent / 'day-07'
MODEL = Path(__import__('os').environ.get('MODEL_PATH', 'models/Qwen3-4B-Instruct-2507-4bit'))
START = DAY7 / 'training-v3/runs/iteration-4/adapter/adapters.safetensors'
LORA = dict(keys=['self_attn.q_proj', 'self_attn.v_proj'], rank=16, scale=2.0, dropout=0.0)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_inputs(path):
    rows = read_rows(path)
    seen = set()
    if not rows: raise ValueError('Empty inference input')
    for row in rows:
        if set(row) != {'id', 'input'}: raise ValueError('Inference accepts only id,input; no targets')
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in seen:
            raise ValueError('Input IDs must be nonempty unique strings')
        seen.add(row['id'])
    return rows


def reserve_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def check_resume(path):
    if not path: raise ValueError('Explicit resume_adapter_file is mandatory')
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    source = json.loads((path.parent / 'adapter_config.json').read_text())
    expected = dict(model=str(MODEL), num_layers=16, fine_tune_type='lora', lora_parameters=LORA)
    for key, value in expected.items():
        if source.get(key) != value: raise ValueError(f'Resume architecture mismatch: {key}')
    return sha(path)


def schedule(examples, batch_size, lr=5e-5):
    if batch_size not in (1, 2) or examples < 4 or examples % 4:
        raise ValueError('Full epoch requires examples divisible by effective batch 4; batch must be 1 or 2')
    if not math.isfinite(lr) or lr <= 0: raise ValueError('Learning rate must be positive finite')
    accum = 4 // batch_size
    updates = examples // 4
    warmup = max(1, round(updates * .03))
    return dict(iters=examples // batch_size, grad_accumulation_steps=accum,
                optimizer_updates=updates, steps_per_eval=max(1, updates // 2) * accum,
                lr_schedule=dict(name='cosine_decay', arguments=[lr, max(1, updates-warmup), lr*.1],
                                 warmup=warmup, warmup_init=lr*.1))


def length_report(samples, limit=2048):
    lengths = []
    for index, (tokens, offset) in enumerate(samples):
        n = len(tokens)
        if n > limit: raise ValueError(f'Example {index} has {n} tokens > {limit}; truncation forbidden')
        if offset < 0 or offset >= n or n < 2: raise ValueError(f'Example {index} has no completion')
        lengths.append(n)
    if not lengths: raise ValueError('Empty dataset')
    ordered = sorted(lengths)
    return dict(count=len(lengths), min=min(lengths), max=max(lengths),
                mean=sum(lengths)/len(lengths), p95=ordered[math.ceil(.95*len(lengths))-1])

def verify_manifest(root=ROOT):
    manifest=json.loads((Path(root)/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(Path(root)/name)!=digest:raise ValueError('Frozen input changed: '+name)
    return sha(Path(root)/'manifest.json')
