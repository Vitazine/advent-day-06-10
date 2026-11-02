"""Day 9: continue Day 7 LoRA. Run train, pilot, or input-only inference.

Continuation uses LR 5e-5 (half Day 7 iteration-4's 1e-4) to reduce drift.
Optimizer state is fresh: this continues weights, not the old optimizer.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
from runner_config import (ROOT, DAY7, MODEL, START, LORA, check_resume, length_report,
                           read_inputs, read_rows, reserve_output, schedule, sha, write_new, verify_manifest)


def train(args):
    manifest_hash = verify_manifest()
    source_hash = check_resume(args.resume_adapter_file)
    if Path(args.resume_adapter_file).resolve() != START.resolve():
        raise ValueError('Day 9 must continue Day 7 iteration-4')
    train_path, valid_path = ROOT/'data/train.jsonl', ROOT/'data/valid.jsonl'
    train_rows, valid_rows = read_rows(train_path), read_rows(valid_path)
    if args.action == 'pilot' and len(train_rows) < 16: raise ValueError('Pilot needs 16 examples')
    count = 16 if args.action == 'pilot' else len(train_rows)
    sched = schedule(count, args.batch_size, args.lr)
    out = Path(args.output).resolve()
    reserve_output(out)
    started = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    import mlx_lm.lora as lora
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.tuner.callbacks import TrainingCallback
    sys.path.insert(0, str(DAY7))
    from training_loss import completion_loss

    class Uniform:
        def __call__(self, x): return mx.zeros((*x.shape, 16))

    batch = mx.array([[1,2,3,4,5,6,7,8,9,10,0,0], [1,2,3,4,5,6,0,0,0,0,0,0]])
    loss, ntokens = completion_loss(Uniform(), batch, mx.array([[6,10],[3,6]]))
    loss_check = dict(target_tokens=int(ntokens.item()), loss=float(loss.item()))
    if loss_check['target_tokens'] != 7 or abs(loss_check['loss'] - 2.7725887) > 1e-5:
        raise ValueError('Completion loss padding boundary check failed')

    class Metrics(TrainingCallback):
        def record(self, kind, values):
            record = dict(kind=kind, **values)
            text = json.dumps(record, allow_nan=False)
            with (out/'metrics.jsonl').open('a') as f: f.write(text+'\n')
            print(text, flush=True)
        def on_train_loss_report(self, values): self.record('train', values)
        def on_val_loss_report(self, values): self.record('valid', values)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    cold = time.perf_counter()
    model, tokenizer = load(str(MODEL), tokenizer_config={'trust_remote_code':False})
    tokenizer.chat_template = (DAY7/'chat_template.jinja').read_text()
    mx.eval(model.parameters())
    cold_seconds = time.perf_counter()-cold
    # Tokenize with the exact implementation consumed by MLX-LM (including offsets).
    trainset = ChatDataset(train_rows, tokenizer, mask_prompt=True)
    validset = ChatDataset(valid_rows, tokenizer, mask_prompt=True)
    train_tokens = [trainset.process(row) for row in train_rows]
    valid_tokens = [validset.process(row) for row in valid_rows]
    lengths = dict(train=length_report(train_tokens), valid=length_report(valid_tokens))
    if args.action == 'pilot':
        indices = sorted(range(len(train_rows)), key=lambda i: len(train_tokens[i][0]), reverse=True)[:16]
        train_rows = [train_rows[i] for i in indices]
        trainset = ChatDataset(train_rows, tokenizer, mask_prompt=True)
        write_new(out/'pilot-indices.json', indices)
        lengths['pilot'] = length_report([train_tokens[i] for i in indices])
    # Same fixed 16 validation examples for both batch sizes (8 or 16 batches).
    valid_rows = valid_rows[:16]
    validset = ChatDataset(valid_rows, tokenizer, mask_prompt=True)
    if len(valid_rows) < args.batch_size: raise ValueError('Insufficient validation examples')
    updates = sched.pop('optimizer_updates')
    config = {**lora.CONFIG_DEFAULTS, **sched,
        'model':str(MODEL), 'train':True, 'data':str(ROOT/'data'), 'seed':args.seed,
        'fine_tune_type':'lora', 'num_layers':16, 'lora_parameters':LORA,
        'batch_size':args.batch_size, 'learning_rate':args.lr,
        'optimizer':'adamw', 'optimizer_config':{'adamw':{'weight_decay':.01}},
        'val_batches':min(16, len(valid_rows)//args.batch_size),
        'steps_per_report':4//args.batch_size if args.action == 'pilot' else 20,
        'save_every':sched['iters'], 'max_seq_length':2048, 'mask_prompt':True,
        'grad_checkpoint':False, 'resume_adapter_file':str(Path(args.resume_adapter_file).resolve()),
        'adapter_path':str(out/'adapter')}
    provenance = dict(manifest_sha256=manifest_hash, start_adapter_sha256=source_hash, tokenizer_lengths=lengths, loss_check=loss_check,
        cold_load_seconds=cold_seconds, optimizer_updates=updates, examples=count,
        nominal_epochs=1, optimizer_state='fresh', versions={name:importlib.metadata.version(name) for name in ('mlx','mlx-lm','transformers')},
        source_sha256={str(p):sha(p) for p in [train_path, valid_path, ROOT/'system.txt', DAY7/'chat_template.jinja', DAY7/'training_loss.py']})
    write_new(out/'config.json', config)
    write_new(out/'provenance.json', provenance)
    print(json.dumps(provenance), flush=True)
    original = lora.train

    def checked_train(**kwargs):
        # train_model uses strict=False on load: independently verify every LoRA value
        # after conversion/loading and before the first optimizer update.
        if sha(args.resume_adapter_file) != source_hash: raise ValueError('Source adapter changed')
        tensors = mx.load(str(args.resume_adapter_file))
        actual = dict(tree_flatten(model.trainable_parameters()))
        if set(tensors) != set(actual): raise ValueError('Resume tensor keys mismatch')
        for key, tensor in tensors.items():
            value = actual[key]
            if tensor.shape != value.shape or not bool(mx.all(tensor == value).item()):
                raise ValueError(f'Resume tensor not loaded exactly: {key}')
        write_new(out/'resume-verified.json', dict(sha256=source_hash, tensors=len(tensors), exact=True))
        return original(**kwargs, loss=completion_loss)

    lora.train = checked_train
    training_started = time.perf_counter()
    try: lora.train_model(SimpleNamespace(**config), model, trainset, validset, Metrics())
    finally: lora.train = original
    mx.synchronize()
    adapter = out/'adapter/adapters.safetensors'
    tensors = mx.load(str(adapter))
    if not tensors or not all(bool(mx.all(mx.isfinite(v)).item()) for v in tensors.values()):
        raise ValueError('Non-finite or empty trained adapter')
    if sha(args.resume_adapter_file) != source_hash: raise ValueError('Source adapter changed')
    verify_manifest()
    summary = dict(wall_seconds=time.perf_counter()-started, train_seconds=time.perf_counter()-training_started,
                   cold_load_seconds=cold_seconds, peak_memory_gb=mx.get_peak_memory()/1e9,
                   optimizer_updates=updates, examples=count, adapter_sha256=sha(adapter))
    write_new(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def infer(args):
    rows = read_inputs(args.inputs)
    adapter_file = Path(args.adapter)/'adapters.safetensors'
    adapter_hash = check_resume(adapter_file)
    out = Path(args.output).resolve()
    reserve_output(out)
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    started = time.perf_counter()
    model, tokenizer = load(str(MODEL), adapter_path=str(Path(args.adapter).resolve()),
                            tokenizer_config={'trust_remote_code':False})
    tokenizer.chat_template = (DAY7/'chat_template.jinja').read_text()
    mx.eval(model.parameters())
    cold_seconds = time.perf_counter()-started
    system = (ROOT/'system.txt').read_text().strip()
    prompts = []
    for row in rows:
        messages = [dict(role='system', content=system),
                    dict(role='user', content=json.dumps(row['input'], ensure_ascii=False))]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n > 2048: raise ValueError(f"Input {row['id']} exceeds 2048 tokens")
        prompts.append((prompt, n))
    with (out/'responses.jsonl').open('x') as f:
        for row, (prompt, n) in zip(rows, prompts):
            mx.random.seed(42)
            begin = time.perf_counter()
            answer = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                              sampler=make_sampler(temp=0.), verbose=False)
            mx.synchronize()
            result = dict(id=row['id'], response=answer, seconds=time.perf_counter()-begin,
                          prompt_tokens=n, response_tokens=len(tokenizer.encode(answer, add_special_tokens=False)))
            f.write(json.dumps(result, ensure_ascii=False)+'\n')
            f.flush()
            print(json.dumps(dict(id=row['id'], seconds=result['seconds'])), flush=True)
    write_new(out/'summary.json', dict(rows=len(rows), cold_load_seconds=cold_seconds,
        wall_seconds=time.perf_counter()-started, peak_memory_gb=mx.get_peak_memory()/1e9,
        input_sha256=sha(args.inputs), adapter_sha256=adapter_hash,
        system_sha256=sha(ROOT/'system.txt'), template_sha256=sha(DAY7/'chat_template.jinja'),
        temperature=0, max_tokens=256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('train', 'pilot'):
        p = subs.add_parser(action)
        p.add_argument('--output', required=True)
        p.add_argument('--resume-adapter-file', required=True)
        p.add_argument('--batch-size', type=int, choices=(1,2), default=1)
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--lr', type=float, default=5e-5)
    p = subs.add_parser('infer')
    p.add_argument('--inputs', required=True, help='JSONL with only id,input fields')
    p.add_argument('--adapter', required=True, help='Adapter directory (Day 7 or Day 9)')
    p.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    (infer if args.action == 'infer' else train)(args)


if __name__ == '__main__': main()
