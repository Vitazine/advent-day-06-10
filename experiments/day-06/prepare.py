import csv
import json
import random
import shutil
from collections import Counter
from common import *
from eval_cases import DEV, FINAL, GENERAL

def main():
    out = ROOT / 'data'
    if (out / 'manifest.json').exists():
        raise FileExistsError('Frozen dataset already exists')
    provenance = json.loads((OLD / 'model-4b-provenance.json').read_text())
    for name, info in provenance['files'].items():
        assert sha(MODEL / name) == info['sha256'], name
    old_report = json.loads((OLD / 'dataset-report.json').read_text())
    raw = OLD / 'data/raw/ru_paradetox'
    for name, digest in old_report['source_sha256'].items():
        assert sha(raw / name) == digest, name
    shutil.copyfile(OLD / 'qwen4b-official-template.jinja', ROOT / 'chat_template.jinja')
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.tuner.datasets import ChatDataset
    tokenizer = configure(load_tokenizer(MODEL, tokenizer_config_extra={'local_files_only': True, 'trust_remote_code': False}))
    evaluations = {}
    for split, cases in [('dev', DEV), ('final', FINAL), ('general', GENERAL)]:
        rows = []
        for i, (category, source, target) in enumerate(cases):
            msgs = messages(source)
            if split == 'general':
                msgs[0]['content'] = 'Ты полезный помощник. Выполни запрос пользователя. Ответь кратко на русском.'
            rows.append(dict(id=f'{split}-{i+1:02}', category=category, source=source, target=target, messages=msgs))
        evaluations[split] = rows
        write_jsonl(out / f'{split}_eval.jsonl', rows)
    all_eval = sum(evaluations.values(), [])
    forbidden = {key(r[f]) for r in all_eval for f in ('source', 'target')}
    raw_rows = {}
    for split in ('train', 'dev'):
        with (raw / f'{split}.tsv').open() as f:
            raw_rows[split] = [dict(id=f'{split}:{i+2}', source=r['ru_toxic_comment'], target=r['ru_neutral_comment']) for i, r in enumerate(csv.DictReader(f, delimiter='\t'))]
    official_dev_keys = {key(r[f]) for r in raw_rows['dev'] for f in ('source', 'target')}
    selected, rejected = {}, {}
    for split in ('train', 'dev'):
        rows, seen, counts = [], set(), Counter()
        for r in raw_rows[split]:
            source, target = r['source'], r['target']
            if not eligible(source, target):
                counts['technical_filter'] += 1; continue
            if toxic_target(target):
                counts['toxic_target_lexical'] += 1; continue
            if key(source) in seen:
                counts['duplicate_or_alternative'] += 1; continue
            if {key(source), key(target)} & (forbidden | (official_dev_keys if split == 'train' else set())):
                counts['split_conflict'] += 1; continue
            record = {'messages': messages(source, target)}
            tokens, offset = ChatDataset([record], tokenizer, mask_prompt=True).process(record)
            prefix = tokenizer.apply_chat_template(messages(source), tokenize=True, add_generation_prompt=True)
            assert tokens[:offset] == prefix and 0 < offset < len(tokens)
            assert tokenizer.decode(tokens[offset:]).startswith(target)
            if len(tokens) > 256:
                counts['too_long'] += 1; continue
            seen.add(key(source))
            rows.append({**r, 'tokens': len(tokens)})
        random.Random(20261101).shuffle(rows)
        selected[split] = rows
        rejected[split] = dict(counts)
    train, valid = selected['train'], selected['dev'][:128]
    assert_disjoint(train, valid)
    assert_disjoint(train, all_eval)
    for split, rows in [('train', train), ('valid', valid)]:
        write_jsonl(out / f'{split}.jsonl', [{'messages': messages(r['source'], r['target'])} for r in rows])
        write_jsonl(out / f'{split}_provenance.jsonl', rows)
    write_jsonl(out / 'audit_sample.jsonl', random.Random(817).sample(train, 30))
    manifest = dict(model=provenance, source_sha256=old_report['source_sha256'], train_rows=len(train), valid_rows=len(valid), rejected=rejected,
                    template_sha256=sha(ROOT / 'chat_template.jinja'), max_tokens=max(r['tokens'] for r in train),
                    evaluation_counts={k:len(v) for k,v in evaluations.items()},
                    limitations=['Normalized exact overlap excluded; semantic duplicates not fully audited', 'Synthetic evaluation is not representative of all real-world language'])
    manifest['files'] = {p.name:sha(p) for p in out.glob('*.jsonl')}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('model','files','source_sha256')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
