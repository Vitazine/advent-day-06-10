"""Build v3 from immutable v2 plus grouped contrasts. Final eval stays frozen."""
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path
from contrast_cases import make_contrasts
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data'
SOURCE=ROOT/'versions/v2'
SYSTEM=(ROOT/'system.txt').read_text().strip()
SEED=20261101

def target(amount=None,category=None,currency='RUB',day='2026-11-01',payment=None,status='ok',reason=None):
    if status=='ignored' or reason in {'multiple_expenses','missing_context'}:
        amount=category=currency=day=payment=None
    return dict(status=status,amount=f'{Decimal(str(amount)):.2f}' if amount is not None else None,
                currency=currency,category=category,date=day,payment_method=payment,clarification_reason=reason)
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def messages(r):return [dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(r['input'],ensure_ascii=False)),dict(role='assistant',content=json.dumps(r['target'],ensure_ascii=False))]
def main():
    if (OUT/'manifest.json').exists():raise FileExistsError('Dataset frozen; create a new version explicitly')
    manifest=json.loads((SOURCE/'data/manifest.json').read_text())
    for name in ('system.txt','data/train_provenance.jsonl','data/valid_provenance.jsonl','data/eval.jsonl','data/eval_provenance.jsonl','data/mobile_eval.jsonl'):
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==manifest['files'][name],name
    assert (ROOT/'system.txt').read_bytes()==(SOURCE/'system.txt').read_bytes()
    additions=make_contrasts(target)
    train=read(SOURCE/'data/train_provenance.jsonl')
    # Keep complete lexical families 1 and 4 of each category (6 rows/category).
    retained=[r for r in train if 'regular' not in r['tags'] or int(r['family'].split('-')[-1]) in {1,4}]
    assert len(retained)==180
    rows={'train':retained+additions['train'],'valid':read(SOURCE/'data/valid_provenance.jsonl')+additions['valid']}
    OUT.mkdir(parents=True,exist_ok=True)
    for split,records in rows.items():
        random.Random(SEED+len(split)).shuffle(records)
        jsonl(OUT/f'{split}_provenance.jsonl',records)
        jsonl(OUT/f'{split}.jsonl',[dict(messages=messages(r)) for r in records])
    for name in ('eval.jsonl','eval_provenance.jsonl','mobile_eval.jsonl'):
        (OUT/name).write_bytes((SOURCE/'data'/name).read_bytes())
    sample=random.Random(817).sample(rows['train'],30)
    jsonl(OUT/'audit_sample.jsonl',sample)
    audit=['# Случайные 30 примеров v3\n','Seed=817. Результаты чтения — audit_review.md.\n']
    for r in sample:audit += [f"## {r['id']}\n",json.dumps(r['input'],ensure_ascii=False),'```json\n'+json.dumps(r['target'],ensure_ascii=False,indent=2)+'\n```']
    (ROOT/'audit_sample.md').write_text('\n\n'.join(audit)+'\n')
    print({**{k:len(v) for k,v in rows.items()},'eval':len(read(OUT/'eval.jsonl'))})
if __name__=='__main__':main()
