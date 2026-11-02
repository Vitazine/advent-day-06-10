"""Assemble audited teacher targets. Run once; keep raw evidence immutable."""
import json,random
from collections import Counter,defaultdict
from dataset_tools import *

def main():
    data=ROOT/'data'; training=[]; exclusions=[]
    for kind,inputs,predfile in [('train','train-inputs.jsonl','train-teacher.jsonl'),('replay','replay-inputs.jsonl','replay-teacher.jsonl')]:
        rows=read(data/inputs); pred=read(data/predfile)
        targets,bad=validate_predictions(rows,pred)
        if bad:raise ValueError('Schema invalid teacher outputs require audit resolution')
        audit=json.loads((ROOT/f'audit-{kind}.json').read_text())
        rows,dropped=audited_rows(rows,audit);exclusions.extend(dropped)
        for r in rows:training.append(dict(r,target=targets[r['id']],teacher_source=predfile))
    near=json.loads((ROOT/'overlap-review.json').read_text())
    remove={r['train_id'] for r in near}
    badfamilies={r['family'] for r in training if r['id'] in remove}
    for r in training:
        if r['family'] in badfamilies:exclusions.append(dict(id=r['id'],reason='Near duplicate heldout input/family'))
    training=[r for r in training if r['family'] not in badfamilies]
    # No repeated examples to pad an epoch. Remove up to 3 standalone replay
    # examples in deterministic order, preserving all contrast groups.
    remainder=len(training)%4
    candidates=sorted([r for r in training if r['id'].startswith('d9-replay-') and 'contrast' not in r['tags']],key=lambda r:r['id'])
    trim={r['id'] for r in candidates[:remainder]}
    exclusions.extend(dict(id=i,reason='Effective batch4 alignment; no duplication') for i in sorted(trim))
    training=[r for r in training if r['id'] not in trim]
    holdaudit=json.loads((ROOT/'audit-holdout.json').read_text())
    accepted=set(holdaudit['accepted_ids']); issues={r['id'] for r in holdaudit['issues']}
    holdouts={s:read(data/f'{s}-reference.jsonl') for s in ('valid','final')}
    allhold={r['id'] for rows in holdouts.values() for r in rows}
    if accepted & issues or accepted|issues != allhold:raise ValueError('Holdout audit incomplete')
    for split,rows in holdouts.items():
        badfamilies={r['family'] for r in rows if r['id'] in issues and 'contrast' in r['tags']}
        holdouts[split]=[r for r in rows if r['id'] in accepted and r['family'] not in badfamilies]
    splits={'train':training,**holdouts};report=check_split_overlap(splits)
    # Training order deterministic and independent of final/model scores.
    random.Random(20261102).shuffle(training)
    system=(ROOT/'system.txt').read_text().strip()
    def messages(r):return {'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(r['input'],ensure_ascii=False)},{'role':'assistant','content':json.dumps(r['target'],ensure_ascii=False,separators=(',',':'))}]}
    write_new(data/'train.jsonl',[messages(r) for r in training])
    # Round-robin status/category buckets for a representative first16 loss subset.
    buckets=defaultdict(list)
    for r in holdouts['valid']:buckets[(r['target']['status'],r['target']['category'])].append(r)
    ordered=[]
    while any(buckets.values()):
        for k in sorted(buckets,key=str):
            if buckets[k]:ordered.append(buckets[k].pop(0))
    write_new(data/'valid.jsonl',[messages(r) for r in ordered])
    for split,rows in splits.items():
        write_new(data/f'{split}-accepted.jsonl',rows)
        if split!='train':write_new(data/f'{split}-accepted-inputs.jsonl',[dict(id=r['id'],input=r['input']) for r in rows])
    report.update(status_counts={s:dict(Counter(r['target']['status'] for r in rows)) for s,rows in splits.items()},excluded_training=exclusions,excluded_holdout=holdaudit['issues'],loss_subset_ids=[r['id'] for r in ordered[:16]],audit='independent AI, not human',near_duplicate_threshold=.85)
    (ROOT/'dataset-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    files=list(data.glob('*.jsonl'))+[ROOT/'system.txt']+list(ROOT.glob('audit-*.json'))
    (ROOT/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):sha(p) for p in files}},indent=2))
    print(json.dumps({k:report[k] for k in ('rows','status_counts')},ensure_ascii=False))
if __name__=='__main__':main()
