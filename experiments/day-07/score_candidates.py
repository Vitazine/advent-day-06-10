"""Score saved responses; never performs model inference or trains weights."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from contract import parse_model_response,FIELDS

def score(rows,predictions):
    ids=[r['id'] for r in rows]
    if not rows or len(set(ids))!=len(ids) or set(ids)!=set(predictions):
        raise ValueError('References and predictions must contain the same unique IDs')
    outcomes={};groups=defaultdict(list);slices=defaultdict(list)
    fields={f:0 for f in sorted(FIELDS)}
    invalid=false_expenses=context_expenses=0
    for r in rows:
        try: candidate=parse_model_response(predictions[r['id']],r['input']['current_date'])
        except ValueError: candidate=None;invalid+=1
        correct=candidate==r['target'];outcomes[r['id']]=correct
        for field in fields:fields[field]+=int(candidate is not None and candidate[field]==r['target'][field])
        if candidate is not None and candidate['status']=='ok':
            false_expenses+=int(r['target']['status']=='ignored')
            context_expenses+=int(r['target']['clarification_reason']=='missing_context')
        slices['status:'+r['target']['status']].append(correct)
        if r['target']['category'] is not None:slices['category:'+r['target']['category']].append(correct)
        if 'mobile' in r['tags']:slices['mobile'].append(correct)
        if 'contrast' in r['tags']:
            groups[r['family']].append(correct)
            slices['dimension:'+r['dimension']].append(correct)
    if any(len(g)!=4 for g in groups.values()):raise ValueError('Contrast groups must be complete')
    return dict(n=len(rows),exact_correct=sum(outcomes.values()),exact_accuracy=sum(outcomes.values())/len(rows),
                invalid_responses=invalid,field_correct=fields,false_expenses_on_ignored=false_expenses,
                false_expenses_on_missing_context=context_expenses,contrast_groups=len(groups),
                contrast_groups_all_correct=sum(all(g) for g in groups.values()),
                slices={k:dict(correct=sum(v),n=len(v)) for k,v in sorted(slices.items())},outcomes=outcomes)
def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def predictions(path):
    records=read(path)
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate prediction IDs')
    return {r['id']:r['response'] for r in records}
def main():
    p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--predictions',required=True);p.add_argument('--baseline');args=p.parse_args()
    rows=read(args.references);result=score(rows,predictions(args.predictions))
    if args.baseline:
        base=score(rows,predictions(args.baseline))
        wins=[i for i,ok in result['outcomes'].items() if ok and not base['outcomes'][i]]
        losses=[i for i,ok in result['outcomes'].items() if not ok and base['outcomes'][i]]
        result['paired']=dict(improved=wins,regressed=losses,unchanged=len(rows)-len(wins)-len(losses),baseline_exact=base['exact_correct'])
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
