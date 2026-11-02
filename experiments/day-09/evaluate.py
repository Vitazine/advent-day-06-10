"""Score frozen accepted references against saved predictions, without inference."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'day-07'))
from score_candidates import score,read,predictions

def evaluate(reference,prediction):
    rows=copy.deepcopy(read(reference))
    for r in rows:
        if 'contrast' in r['tags'] and 'dimension' not in r:
            dimensions=[t.split(':',1)[1] for t in r['tags'] if t.startswith('dimension:')]
            if len(dimensions)!=1:raise ValueError('Contrast dimension missing: '+r['id'])
            r['dimension']=dimensions[0]
    ids={r['id'] for r in rows};pred=predictions(prediction)
    result=score(rows,{k:v for k,v in pred.items() if k in ids})
    result['excluded_prediction_ids']=sorted(set(pred)-ids)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--references',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=evaluate(a.references,a.predictions)
    with Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('n','exact_correct','invalid_responses','false_expenses_on_ignored','false_expenses_on_missing_context')}))
