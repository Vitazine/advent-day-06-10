"""CPU-only integrity checks; schema validity is not semantic correctness."""
import json,sys,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'day-07'))
from contract import parse_model_response

def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def write_new(path,rows):
    with Path(path).open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def validate_predictions(inputs,predictions):
    ids=[r['id'] for r in inputs]; predids=[r['id'] for r in predictions]
    if len(set(ids))!=len(ids) or len(set(predids))!=len(predids) or set(ids)!=set(predids):
        raise ValueError('Prediction IDs must match unique input IDs exactly')
    source={r['id']:r for r in inputs}
    valid={};rejected=[]
    for r in predictions:
        try:valid[r['id']]=parse_model_response(r['response'],source[r['id']]['input']['current_date'])
        except (ValueError,TypeError) as e:rejected.append({'id':r['id'],'reason':str(e),'response':r['response']})
    return valid,rejected

def normalized(text):return ' '.join(text.casefold().replace('ё','е').split())

def check_split_overlap(splits):
    seen_ids=set();families={};texts={}
    for split,rows in splits.items():
        for r in rows:
            if r['id'] in seen_ids:raise ValueError('Duplicate ID: '+r['id'])
            seen_ids.add(r['id'])
            for key,seen in [(r['family'],families),(normalized(r['input']['text']),texts)]:
                if key in seen and seen[key]!=split:raise ValueError('Cross-split overlap: '+key)
                seen[key]=split
    return {'rows':{s:len(r) for s,r in splits.items()},'unique_ids':len(seen_ids)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audited_rows(rows,audit):
    ids={r['id'] for r in rows}; accepted=set(audit['accepted_ids'])
    rejected={r['id']:r['reason'] for r in audit['rejected']}
    if accepted & set(rejected) or accepted | set(rejected) != ids:
        raise ValueError('Audit must cover all IDs exactly once without conflicts')
    badfamilies={r['family'] for r in rows if r['id'] in rejected and 'contrast' in r['tags']}
    kept=[]; dropped=[]
    for r in rows:
        if r['id'] in rejected or r['family'] in badfamilies:
            dropped.append({'id':r['id'],'reason':rejected.get(r['id'],'Rejected contrast family member')})
        else:kept.append(r)
    return kept,dropped
