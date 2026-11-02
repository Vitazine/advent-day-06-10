"""Validate data artifacts and optionally the local Qwen tokenizer, never weights."""
import argparse
from collections import Counter
from datetime import date, timedelta
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
CATEGORIES={'groceries','cafes','transport','housing','utilities','health','shopping','entertainment','education','other'}
FIELDS={'status','amount','currency','category','date','payment_method','clarification_reason'}
MODEL=Path(__import__('os').environ.get('MODEL_PATH', 'models/Qwen3-4B-Instruct-2507-4bit'))
def read(name): return [json.loads(l) for l in (DATA/name).read_text().splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def key(text): return ' '.join(re.findall(r'\w+|[$€₽₱]',unicodedata.normalize('NFKC',text).casefold().replace('ё','е')))
def shape(text): return re.sub(r'\d+(?:[.,]\d+)?','N',key(text))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--tokenizer',action='store_true'); args=parser.parse_args()
    rows={s:read(f'{s}_provenance.jsonl') for s in ('train','valid','eval')}
    report={'counts':{},'distribution':{},'normalized_duplicates':0,'cross_split_shape_duplicates':0,'near_duplicate_candidates':[]}
    seen=set(); shape_sets={}; family_sets={}; system=(ROOT/'system.txt').read_text().strip()
    for split, records in rows.items():
        assert len(records)=={'train':420,'valid':137,'eval':140}[split]
        assert len({r['id'] for r in records})==len(records)
        report['counts'][split]=len(records)
        report['distribution'][split]={field:dict(Counter(str(r['target'][field]) for r in records)) for field in ('status','category','currency','payment_method','clarification_reason')}
        shape_sets[split]={shape(r['input']['text']) for r in records}
        family_sets[split]={r['family'] for r in records}
        wire=read(f'{split}.jsonl')
        assert len(wire)==len(records)
        for r,w in zip(records,wire):
            t=r['target']; inp=r['input'];
            from contract import parse_model_response
            assert parse_model_response(json.dumps(t), inp['current_date']) == t
            text=inp['text']; k=key(text)
            assert k not in seen, f'Duplicate: {r["id"]}'
            seen.add(k)
            assert set(t)==FIELDS
            assert set(inp)=={'text','current_date'}
            date.fromisoformat(inp['current_date'])
            assert t['status'] in {'ok','ignored','needs_clarification'}
            assert t['category'] in CATEGORIES|{None}
            assert t['currency'] in {'RUB','PHP','USD','EUR',None}
            assert t['payment_method'] in {'cash','card','transfer',None}
            assert t['amount'] is None or (re.fullmatch(r'\d+\.\d{2}',t['amount']) and float(t['amount'])>0)
            if t['date'] is not None:
                assert date.fromisoformat(t['date'])<=date.fromisoformat(inp['current_date'])
            if t['status']=='ignored': assert all(v is None for k,v in t.items() if k!='status')
            elif t['status']=='ok':
                assert all(t[f] is not None for f in ('amount','currency','category','date'))
                assert t['clarification_reason'] is None
            else:
                reason=t['clarification_reason']
                null_field={'missing_amount':'amount','ambiguous_amount':'amount','ambiguous_currency':'currency','ambiguous_date':'date'}
                assert reason in set(null_field)|{'multiple_expenses','missing_context'}
                if reason in {'multiple_expenses','missing_context'}: assert all(t[f] is None for f in ('amount','currency','category','date','payment_method'))
                else: assert t[null_field[reason]] is None
            if t['date'] is not None:
                for word,offset in [('позавчера',2),('вчера',1),('сегодня',0)]:
                    if re.search(r'\b'+word+r'\b',text.lower()):
                        assert t['date']==(date.fromisoformat(inp['current_date'])-timedelta(days=offset)).isoformat(),r['id']
                        break
            assert w['messages'][0]=={'role':'system','content':system}
            assert json.loads(w['messages'][1]['content'])==inp
            if split=='eval':
                assert len(w['messages'])==2 and w['target']==t and w['id']==r['id']
            else:
                assert len(w['messages'])==3 and w['messages'][2]['role']=='assistant'
                assert json.loads(w['messages'][2]['content'])==t
    for a,b in [('train','valid'),('train','eval'),('valid','eval')]:
        assert not family_sets[a]&family_sets[b]
        assert not shape_sets[a]&shape_sets[b], f'Shape overlap: {a}/{b}'
        for x in rows[a]:
            for y in rows[b]:
                sx,sy=shape(x['input']['text']),shape(y['input']['text'])
                ratio=SequenceMatcher(None,sx,sy,autojunk=False).ratio()
                jaccard=len(set(sx.split())&set(sy.split()))/len(set(sx.split())|set(sy.split()))
                if ratio>=0.82 or jaccard>=0.75:
                    report['near_duplicate_candidates'].append(dict(a=x['id'],b=y['id'],similarity=round(ratio,3),jaccard=round(jaccard,3)))
    sample=read('audit_sample.jsonl'); assert len(sample)==30
    assert all(r in rows['train'] for r in sample)
    for name in ('eval.jsonl','eval_provenance.jsonl','mobile_eval.jsonl'):
        assert (DATA/name).read_bytes()==(ROOT/'versions/v2/data'/name).read_bytes(), name
    assert (ROOT/'system.txt').read_bytes()==(ROOT/'versions/v2/system.txt').read_bytes()
    groups={}
    for split in ('train','valid'):
        for r in rows[split]:
            if 'contrast' in r['tags']: groups.setdefault(r['family'],[]).append((split,r))
    for family, members in groups.items():
        assert len(members)==4 and len({s for s,r in members})==1, family
    report['contrast_groups']=len(groups)
    report['contrast_counts']={s:sum('contrast' in r['tags'] for r in records) for s,records in rows.items()}
    report['frozen_v2_eval_and_prompt']=True
    report['mobile_counts']={s:sum('mobile' in r['tags'] for r in records) for s,records in rows.items()}
    mobile=read('mobile_eval.jsonl')
    assert mobile == [w for r,w in zip(rows['eval'],read('eval.jsonl')) if 'mobile' in r['tags']]
    report['tokenizer']={'checked':False}
    if args.tokenizer:
        from mlx_lm.utils import load_tokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        tokenizer=load_tokenizer(MODEL,tokenizer_config_extra={'local_files_only':True,'trust_remote_code':False})
        tokenizer.chat_template=(ROOT/'chat_template.jinja').read_text()
        lengths=[]; completions=[]
        for split in ('train','valid'):
            for record in read(f'{split}.jsonl'):
                tokens,offset=ChatDataset([record],tokenizer,mask_prompt=True).process(record)
                prefix=tokenizer.apply_chat_template(record['messages'][:-1],tokenize=True,add_generation_prompt=True)
                assert tokens[:offset]==prefix and 0<offset<len(tokens)
                assert tokenizer.decode(tokens[offset:]).startswith(record['messages'][-1]['content'])
                lengths.append(len(tokens)); completions.append(len(tokens)-offset)
        eval_lengths=[len(tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)) for r in read('eval.jsonl')]
        report['tokenizer']=dict(checked=True,model=MODEL.name,train_valid_records=len(lengths),eval_prompts=len(eval_lengths),min_tokens=min(lengths),max_tokens=max(lengths),max_completion_tokens=max(completions),max_eval_prompt_tokens=max(eval_lengths),recommended_max_seq_length=2048,template_sha256=sha(ROOT/'chat_template.jinja'),tokenizer_sha256=sha(MODEL/'tokenizer.json'))
        assert max(lengths)<2048
    (ROOT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
