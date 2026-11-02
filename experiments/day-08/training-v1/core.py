"""CPU-only run contracts and conservative diagnostics."""
import hashlib,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DAY=ROOT.parent
# Same pinned local weights used by Days 6/7; never mutate them.
MODEL=Path(__import__('os').environ.get('MODEL_PATH', 'models/Qwen3-4B-Instruct-2507-4bit'))
def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
def verify():
    m=json.loads((DAY/'data/manifest.json').read_text())
    for n,h in m['sha256'].items():
        if sha(DAY/'data'/n)!=h: raise ValueError('Changed data: '+n)
    for n,h in m['source_sha256'].items():
        if sha(DAY/n)!=h: raise ValueError('Changed dataset source: '+n)
    return sha(DAY/'data/manifest.json')
def guard_final(split,selection,adapter_hash=None,manifest_hash=None):
    if split!='final' and not split.endswith('_final'): return
    if not isinstance(selection,dict) or selection.get('frozen_before_final') is not True:
        raise RuntimeError('Final is locked until selection is frozen')
    selected=selection.get('selected_adapter_sha256','')
    if not re.fullmatch(r'[0-9a-f]{64}',selected) or selection.get('manifest_sha256')!=manifest_hash:
        raise RuntimeError('Invalid selection or dataset hash')
    allowed={selected,selection.get('diagnostic_adapter_sha256')}
    if adapter_hash is not None and adapter_hash not in allowed:
        raise RuntimeError('Adapter differs from frozen selection')
def make_config(name,steps):
    if steps<=0 or steps%4: raise ValueError('Steps must be positive and divisible by 4')
    pilot=name.startswith('pilot')
    updates=steps//4;warmup=min(8,max(1,updates//10))
    return {'model':str(MODEL),'data':str(DAY/'data'),'train':True,'fine_tune_type':'lora',
      'resume_adapter_file':None,'num_layers':16,'batch_size':1,'grad_accumulation_steps':4,
      'iters':steps,'learning_rate':1e-4,'optimizer':'adamw','optimizer_config':{'adamw':{'weight_decay':0.01}},
      'max_seq_length':256,'mask_prompt':True,'grad_checkpoint':False,'seed':42,
      'steps_per_eval':steps if pilot else 176,'save_every':steps if pilot else 176,
      'steps_per_report':10 if pilot else 32,'val_batches':-1,
      'adapter_path':str(ROOT/'runs'/name/'adapter'),
      'lora_parameters':{'keys':['self_attn.q_proj','self_attn.v_proj'],'rank':16,'scale':2.0,'dropout':0.0},
      'lr_schedule':{'name':'cosine_decay','arguments':[1e-4,max(1,updates-warmup),1e-5],
                     'warmup':warmup,'warmup_init':1e-5}}
def strict_json(s):
    def pairs(kvs):
        d={}
        for k,v in kvs:
            if k in d: raise ValueError('Duplicate key')
            d[k]=v
        return d
    return json.loads(s,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFC',s).casefold().replace('ё','е')))
def exact(a,b):return norm(a)==norm(b)
def rows_for(split):
    rows=read(DAY/'data'/f'{split}.jsonl')
    if split=='valid':
        refs=read(DAY/'data/valid_provenance.jsonl')
        return [{'id':p['id'],'messages':r['messages'][:2]} for r,p in zip(rows,refs)]
    return rows

def references(split):
    name=split+'_provenance' if split in ('valid','final') else split+'_references'
    return {r['id']:r for r in read(DAY/'data'/f'{name}.jsonl')}
def score(split,rows):
    refs=references(split)
    if len(rows)!=len(refs) or {r['id'] for r in rows}!=set(refs):raise ValueError('Missing/duplicate prediction IDs')
    outcomes={}; schema={}
    for r in rows:
        t=refs[r['id']]['target'];a=r['answer'];ok=False
        try:
            if t.startswith(('{','[')):
                actual=strict_json(a);expected=strict_json(t)
                ok=(actual==expected and json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True))
                if isinstance(expected,dict) and set(expected)=={'answer','status'}:
                    schema[r['id']]=(isinstance(actual,dict) and set(actual)=={'answer','status'} and
                        ((actual.get('status')=='known' and isinstance(actual.get('answer'),str)) or
                         (actual.get('status')=='not_specified' and actual.get('answer') is None)))
            elif split=='general':ok=a.strip()==t.strip()
            else:ok=exact(a,t)
        except (ValueError,TypeError):pass
        outcomes[r['id']]=bool(ok and r.get('finish_reason')!='length')
    return {'n':len(rows),'normalized_exact':sum(outcomes.values()),'outcomes':outcomes,
            'schema_valid':sum(schema.values()),'schema_n':len(schema),
            'truncated':sum(r.get('finish_reason')=='length' for r in rows),
            'note':'Conservative lexical diagnostic; non-matches require semantic review, style cannot be judged by exact match.'}
