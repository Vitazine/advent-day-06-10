"""Deterministic Day 8 corpus. No network, generation, or model weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
SEED = 20261102
SYSTEM = (ROOT/'system.txt').read_text().strip()
CARD = '\nВерни только JSON с ключами answer (строка или null) и status ("known" или "not_specified"). Если сведения не указаны в книгах, answer=null.'

def dumps(x): return json.dumps(x, ensure_ascii=False, separators=(',', ':'))
def write_rows(path, rows): path.write_text(''.join(dumps(r)+'\n' for r in rows))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def messages(q, a=None, system=SYSTEM):
    m=[{'role':'system','content':system},{'role':'user','content':q}]
    if a is not None: m.append({'role':'assistant','content':a})
    return m

def facts():
    result=[]
    for line in (ROOT/'facts_seed.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        if line.startswith('@ '):
            group,entity,category,slug,locator=line[2:].split('|')
            split='valid' if group.startswith('valid_') else 'final' if group.startswith('final_') else 'train'
            group=group.replace('description_', '').replace('_description','').replace('unknown_', '')
            # Merge descriptions and equivalent claims into their semantic group.
            group={'ents':'treebeard','smaug':'bard','smaugdate':'bard','asfaloth':'glorfindel',
                   'ring':'ring','destruction':'ring','partydate':'bilbo','rohanhorse':'gandalf',
                   'sarumandate':'saruman','fellowship':'departure'}.get(group,group)
            continue
        question,answer,paraphrase=line.split('|')
        precision=None
        if category=='dates':
            precision='approximate_year' if 'Около' in answer else 'day' if re.search(r'\d+ (января|марта|мая|сентября|октября|ноября|декабря)',answer) else 'year' if 'году' in answer else 'age'
        result.append(dict(id='f-'+hashlib.sha256(question.encode()).hexdigest()[:12],
            split=split,split_group=group,entity=entity,category=category,
            question=question,answer=answer,paraphrase=paraphrase,
            source_id=slug,book_locator=locator,date_precision=precision,
            evidence_status='editorial_scope_limit' if category=='unknown' else 'curated_with_secondary_reference',
            baseline_knowledge='not_measured'))
    return result

def build(out):
    if any(out.iterdir()):
        raise SystemExit('Output directory must be empty; use --output for a new version. Frozen data is never overwritten.')
    fs=facts(); write_rows(out/'facts.jsonl',fs)
    sources=[]
    for slug in sorted({f['source_id'] for f in fs}):
        sources.append({'id':slug,'url':'https://tolkiengateway.net/wiki/'+slug,
            'role':'secondary_reference','primary_text_opened':False,
            'book_locators':sorted({f['book_locator'] for f in fs if f['source_id']==slug}),
            'note':'Справочная ссылка и книжный locator. Не означает построчную сверку с полным текстом книги; см. source_audit.md.'})
    (out/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    for split in ('train','valid','final'):
        pairs=[]
        for f in fs:
            if f['split']!=split: continue
            modes=['qa'] if f['category']=='corrections' else ['qa','json']
            for mode in modes:
                q=f['question']+(CARD if mode=='json' else '')
                a=f['answer'] if mode=='qa' else dumps({'answer':None if f['category']=='unknown' else f['answer'], 'status':'not_specified' if f['category']=='unknown' else 'known'})
                rid=f'{split}-{f["id"]}-{mode}'
                row={'messages':messages(q,a)} if split!='final' else {'id':rid,'messages':messages(q)}
                prov={'id':rid,'fact_ids':[f['id']], 'split_group':f['split_group'], 'category':f['category'],'task':mode,'target':a}
                pairs.append((row,prov))
        random.Random(SEED).shuffle(pairs)
        write_rows(out/f'{split}.jsonl',[p[0] for p in pairs])
        write_rows(out/f'{split}_provenance.jsonl',[p[1] for p in pairs])
    # Hold out paraphrase questions for learned facts; groups split dev/final.
    train_facts=[f for f in fs if f['split']=='train' and f['category'] not in {'corrections','unknown','descriptions'}]
    groups=sorted({f['split_group'] for f in train_facts})
    random.Random(SEED).shuffle(groups)
    dev_groups=set(groups[:max(1,len(groups)//3)])
    for name, is_dev in [('knowledge_dev',True),('knowledge_final',False)]:
        picked=[f for f in train_facts if (f['split_group'] in dev_groups)==is_dev]
        write_rows(out/(name+'.jsonl'),[{'id':name+'-'+f['id'],'messages':messages(f['paraphrase'])} for f in picked])
        write_rows(out/(name+'_references.jsonl'),[{'id':name+'-'+f['id'],'fact_ids':[f['id']],'category':f['category'],'target':f['answer'],'context':f['question']+' '+f['answer'],'split_group':f['split_group']} for f in picked])
        # Exact training questions are a separate diagnostic, not an independent eval.
        recall=name.replace('knowledge','recall')
        write_rows(out/(recall+'.jsonl'),[{'id':recall+'-'+f['id'],'messages':messages(f['question'])} for f in picked])
        write_rows(out/(recall+'_references.jsonl'),[{'id':recall+'-'+f['id'],'fact_ids':[f['id']],'target':f['answer'],'category':f['category']} for f in picked])
    # Learned correction/abstention behavior, in held-out wording.
    for split in ('dev','final'):
        picked=[f for f in fs if f['split']=='train' and f['category'] in {'corrections','unknown'}
                and ((f['split_group'] in dev_groups)==(split=='dev'))]
        name='robustness_'+split
        write_rows(out/(name+'.jsonl'),[{'id':name+'-'+f['id'],'messages':messages(f['paraphrase'])} for f in picked])
        write_rows(out/(name+'_references.jsonl'),[{'id':name+'-'+f['id'],'fact_ids':[f['id']],
            'target':f['answer'],'category':f['category'],'split_group':f['split_group']} for f in picked])
    # Known factual context makes format measurement independent of memory.
    for split in ('valid','final'):
        held=[f for f in fs if f['split']==split]
        for task in ('format','brevity'):
            rows=[]; refs=[]
            selected=held+[f for f in fs if task=='format' and f['category']=='unknown' and ((f['split_group'] in dev_groups)==(split=='valid'))]
            for f in selected:
                rid=f'{task}_{split}-{f["id"]}'
                context=f'Справка: {f["question"]} {f["answer"]}'
                if task=='format':
                    q=context+'\nОтветь на вопрос по справке.'+CARD
                    target=dumps({'answer':None if f['category']=='unknown' else f['answer'],'status':'not_specified' if f['category']=='unknown' else 'known'})
                else:
                    q=context+'\nОтветь на вопрос из справки кратко, нейтрально, без вступлений, списков и новых сведений. Не более двух предложений.'
                    target=f['answer']
                rows.append({'id':rid,'messages':messages(q)})
                refs.append({'id':rid,'fact_ids':[f['id']],'target':target,'category':f['category'],'context_provided':True})
            write_rows(out/f'{task}_{split}.jsonl',rows)
            write_rows(out/f'{task}_{split}_references.jsonl',refs)
    # Tone/style transformation uses independently authored informal notes.
    style_rows={s:[] for s in ('train','valid','final')}; style_refs={s:[] for s in style_rows}
    for line in (ROOT/'style_seed.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        split,group,note,target=line.split('|')
        linked=[f['id'] for f in fs if f['split_group']==group and f['split']==split]
        assert linked,group
        rid=f'style_{split}-{len(style_rows[split])+1:03}'
        q='Перепиши заметку нейтрально и кратко, сохрани все факты. Не более двух предложений, без вступления. Заметка: '+note
        style_rows[split].append({'id':rid,'messages':messages(q)})
        style_refs[split].append({'id':rid,'fact_ids':linked,'split_group':group,'category':'style',
                                 'task':'style','target':target,'context_provided':True})
    for split in ('valid','final'):
        write_rows(out/f'style_{split}.jsonl',style_rows[split])
        write_rows(out/f'style_{split}_references.jsonl',style_refs[split])
    train=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
    prov=[json.loads(x) for x in (out/'train_provenance.jsonl').read_text().splitlines()]
    for r,p in zip(style_rows['train'],style_refs['train']):
        train.append({'messages':r['messages']+[{'role':'assistant','content':p['target']}]}); prov.append(p)
    pairs=list(zip(train,prov)); random.Random(SEED+1).shuffle(pairs)
    write_rows(out/'train.jsonl',[p[0] for p in pairs]); write_rows(out/'train_provenance.jsonl',[p[1] for p in pairs])
    general=[]; refs=[]
    for line in (ROOT/'general_seed.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        category,q,a=line.split('|'); rid=f'general-{len(general)+1:03}'
        general.append({'id':rid,'messages':messages(q,system='Выполни задание пользователя. Соблюдай указанный формат ответа.')})
        refs.append({'id':rid,'target':a,'category':category,'scoring':'json_equal' if q.startswith('Верни JSON') else 'normalized_exact'})
    write_rows(out/'general.jsonl',general); write_rows(out/'general_references.jsonl',refs)
    # Baseline screen is separate; no unknown-to-base claim is made without generation.
    write_rows(out/'baseline_screen.jsonl',[{'id':f['id'],'messages':messages(f['question'])} for f in fs if f['split']=='train' and f['category'] not in {'corrections','unknown','descriptions'}])
    write_rows(out/'baseline_screen_references.jsonl',[{'id':f['id'],'target':f['answer'],'category':f['category']} for f in fs if f['split']=='train' and f['category'] not in {'corrections','unknown','descriptions'}])
    audit=random.Random(817).sample([f for f in fs if f['split']=='train'],30)
    write_rows(out/'audit_sample.jsonl',audit)
    manifest={'version':'1.0','seed':SEED,'baseline_run':False,'training_run':False,
              'facts':len(fs),'facts_by_split':dict(Counter(f['split'] for f in fs)),
              'rows':{p.stem:sum(1 for _ in p.open()) for p in sorted(out.glob('*.jsonl'))},
              'sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},
              'source_sha256':{p.name:digest(p) for p in [ROOT/'facts_seed.txt',ROOT/'general_seed.txt',ROOT/'style_seed.txt',ROOT/'build_dataset.py',ROOT/'system.txt',ROOT/'chat_template.jinja']}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'facts':manifest['facts'],'facts_by_split':manifest['facts_by_split'],'rows':manifest['rows']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=ROOT/'data'); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); build(args.output)
