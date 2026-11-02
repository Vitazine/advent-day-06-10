"""Verify integrity, isolation, references and optional local Qwen tokenization."""
import argparse
import contextlib
import io
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unicodedata

from build_dataset import ROOT, SYSTEM, build
MODEL=Path(__import__('os').environ.get('MODEL_PATH', 'models/Qwen3-4B-Instruct-2507-4bit'))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold().replace('ё','е')))
def read(data,name): return [json.loads(x) for x in (data/(name+'.jsonl')).read_text().splitlines()]

def verify(data,tokenizer=False):
    manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items(): assert sha(data/name)==digest, f'Changed data: {name}'
    for name,digest in manifest['source_sha256'].items(): assert sha(ROOT/name)==digest, f'Changed source: {name}'
    facts=read(data,'facts'); byid={f['id']:f for f in facts}
    assert len(byid)==len(facts)
    sources={s['id']:s for s in json.loads((data/'sources.json').read_text())}
    questions=set()
    for f in facts:
        assert f['source_id'] in sources
        assert f['book_locator'].startswith(('HOBBIT:','LOTR:'))
        assert f['question'] and f['answer'] and f['paraphrase']
        assert norm(f['question']) != norm(f['paraphrase']),f['id']
        assert norm(f['question']) not in questions,f['id']
        questions.add(norm(f['question']))
        if f['category']=='dates' and f['date_precision']!='age':
            assert re.search(r'(Второй|Третьей) эпохи',f['answer']),f['id']
        assert f['baseline_knowledge']=='not_measured'
    splitgroups={s:{f['split_group'] for f in facts if f['split']==s} for s in ('train','valid','final')}
    for a,b in [('train','valid'),('train','final'),('valid','final')]: assert not splitgroups[a]&splitgroups[b],(a,b)
    all_messages=[]; prompts={}
    for split in ('train','valid','final'):
        rows=read(data,split); prov=read(data,split+'_provenance'); assert len(rows)==len(prov)
        prompts[split]=set()
        for r,p in zip(rows,prov):
            assert all(fid in byid and byid[fid]['split']==split for fid in p['fact_ids'])
            assert r['messages'][0]=={'role':'system','content':SYSTEM}
            expected=['system','user'] if split=='final' else ['system','user','assistant']
            assert [m['role'] for m in r['messages']]==expected
            assert 'target' not in r and 'fact_ids' not in r
            if split!='final': assert r['messages'][2]['content']==p['target']
            q=norm(r['messages'][1]['content']); assert q not in prompts[split]; prompts[split].add(q)
            if p['task']=='json':
                t=json.loads(p['target']); assert set(t)=={'answer','status'}
                assert t['status'] in {'known','not_specified'}
                assert (t['answer'] is None)==(t['status']=='not_specified')
            all_messages.append(r)
    for a,b in [('train','valid'),('train','final'),('valid','final')]: assert not prompts[a]&prompts[b]
    eval_counts={}
    for path in sorted(data.glob('*_references.jsonl')):
        name=path.stem.removesuffix('_references')
        rows=read(data,name); refs=read(data,name+'_references')
        assert len(rows)==len(refs) and len({r['id'] for r in rows})==len(rows)
        assert {r['id'] for r in rows}=={r['id'] for r in refs}
        for r in rows:
            assert set(r)=={'id','messages'}
            assert [m['role'] for m in r['messages']]==['system','user']
        for r in refs:
            assert 'target' in r
            assert all(fid in byid for fid in r.get('fact_ids',[]))
        all_messages.extend(rows); eval_counts[name]=len(rows)
    dev=read(data,'knowledge_dev_references'); final=read(data,'knowledge_final_references')
    assert not {r['split_group'] for r in dev}&{r['split_group'] for r in final}
    for split in ('knowledge_dev','knowledge_final'):
        for r,p in zip(read(data,split),read(data,split+'_references')):
            assert byid[p['fact_ids'][0]]['split']=='train'
            assert norm(r['messages'][1]['content']) not in prompts['train']
    for r in all_messages:
        for m in r['messages']:
            assert not re.search(r'<\|.*?\|>|</?think>',m['content'])
    report={'integrity':'passed','group_isolation':'passed','eval_targets_separated':True,
            'knowledge_fact_overlap':'intentional; learned facts, held-out wording',
            'counts':{s:len(read(data,s)) for s in ('train','valid','final')},
            'fact_records_by_category':dict(Counter(f['category'] for f in facts)),
            'train_records_by_category':dict(Counter(r['category'] for r in read(data,'train_provenance'))),
            'eval_counts':eval_counts,'tokenizer':{'checked':False},
            'limitations':['Source locators and secondary checks are not a full primary-text or human audit.',
                          'Template-derived JSON examples are not independent facts.',
                          'No baseline, training, throughput or knowledge gain measured.',
                          'Holdout groups can mention known entities; they hold out target facts, not all entity mentions.']}
    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()): build(Path(tmp))
        for name in manifest['sha256']: assert sha(Path(tmp)/name)==sha(data/name),name
        assert sha(Path(tmp)/'manifest.json')==sha(data/'manifest.json')
    report['deterministic_rebuild']='passed'
    if tokenizer:
        from mlx_lm.utils import load_tokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        tok=load_tokenizer(MODEL,tokenizer_config_extra={'local_files_only':True,'trust_remote_code':False})
        original_template=tok.chat_template
        approved_template=MODEL.parent.parent/"qwen4b-official-template.jinja"
        assert sha(ROOT/"chat_template.jinja")==sha(approved_template)
        tok.chat_template=(ROOT/'chat_template.jinja').read_text()
        lengths=[]; completions=[]; official_equal=True
        for split in ('train','valid'):
            for r in read(data,split):
                ids,offset=ChatDataset([r],tok,mask_prompt=True).process(r)
                prefix=tok.apply_chat_template(r['messages'][:-1],tokenize=True,add_generation_prompt=True)
                assert ids[:offset]==prefix and 0<offset<len(ids)
                assert tok.decode(ids[offset:]).startswith(r['messages'][-1]['content'])
                assert '<think>' not in tok.decode(ids)
                assert tok.decode(ids).endswith('<|im_end|>\n')
                lengths.append(len(ids)); completions.append(len(ids)-offset)
                if original_template:
                    official_ids=tok.apply_chat_template(r['messages'],tokenize=True,chat_template=original_template)
                    official_equal &= ids==official_ids
        elengths=[len(tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)) for r in all_messages if len(r['messages'])==2]
        assert max(lengths)<=256,(max(lengths),'training truncation')
        assert max(elengths)+128<=512,(max(elengths),'eval budget')
        report['tokenizer']={'checked':True,'model':MODEL.name,'train_valid_records':len(lengths),
            'min_tokens':min(lengths),'mean_tokens':round(sum(lengths)/len(lengths),2),
            'p95_tokens':sorted(lengths)[int(.95*(len(lengths)-1))],'max_tokens':max(lengths),
            'max_completion_tokens':max(completions),'max_eval_prompt_tokens':max(elengths),
            'recommended_max_seq_length':256,'chatdataset_mask_prefix':'passed',
            'same_tokens_as_model_template':official_equal,'tokenizer_sha256':sha(MODEL/'tokenizer.json'),
            'template_sha256':sha(ROOT/'chat_template.jinja')}
        report['tokenizer']['matches_day6_approved_non_thinking_template']=True
        report['tokenizer']['local_model_template_requires_override']=not official_equal
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=ROOT/'data'); p.add_argument('--tokenizer',action='store_true'); p.add_argument('--report',type=Path)
    a=p.parse_args(); result=verify(a.data,a.tokenizer)
    if a.report: a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
