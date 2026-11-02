"""Verify finished runs and evaluation artifacts without judging text quality."""
import json
import numpy as np
from safetensors.numpy import load_file
from common import *

manifest=json.loads((ROOT/'data/manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert sha(ROOT/'data'/name)==digest,name
for name,info in manifest['model']['files'].items():
    assert sha(MODEL/name)==info['sha256'],name
assert sha(ROOT/'chat_template.jinja')==manifest['template_sha256']
data_hashes=[]
runs=[]
for i in range(1,5):
    root=ROOT/'runs'/f'iteration-{i}'
    config=json.loads((root/'config.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert config['resume_adapter_file'] is None
    assert config['iters']==2048 and config['batch_size']==1 and config['grad_accumulation_steps']==4
    assert provenance['optimizer_updates']==512 and provenance['nominal_epochs']==1
    assert sha(root/'adapter/adapters.safetensors')==summary['adapter_sha256']
    tensors=load_file(root/'adapter/adapters.safetensors')
    assert all(np.isfinite(t).all() for t in tensors.values())
    assert any(np.any(t!=0) for n,t in tensors.items() if n.endswith('lora_b'))
    for name,digest in provenance['training_files_sha256'].items():
        assert sha(root/'data'/name)==digest
    data_hashes.append(provenance['training_files_sha256'])
    metrics=read_jsonl(root/'metrics.jsonl')
    assert [m for m in metrics if m['kind']=='train'][-1]['iteration']==2048
    runs.append({'name':f'iteration-{i}','tensor_count':len(tensors),'all_finite':True,'steps':2048})
assert all(x==data_hashes[0] for x in data_hashes)
selection=json.loads((ROOT/'selection.json').read_text())
selected=selection['selected']
assert sha(ROOT/'runs'/selected/'adapter/adapters.safetensors')==selection['adapter_sha256']
assert sha(ROOT/'data/final_eval.jsonl')==selection['final_input_sha256']
for split,count,names in [('dev',24,['base',*[f'iteration-{i}' for i in range(1,5)]]),('final',40,['base',selected]),('general',8,['base',selected])]:
    source=read_jsonl(ROOT/'data'/f'{split}_eval.jsonl')
    assert len(source)==count
    for name in names:
        output=ROOT/'evaluation'/f'{name}-{split}.jsonl'
        rows=read_jsonl(output)
        meta=json.loads(output.with_suffix('.meta.json').read_text())
        assert len(rows)==count
        assert meta['temperature']==0 and meta['max_tokens']==128
        assert meta['data_sha256']==sha(ROOT/'data'/f'{split}_eval.jsonl')
        for inp,out in zip(source,rows):
            assert inp['id']==out['id'] and inp['messages']==out['messages']
            assert out['answer'].strip()
        if name!='base':
            assert meta['adapter_sha256']==sha(ROOT/'runs'/name/'adapter/adapters.safetensors')
result={'status':'PASS','runs':runs,'identical_training_data':True,'model_hashes_unchanged':True,
        'generation_counts':{'dev':120,'final':80,'general':16},'selected':selected,
        'limits':'Does not independently establish semantic correctness, absence of semantic duplicates, or unbiased annotation.'}
write_json(ROOT/'integrity.json',result)
print(json.dumps(result,ensure_ascii=False))
