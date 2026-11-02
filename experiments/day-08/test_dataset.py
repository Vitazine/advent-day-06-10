"""End-to-end contract checks: real generated artifacts, no model weights."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

def read(name):
    return [json.loads(line) for line in (ROOT / 'data' / name).read_text().splitlines()]

class DatasetContract(unittest.TestCase):
    def test_splits_and_hidden_targets(self):
        facts = {r['id']: r for r in read('facts.jsonl')}
        groups = {}
        for split in ('train', 'valid', 'final'):
            rows = read(split + '_provenance.jsonl')
            groups[split] = {facts[f]['split_group'] for r in rows for f in r['fact_ids']}
            wire = read(split + '.jsonl')
            self.assertEqual(len(rows), len(wire))
            for p, w in zip(rows, wire):
                if split == 'final':
                    self.assertEqual([m['role'] for m in w['messages']], ['system', 'user'])
                    self.assertNotIn('target', w)
                else:
                    self.assertEqual([m['role'] for m in w['messages']], ['system', 'user', 'assistant'])
                    self.assertEqual(w['messages'][-1]['content'], p['target'])
        for a,b in [('train','valid'),('train','final'),('valid','final')]:
            self.assertFalse(groups[a] & groups[b])

    def test_knowledge_transfer_intentional_overlap(self):
        train = read('train_provenance.jsonl')
        seen = {f for r in train for f in r['fact_ids']}
        prompts = {r['messages'][1]['content'] for r in read('train.jsonl')}
        dev = read('knowledge_dev_references.jsonl')
        final = read('knowledge_final_references.jsonl')
        self.assertFalse({r['fact_ids'][0] for r in dev} & {r['fact_ids'][0] for r in final})
        for split, refs in [('knowledge_dev',dev),('knowledge_final',final)]:
            for r,w in zip(refs,read(split+'.jsonl')):
                self.assertTrue(set(r['fact_ids']) <= seen)
                self.assertNotIn(w['messages'][1]['content'],prompts)
                self.assertEqual(len(w['messages']),2)
                self.assertNotIn('target',w)

    def test_review_regressions(self):
        f = next(f for f in read('facts.jsonl') if f['question'].startswith('Тускнеет ли митрил'))
        self.assertTrue(f['paraphrase'].startswith('Теряет ли'))
        for r in read('format_final_references.jsonl'):
            self.assertEqual(set(json.loads(r['target'])), {'answer','status'})
        refs = {r['id']: r for r in read('style_final_references.jsonl')}
        for r in read('style_final.jsonl'):
            self.assertNotIn(refs[r['id']]['target'], r['messages'][1]['content'])
            self.assertIn('Перепиши заметку',r['messages'][1]['content'])

    def test_general_not_domain_prompt(self):
        rows = read('general.jsonl')
        self.assertGreaterEqual(len(rows),40)
        for r in rows:
            self.assertEqual(len(r['messages']),2)
            self.assertNotIn('Толкин',r['messages'][0]['content'])
            self.assertNotIn('target',r)

    def test_preserve_approximate_date(self):
        rows = read('facts.jsonl')
        ring_date = next(r for r in rows if r['question']=='Когда Саурон выковал Единое Кольцо?')
        self.assertEqual(ring_date['date_precision'],'approximate_year')
        self.assertIn('Около',ring_date['answer'])
        self.assertIn('Второй эпохи',ring_date['answer'])

if __name__=='__main__': unittest.main()
