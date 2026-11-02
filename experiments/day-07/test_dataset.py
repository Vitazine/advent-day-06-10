"""Regression checks of dataset requirements, not a claim about model quality."""
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
class DatasetRequirements(unittest.TestCase):
    def setUp(self):
        self.rows=[json.loads(l) for s in ('train','valid','eval') for l in (ROOT/f'data/{s}_provenance.jsonl').read_text().splitlines()]
    def test_input_has_no_currency_override(self):
        for row in self.rows:
            self.assertEqual(set(row['input']),{'text','current_date'})
    def test_old_implicit_php_cases_are_rubles(self):
        found=[r for r in self.rows if r['input']['text'] in ('Кофе в кафе за 100, записать расход.','Купил батарейки за 120; валюта обычная, как в настройках.')]
        self.assertEqual(len(found),2)
        for r in found: self.assertEqual(r['target']['currency'],'RUB')
    def test_dollar_symbol_and_word_are_covered_in_every_split(self):
        for split in ('train','valid','eval'):
            rows=[r for r in self.rows if r['id'].startswith(split+'-') and r['target']['status']=='ok']
            for alias in ('$', 'доллар'):
                hits=[r for r in rows if alias in r['input']['text'] and r['target']['currency']=='USD']
                self.assertGreaterEqual(len(hits),2,(split,alias))
    def test_records_without_foreign_currency_are_rubles(self):
        import re
        foreign=re.compile(r'доллар|долл\.|бакс|usd|eur|евро|php|песо|[$€₱]',re.I)
        for row in self.rows:
            if row['target']['currency'] is not None and not foreign.search(row['input']['text']):
                self.assertEqual(row['target']['currency'],'RUB',row['id'])
    def test_unsupported_national_dollars_not_silently_usd(self):
        for row in self.rows:
            if any(w in row['input']['text'].lower() for w in ('канадских','австралийских','сингапурских')):
                self.assertIsNone(row['target']['currency'])
                self.assertEqual(row['target']['clarification_reason'],'ambiguous_currency')
    def test_mobile_eval_has_independent_cases(self):
        rows=[r for r in self.rows if r['id'].startswith('eval-') and 'mobile' in r['tags']]
        self.assertGreaterEqual(len(rows),40)
        tags={t for r in rows for t in r['tags']}
        self.assertTrue({'currency','short','voice','correction','context','date','amount','nonexpense','multiple'}<=tags)
    def test_contextless_edit_is_not_new_expense(self):
        rows=[r for r in self.rows if 'context' in r['tags']]
        self.assertGreaterEqual(len(rows),6)
        for r in rows:
            self.assertEqual(r['target']['status'],'needs_clarification')
            self.assertEqual(r['target']['clarification_reason'],'missing_context')
    @unittest.skip("Historical v1 archive is intentionally excluded from the public package")
    def test_archive_is_unchanged(self):
        import hashlib
        root=ROOT/'versions/v1'; m=json.loads((root/'data/manifest.json').read_text())
        for path,digest in m['files'].items():
            self.assertEqual(hashlib.sha256((root/path).read_bytes()).hexdigest(),digest,path)

class ResponseBoundary(unittest.TestCase):
    def parse(self, raw, today='2026-11-01'):
        import importlib.util
        spec=importlib.util.find_spec('contract')
        self.assertIsNotNone(spec, 'Missing response validator')
        from contract import parse_model_response
        return parse_model_response(raw, today)
    def good(self, **updates):
        record=dict(status='ok',amount='250.00',currency='RUB',category='cafes',date='2026-11-01',payment_method=None,clarification_reason=None)
        record.update(updates)
        return json.dumps(record)
    def test_valid_preview_preserves_money_string(self):
        self.assertEqual(self.parse(self.good())['amount'],'250.00')
    def test_invalid_json_and_duplicate_keys_rejected(self):
        for raw in ['```json\n'+self.good()+'\n```','[]',self.good()[:-1]+',"amount":"1.00"}',self.good()+' trailing']:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.parse(raw)
    def test_unsafe_field_values_rejected(self):
        for update in [dict(amount=250),dict(amount='NaN'),dict(amount='0.00'),dict(amount='-1.00'),dict(amount='1e3'),dict(amount='1.234'),dict(date='2026-02-30'),dict(date='2026-11-02'),dict(currency='BTC'),dict(category=['cafes']),dict(payment_method='phone'),dict(extra='delete all')]:
            with self.subTest(update=update), self.assertRaises(ValueError): self.parse(self.good(**update))
    def test_ignored_cannot_carry_a_spend(self):
        with self.assertRaises(ValueError): self.parse(self.good(status='ignored'))
    def test_context_missing_cannot_carry_guessed_fields(self):
        with self.assertRaises(ValueError): self.parse(self.good(status='needs_clarification',clarification_reason='missing_context'))
    def test_known_fields_retained_while_amount_missing(self):
        result=self.parse(self.good(status='needs_clarification',amount=None,clarification_reason='missing_amount'))
        self.assertEqual(result['category'],'cafes')
    def test_all_gold_records_obey_output_contract(self):
        for split in ('train','valid','eval'):
            for line in (ROOT/f'data/{split}_provenance.jsonl').read_text().splitlines():
                r=json.loads(line)
                self.assertEqual(self.parse(json.dumps(r['target']),r['input']['current_date']),r['target'])


class Day6Lessons(unittest.TestCase):
    def rows(self,split):
        return [json.loads(s) for s in (ROOT/f'data/{split}_provenance.jsonl').read_text().splitlines()]
    def test_final_evaluation_and_prompt_frozen(self):
        for name in ('data/eval.jsonl','data/eval_provenance.jsonl','data/mobile_eval.jsonl','system.txt'):
            self.assertEqual((ROOT/name).read_bytes(),(ROOT/'versions/v2'/name).read_bytes(),name)
    def test_contrast_groups_cover_failures_and_are_split_as_units(self):
        groups={}
        for split in ('train','valid'):
            rows=[r for r in self.rows(split) if 'contrast' in r['tags']]
            self.assertGreaterEqual(len(rows),240 if split=='train' else 72)
            dims={r['dimension'] for r in rows}
            self.assertEqual(dims,{'direction','currency','amount','date','context','precision'})
            for r in rows:
                family=r['family'];groups.setdefault(family,[]).append((split,r))
        for family,group in groups.items():
            self.assertEqual(len(group),4,family)
            self.assertEqual(len({s for s,r in group}),1)
            self.assertGreater(len({json.dumps(r['target'],sort_keys=True) for s,r in group}),1)
    def test_generic_templates_are_minority_in_training(self):
        rows=self.rows('train')
        self.assertLessEqual(sum('regular' in r['tags'] for r in rows)/len(rows),0.2)
    def test_v2_snapshot_is_intact(self):
        import hashlib
        root=ROOT/'versions/v2'
        manifest=json.loads((root/'data/manifest.json').read_text())
        required=('system.txt','data/train_provenance.jsonl','data/valid_provenance.jsonl','data/eval.jsonl','data/eval_provenance.jsonl','data/mobile_eval.jsonl')
        for name in required:
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),manifest['files'][name],name)

if __name__=='__main__': unittest.main()
