import importlib.util
import json
import unittest
class ScoringChecks(unittest.TestCase):
    def score(self,rows,predictions):
        self.assertIsNotNone(importlib.util.find_spec('score_candidates'),'Missing evaluator')
        from score_candidates import score
        return score(rows,predictions)
    def rows(self):
        target=dict(status='ok',amount='100.00',currency='RUB',category='cafes',date='2026-11-01',payment_method=None,clarification_reason=None)
        rows=[]
        for i in range(4):rows.append(dict(id=str(i),family='group',tags=['contrast'],dimension='amount',input={'current_date':'2026-11-01'},target=dict(target)))
        rows[3]['target']={k:('ignored' if k=='status' else None) for k in target}
        return rows
    def test_order_does_not_change_exact_match(self):
        rows=self.rows();result=self.score(rows,{r['id']:json.dumps(dict(reversed(list(r['target'].items())))) for r in rows})
        self.assertEqual(result['exact_correct'],4);self.assertEqual(result['contrast_groups_all_correct'],1)
    def test_group_failure_and_false_expense_are_visible(self):
        rows=self.rows();pred={r['id']:json.dumps(r['target']) for r in rows};pred['3']=pred['0']
        result=self.score(rows,pred)
        self.assertEqual(result['exact_correct'],3);self.assertEqual(result['false_expenses_on_ignored'],1);self.assertEqual(result['contrast_groups_all_correct'],0)
    def test_bad_json_is_failure(self):
        rows=self.rows();pred={r['id']:json.dumps(r['target']) for r in rows};pred['0']='not json'
        result=self.score(rows,pred);self.assertEqual(result['invalid_responses'],1);self.assertEqual(result['exact_correct'],3)
    def test_missing_or_extra_ids_are_not_silently_skipped(self):
        rows=self.rows()
        with self.assertRaises(ValueError):self.score(rows,{})
if __name__=='__main__':unittest.main()
