import tempfile,unittest,json
from pathlib import Path
import build_report
class ReportTests(unittest.TestCase):
 def test_no_report_for_running_training(self):
  old=build_report.ROOT
  try:
   with tempfile.TemporaryDirectory() as d:
    build_report.ROOT=Path(d);(Path(d)/'status.json').write_text(json.dumps({'state':'running'}))
    with self.assertRaises(RuntimeError):build_report.main()
    self.assertFalse((Path(d)/'final-report.md').exists())
  finally:build_report.ROOT=old
 def test_failed_quality_is_reported_as_keep_before(self):
  old=build_report.ROOT
  try:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);build_report.ROOT=p
    def put(name,obj):
     f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(obj))
    put('status.json',{'state':'complete'})
    put('selection.json',{'selected_after':'iteration-1','improvement_repeated_across_seeds':False,'candidates':[{'name':'iteration-1','exact_correct':1}]})
    put('recommendation.json',{'recommended':'before','new_final_false_expense_ids':['a']})
    for label in ('before','after','teacher'):
     put(f'evaluation/{label}-final-scores.json',{'n':1,'exact_correct':0 if label=='after' else 1,'invalid_responses':0,'false_expenses_on_ignored':1 if label=='after' else 0,'false_expenses_on_missing_context':0,'outcomes':{'a':label!='after'}})
     if label!='teacher':put(f'evaluation/{label}-final/responses.jsonl',{'seconds':2.0})
    put('runs/iteration-1/summary.json',{'wall_seconds':60,'peak_memory_gb':6,'optimizer_updates':99})
    build_report.main();report=(p/'final-report.md').read_text()
    self.assertIn('сохранить исходный адаптер',report)
    self.assertIn('регрессий: 1',report)
  finally:build_report.ROOT=old
