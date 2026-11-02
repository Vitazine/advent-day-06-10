import json,tempfile,unittest
from pathlib import Path
from pipeline import dangers
class DangerTests(unittest.TestCase):
 def test_nonobject_response_does_not_crash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'r').write_text(json.dumps({'id':'a','target':{'status':'ignored','clarification_reason':None}}))
   (p/'p').write_text(json.dumps({'id':'a','response':'[]'}))
   self.assertEqual(dangers(p/'p',p/'r'),set())
 def test_detect_false_expense_on_context(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'r').write_text(json.dumps({'id':'a','target':{'status':'needs_clarification','clarification_reason':'missing_context'}}))
   (p/'p').write_text(json.dumps({'id':'a','response':'{"status":"ok"}'}))
   self.assertEqual(dangers(p/'p',p/'r'),{'a'})

class RecommendationTests(unittest.TestCase):
 def test_final_new_false_expense_blocks_after(self):
  from pipeline import recommend_after
  self.assertFalse(recommend_after(40,50,{'a'},{'b'}))
 def test_final_quality_regression_blocks_after(self):
  from pipeline import recommend_after
  self.assertFalse(recommend_after(50,49,set(),set()))
 def test_improvement_without_new_false_expense(self):
  from pipeline import recommend_after
  self.assertTrue(recommend_after(40,50,{'a'},set()))
