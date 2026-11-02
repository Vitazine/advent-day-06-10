import json,unittest
from pathlib import Path
class VariantTests(unittest.TestCase):
 def test_same_pilot_examples_targets_only_system_differs(self):
  root=Path(__file__).resolve().parent
  original=[json.loads(s) for s in (root/'data/train.jsonl').read_text().splitlines()]
  indices=json.loads((root/'runs/pilot-b1/pilot-indices.json').read_text())
  variant=[json.loads(s) for s in (root/'short-prompt/data/train.jsonl').read_text().splitlines()]
  self.assertEqual(len(variant),16)
  for i,row in zip(indices,variant):
   self.assertEqual(row['messages'][1:],original[i]['messages'][1:])
   self.assertNotEqual(row['messages'][0],original[i]['messages'][0])
 def test_isolated_manifest_verifies(self):
  from runner_config import verify_manifest
  verify_manifest(Path(__file__).resolve().parent/'short-prompt')
