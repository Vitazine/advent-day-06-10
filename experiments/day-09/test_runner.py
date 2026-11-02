import json
import tempfile
import unittest
from pathlib import Path
import runner_config as cfg


class RunnerTests(unittest.TestCase):
    def test_pilots_equal_examples_updates(self):
        a = cfg.schedule(16, 1)
        b = cfg.schedule(16, 2)
        self.assertEqual((a['iters'], a['grad_accumulation_steps']), (16, 4))
        self.assertEqual((b['iters'], b['grad_accumulation_steps']), (8, 2))
        self.assertEqual(a['optimizer_updates'], 4)
        self.assertEqual(b['optimizer_updates'], 4)
        self.assertEqual(a['lr_schedule'], b['lr_schedule'])
        self.assertEqual(a['steps_per_eval'], 8)
        self.assertEqual(b['steps_per_eval'], 4)

    def test_epoch_must_not_drop_examples_or_updates(self):
        with self.assertRaises(ValueError): cfg.schedule(17, 2)
        self.assertEqual(cfg.schedule(400, 2)['optimizer_updates'], 100)

    def test_resume_required_and_architecture_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'adapters.safetensors'
            with self.assertRaises(FileNotFoundError): cfg.check_resume(p)
            p.write_bytes(b'synthetic adapter')
            source = dict(model=str(cfg.MODEL), num_layers=16, fine_tune_type='lora', lora_parameters=cfg.LORA)
            (p.parent / 'adapter_config.json').write_text(json.dumps(source))
            self.assertEqual(cfg.check_resume(p), cfg.sha(p))
            source['num_layers'] = 8
            (p.parent / 'adapter_config.json').write_text(json.dumps(source))
            with self.assertRaises(ValueError): cfg.check_resume(p)

    def test_duplicate_ids_and_targets_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'inputs.jsonl'
            p.write_text('\n'.join(json.dumps({'id':'x','input':{}}) for _ in range(2)))
            with self.assertRaises(ValueError): cfg.read_inputs(p)
            p.write_text(json.dumps({'id':'x','input':{},'target':'secret'}))
            with self.assertRaises(ValueError): cfg.read_inputs(p)

    def test_outputs_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'run'
            cfg.reserve_output(p)
            with self.assertRaises(FileExistsError): cfg.reserve_output(p)
            f = p / 'result.json'
            cfg.write_new(f, {'first': True})
            with self.assertRaises(FileExistsError): cfg.write_new(f, {'second': True})
            self.assertEqual(json.loads(f.read_text()), {'first':True})

    def test_truncation_and_empty_completion_rejected(self):
        self.assertEqual(cfg.length_report([([1]*2048, 2040)])['max'], 2048)
        with self.assertRaises(ValueError): cfg.length_report([([1]*2049, 2040)])
        with self.assertRaises(ValueError): cfg.length_report([([1]*10, 10)])

if __name__ == '__main__': unittest.main()

class ManifestTests(unittest.TestCase):
    def test_changed_frozen_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data.json').write_text('original')
            (root/'manifest.json').write_text(json.dumps({'files':{'data.json':cfg.sha(root/'data.json')}}))
            cfg.verify_manifest(root)
            (root/'data.json').write_text('modified')
            with self.assertRaises(ValueError):cfg.verify_manifest(root)
