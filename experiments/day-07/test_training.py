import importlib.util
import unittest
class TrainingProtocol(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('training_common'))
        import training_common
        return training_common
    def test_epoch_and_mask(self):
        c=self.module().configuration('iteration-1',5e-5,42,False)
        self.assertEqual(c['iters'],420);self.assertEqual(c['grad_accumulation_steps'],4)
        self.assertEqual(c['batch_size'],1);self.assertEqual(c['max_seq_length'],2048)
        self.assertTrue(c['mask_prompt']);self.assertIsNone(c['resume_adapter_file'])
        self.assertEqual(c['lora_parameters']['scale'],2.0)
    def test_wide_projections_and_seed(self):
        c=self.module().configuration('iteration-4',1e-4,43,True)
        self.assertEqual(len(c['lora_parameters']['keys']),7);self.assertEqual(c['seed'],43)
    def test_selection_prefers_accuracy_then_fewer_dangerous_errors(self):
        m=self.module()
        a={'exact_correct':100,'false_expenses_on_ignored':1,'false_expenses_on_missing_context':0,'outcomes':{'x':True}}
        b={**a,'false_expenses_on_ignored':0}
        self.assertGreater(m.selection_key(b,{},False,5e-5),m.selection_key(a,{},False,5e-5))
    def test_manifest_is_verified(self):
        self.assertEqual(self.module().verify_data()['version'],'3')
    def test_stage_logs_are_unique_across_eval_splits(self):
        import pipeline_v3
        self.assertTrue(hasattr(pipeline_v3,'stage_log_name'))
        self.assertNotEqual(pipeline_v3.stage_log_name(['infer','--name','base','--split','valid']),pipeline_v3.stage_log_name(['infer','--name','base','--split','eval']))
if __name__=='__main__':unittest.main()
