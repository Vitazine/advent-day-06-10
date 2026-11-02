import unittest
from core import make_config, strict_json, exact, guard_final
class CoreTests(unittest.TestCase):
    def test_real_updates(self):
        c=make_config('main',704)
        self.assertEqual(c['grad_accumulation_steps'],4)
        self.assertEqual(c['iters']//c['grad_accumulation_steps'],176)
        self.assertIsNone(c['resume_adapter_file'])
    def test_bad_steps(self):
        with self.assertRaises(ValueError): make_config('main',705)
    def test_json_rejects_duplicates(self):
        with self.assertRaises(ValueError): strict_json('{"answer":1,"answer":2}')
    def test_exact_never_substring(self):
        self.assertFalse(exact('Не Бард.','Бард.'))
        self.assertTrue(exact('Бард!','Бард.'))
    def test_final_rejects_empty_selection_and_wrong_weights(self):
        with self.assertRaises(RuntimeError):guard_final('final',{})
        selection={'frozen_before_final':True,'selected_adapter_sha256':'a'*64,'manifest_sha256':'b'*64}
        with self.assertRaises(RuntimeError):guard_final('final',selection,'c'*64,'b'*64)
        guard_final('final',selection,'a'*64,'b'*64)

    def test_final_gate(self):
        with self.assertRaises(RuntimeError): guard_final('knowledge_final',False)
        guard_final('knowledge_dev',False)
if __name__=='__main__': unittest.main()
