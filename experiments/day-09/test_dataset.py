import unittest
from dataset_tools import validate_predictions, check_split_overlap

class DatasetTests(unittest.TestCase):
    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            validate_predictions([{'id':'a','input':{'current_date':'2026-11-02'}}],[{'id':'a','response':'{}'},{'id':'a','response':'{}'}])
    def test_missing_predictions_rejected(self):
        with self.assertRaises(ValueError):validate_predictions([{'id':'a'}],[])
    def test_normalized_overlap_rejected(self):
        a={'id':'a','family':'one','input':{'text':'Кофе   250','current_date':'2026-11-02'}}
        b={'id':'b','family':'two','input':{'text':'кофе 250','current_date':'2026-11-01'}}
        with self.assertRaises(ValueError):check_split_overlap({'train':[a],'final':[b]})
    def test_family_overlap_rejected(self):
        a={'id':'a','family':'one','input':{'text':'кофе 250'}}
        b={'id':'b','family':'one','input':{'text':'не покупал кофе'}}
        with self.assertRaises(ValueError):check_split_overlap({'train':[a],'final':[b]})
if __name__=='__main__':unittest.main()

class AuditTests(unittest.TestCase):
    def test_incomplete_audit_rejected(self):
        from dataset_tools import audited_rows
        with self.assertRaises(ValueError):audited_rows([{'id':'a','family':'g','tags':[]}],{'accepted_ids':[],'rejected':[]})
    def test_contrast_family_removed_together(self):
        from dataset_tools import audited_rows
        rows=[{'id':str(i),'family':'g','tags':['contrast']} for i in range(4)]
        kept,dropped=audited_rows(rows,{'accepted_ids':['0','1','2'],'rejected':[{'id':'3','reason':'bad'}]})
        self.assertEqual(kept,[])
        self.assertEqual(len(dropped),4)
