import unittest
from assessment import paired_stats

class AssessmentTests(unittest.TestCase):
    def test_ids_must_match(self):
        with self.assertRaises(ValueError):
            paired_stats([{'id':'a','pass':True}], [{'id':'b','pass':True}])

    def test_count_improvement_regression(self):
        a=[{'id':str(i),'pass':x} for i,x in enumerate([False, True, True, False])]
        b=[{'id':str(i),'pass':x} for i,x in enumerate([True, False, True, True])]
        result=paired_stats(a,b)
        self.assertEqual(result['improved'],2)
        self.assertEqual(result['regressed'],1)
        self.assertEqual(result['net_percentage_points'],25)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            paired_stats([{'id':'a','pass':True}]*2,[{'id':'a','pass':True}]*2)

if __name__=='__main__': unittest.main()
