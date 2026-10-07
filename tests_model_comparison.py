"""Checks for the four-organ model comparison pipeline."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('lifelink_organ_models', Path(__file__).parent/'app'/'ml'/'organ_models.py')
om=importlib.util.module_from_spec(spec); spec.loader.exec_module(om)

class ModelComparisonTests(unittest.TestCase):
    def test_four_organs_and_four_models(self):
        bundles=om.train_all()
        self.assertEqual(set(bundles), {'kidney','liver','heart','lung'})
        for task,bundle in bundles.items():
            self.assertEqual(len(bundle['metrics']),4)
            self.assertGreater(len({r['accuracy'] for r in bundle['metrics']}),1)
            self.assertNotIn('survtime_days', bundle['features'])
            self.assertIn(bundle['best_model'], [r['model'] for r in bundle['metrics']])
            self.assertEqual(bundle['target'],'match_status')

if __name__=='__main__': unittest.main(verbosity=2)
