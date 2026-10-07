"""Standalone tests for LifeLink-AI research/decision-support utilities."""
import importlib.util, pathlib, sys, types, unittest
from types import SimpleNamespace
ROOT=pathlib.Path(__file__).parent/'app'
app=types.ModuleType('app'); app.__path__=[str(ROOT)]; sys.modules['app']=app
ml=types.ModuleType('app.ml'); ml.__path__=[str(ROOT/'ml')]; sys.modules['app.ml']=ml
for name,path in [('app.transport',ROOT/'transport.py'),('app.ml.engine',ROOT/'ml'/'engine.py'),('app.ai_lab',ROOT/'ai_lab.py')]:
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
from app.ai_lab import enhanced_match, dataset_quality, fairness_audit

class AILabTests(unittest.TestCase):
    def setUp(self):
        self.d=SimpleNamespace(organ='Kidney',blood_group='A+',age=36,hla_match=94,antibodies=10,medical_history=14,location='Pune',harvest_hours=4,status='AVAILABLE')
        self.r=SimpleNamespace(organ='Kidney',blood_group='A+',age=35,hla_match=80,antibodies=22,medical_history=20,waiting_days=240,location='Nashik',urgency='Urgent',status='REQUEST_ACTIVE')
    def test_multifactor_score_contains_confidence_priority(self):
        out=enhanced_match(self.d,self.r)
        self.assertIsNotNone(out)
        self.assertGreater(out['score'],0)
        self.assertGreater(out['confidence'],0)
        self.assertGreater(out['priority'],0)
        self.assertIn('transport_feasibility',out['breakdown'])
    def test_dataset_quality(self):
        q=dataset_quality(); self.assertEqual(q['rows'],3200); self.assertGreater(q['features'],5)
    def test_fairness_groups(self):
        rows=fairness_audit(); self.assertEqual(len(rows),3)

if __name__=='__main__': unittest.main(verbosity=2)
