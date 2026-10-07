"""Static integrity checks for the end-to-end workflow implementation."""
from pathlib import Path
import ast, csv, unittest
ROOT=Path(__file__).parent

class WorkflowIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.routes=(ROOT/'app'/'routes.py').read_text()
        cls.models=(ROOT/'app'/'models.py').read_text()
    def test_match_request_entity_exists(self):
        self.assertIn('class MatchRequest', self.models)
        for field in ['recipient_status','donor_status','doctor_status','hospital_status','admin_status','final_status']:
            self.assertIn(field, self.models)
    def test_required_backend_actions_exist(self):
        for text in ['check_waiting_recipient_requests','can_finalize_match','/donor/<int:did>/approval','/recipient/<int:rid>/approval','/match-request/<int:reqid>/donor-decision','/match-request/<int:reqid>/approval','/transport/<int:tid>/authorize']:
            self.assertIn(text, self.routes)
    def test_route_requires_final_approval(self):
        section=self.routes.split("@bp.route('/route/<int:mid>'",1)[1].split("@bp.route('/transport/<int:tid>/reoptimize'",1)[0]
        self.assertIn("workflow_state!='FINAL_APPROVED'",section)
    def test_four_organ_datasets_have_schema_and_5000_rows(self):
        for organ in ['kidney','liver','heart','lung']:
            path=ROOT/'datasets'/organ/f'{organ}_matching_synthetic_academic.csv'
            self.assertTrue(path.exists())
            with path.open(newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),5000)
            self.assertIn('match_status',rows[0])
            self.assertIn('donor_blood',rows[0])
            self.assertIn('recipient_blood',rows[0])

if __name__=='__main__': unittest.main(verbosity=2)
