
import unittest
from unittest.mock import MagicMock
from src.pipeline import AudioPipeline
from src.analysis.intent import IntentClassifier

class TestMLRoles(unittest.TestCase):
    def setUp(self):
        self.pipeline = AudioPipeline()
        # Mock the intent classifier and its inner pipeline
        self.pipeline.intent = MagicMock(spec=IntentClassifier)
        self.pipeline.intent.classifier = MagicMock()
        
    def test_assign_roles_collector_debtor(self):
        """Test standard scenario: One speaker is clearly collector."""
        AGENT_LABEL = "Debt collection agent calling about a loan"
        CUSTOMER_LABEL = "Customer responding about repayment"
        
        def classifier_side_effect(text, **kwargs):
            if "bank" in text or "payment" in text:
                return {"labels": [AGENT_LABEL, CUSTOMER_LABEL], "scores": [0.9, 0.1]}
            else:
                return {"labels": [CUSTOMER_LABEL, AGENT_LABEL], "scores": [0.8, 0.2]}
                
        self.pipeline.intent.classifier.side_effect = classifier_side_effect
        
        turns = [
            {"speaker": "SPEAKER_00", "text": "Hello this is HDFC bank payment department."},
            {"speaker": "SPEAKER_01", "text": "I am busy right now."},
            {"speaker": "SPEAKER_01", "text": "I will pay later."}
        ]
        
        result = self.pipeline._assign_roles(turns)
        
        self.assertEqual(result[0]["role"], "collector")
        self.assertEqual(result[0]["speaker"], "COLLECTOR")
        self.assertEqual(result[1]["role"], "debtor")
        self.assertEqual(result[1]["speaker"], "DEBTOR")
        
    def test_assign_roles_ambiguous(self):
        """Test ambiguous scenario: No clear collector signal."""
        AGENT_LABEL = "Debt collection agent calling about a loan"
        CUSTOMER_LABEL = "Customer responding about repayment"
        
        # Both look like customers (low agent prob)
        def classifier_side_effect(text, **kwargs):
            return {"labels": [CUSTOMER_LABEL, AGENT_LABEL], "scores": [0.6, 0.4]}
            
        self.pipeline.intent.classifier.side_effect = classifier_side_effect
        
        turns = [
            {"speaker": "SPEAKER_00", "text": "Hello."},
            {"speaker": "SPEAKER_01", "text": "Hi."}
        ]
        
        result = self.pipeline._assign_roles(turns)
        
        # Max agent score is 0.4.
        # Since >1 speakers, logic currently picks best candidate regardless of threshold.
        # This is strictly following "Assign COLLECTOR to speaker with higher agent probability".
        # If both are 0.4, it picks the first one? Or sorts stable?
        # If scores are identical, Python's sort is stable.
        self.assertIsNotNone(result[0]["role"])
        
    def test_single_speaker_low_conf(self):
        """Test single speaker with low confidence -> Unknown."""
        AGENT_LABEL = "Debt collection agent calling about a loan"
        CUSTOMER_LABEL = "Customer responding about repayment"

        def classifier_side_effect(text, **kwargs):
            # High customer score, low agent score
            return {"labels": [CUSTOMER_LABEL, AGENT_LABEL], "scores": [0.9, 0.1]}
            
        self.pipeline.intent.classifier.side_effect = classifier_side_effect
         
        turns = [
            {"speaker": "SPEAKER_00", "text": "Just talking to myself."}
        ]
        
        result = self.pipeline._assign_roles(turns)
        
        # Single speaker logic requires score > 0.6
        self.assertIsNone(result[0]["role"])
        self.assertIn("(undetermined)", result[0]["speaker"])

if __name__ == "__main__":
    unittest.main()
