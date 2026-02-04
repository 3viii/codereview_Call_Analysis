import sys
from unittest.mock import MagicMock

# MOCK PYDANTIC BEFORE IMPORTS
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def MockField(*args, **kwargs):
    return None

mock_pydantic = MagicMock()
mock_pydantic.BaseModel = MockBaseModel
mock_pydantic.Field = MockField
sys.modules["pydantic"] = mock_pydantic

# Mock other missing deps
sys.modules["soundfile"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["faster_whisper"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["vaderSentiment"] = MagicMock()
sys.modules["vaderSentiment.vaderSentiment"] = MagicMock()

import unittest
from src.pipeline import AudioPipeline
from src.config import CONFIG

class TestRoleRules(unittest.TestCase):
    def setUp(self):
        # Ensure we are in mock mode so we don't load heavy models
        CONFIG.use_api = "mock"
        self.pipeline = AudioPipeline()
        # We don't need to load resources for this test as we are calling _assign_roles directly
        
    def test_collector_keywords(self):
        print("\nTesting Collector Keywords...")
        turns = [
            {"speaker": "SPEAKER_01", "text": "This is a call regarding your loan payment which is due."},
            {"speaker": "SPEAKER_01", "text": "I am calling from the bank to confirm your account details."},
            {"speaker": "SPEAKER_02", "text": "Hello who is this?"}
        ]
        
        # SPEAKER_01 has "loan", "due", "calling from", "bank", "account" -> High Collector Score
        
        result_turns = self.pipeline._assign_roles(turns)
        
        spk1 = next(t for t in result_turns if t["speaker_id"] == "SPEAKER_01")
        spk2 = next(t for t in result_turns if t["speaker_id"] == "SPEAKER_02")
        
        print(f"SPEAKER_01 Role: {spk1.get('role')}")
        self.assertEqual(spk1.get("role"), "COLLECTOR")
        # If 01 is collector, 02 should be debtor by default logic
        self.assertEqual(spk2.get("role"), "DEBTOR")

    def test_debtor_keywords(self):
        print("\nTesting Debtor Keywords...")
        turns = [
            {"speaker": "SPEAKER_A", "text": "Hello are you there?"},
            {"speaker": "SPEAKER_B", "text": "Yes, I will pay the amount tomorrow."},
            {"speaker": "SPEAKER_B", "text": "My salary comes next week."}
        ]
        
        # SPEAKER_B has "i will pay", "tomorrow", "salary", "next week" -> High Debtor Score
        
        result_turns = self.pipeline._assign_roles(turns)
        
        spkA = next(t for t in result_turns if t["speaker_id"] == "SPEAKER_A")
        spkB = next(t for t in result_turns if t["speaker_id"] == "SPEAKER_B")
        
        print(f"SPEAKER_B Role: {spkB.get('role')}")
        self.assertEqual(spkB.get("role"), "DEBTOR")
        # Implies A is collector
        self.assertEqual(spkA.get("role"), "COLLECTOR")

    def test_mixed_strong_signals(self):
        print("\nTesting Mixed Strong Signals...")
        turns = [
            {"speaker": "AGENT", "text": "Calling from HDFC bank regarding your EMI."},
            {"speaker": "USER", "text": "I understand, I will pay the transfer tomorrow."}
        ]
        
        result_turns = self.pipeline._assign_roles(turns)
        
        agent = next(t for t in result_turns if t["speaker_id"] == "AGENT")
        user = next(t for t in result_turns if t["speaker_id"] == "USER")
        
        self.assertEqual(agent.get("role"), "COLLECTOR")
        self.assertEqual(user.get("role"), "DEBTOR")

    def test_no_keywords_fallback(self):
        print("\nTesting No Keywords (Fallback)...")
        # Without mock ML (which is mocked in pipeline unless loaded), this might return None or unpredictable if we don't mock the classifier.
        # However, we just want to ensure it doesn't crash and *attempts* to use standard logic.
        # Since we didn't mock the classifier in this test setup and pipeline.intent is None by default if not loaded,
        # it will prob hit "Zero-shot classifier not available" and mark all unknown.
        
        turns = [
            {"speaker": "S1", "text": "Hello how are you"},
            {"speaker": "S2", "text": "I am fine thanks"}
        ]
        
        result_turns = self.pipeline._assign_roles(turns)
        
        s1 = next(t for t in result_turns if t["speaker_id"] == "S1")
        # Should be None/Undetermined because no keywords and no ML loaded
        self.assertIsNone(s1.get("role"))


if __name__ == '__main__':
    unittest.main()
