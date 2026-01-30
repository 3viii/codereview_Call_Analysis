from .interface import ASRInterface
from typing import Tuple, List, Dict, Any

class MockASR(ASRInterface):
    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        transcript = (
            "Hello, this is John from HDFC Bank. Am I speaking with Mr. Sharma? "
            "Yes, speaking. "
            "Sir, this call is regarding your overdue payment of 15,000 rupees. "
            "I know, I will pay next week by UPI. "
            "Okay, please do so by Monday. Thank you."
        )
        turns = [
            {"speaker": "Speaker 1", "start": 0.0, "end": 5.0, "text": "Hello, this is John from HDFC Bank. Am I speaking with Mr. Sharma?"},
            {"speaker": "Speaker 2", "start": 5.5, "end": 7.0, "text": "Yes, speaking."},
            {"speaker": "Speaker 1", "start": 7.5, "end": 12.0, "text": "Sir, this call is regarding your overdue payment of 15,000 rupees."},
            {"speaker": "Speaker 2", "start": 12.5, "end": 16.0, "text": "I know, I will pay next week by UPI."},
            {"speaker": "Speaker 1", "start": 16.5, "end": 20.0, "text": "Okay, please do so by Monday. Thank you."}
        ]
        return transcript, turns
