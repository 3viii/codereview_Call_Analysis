from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

class ASRInterface(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Transcribe the audio file.
        Returns:
            transcript (str): The full text transcript.
            turns (List[Dict]): List of turns, e.g. [{"speaker": "Speaker 0", "start": 0.0, "end": 1.0, "text": "..."}]
        """
        pass
