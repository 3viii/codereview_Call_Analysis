import os
import torch
from typing import List, Dict, Any
from src.config import CONFIG
from src.utils import get_logger

logger = get_logger(__name__)

class PyannoteDiarizer:
    def __init__(self):
        self.pipeline = None
        
    def load(self):
        token = CONFIG.hf_token or os.environ.get("INTELLISCORE_HF_TOKEN")
        if not token:
            logger.warning("No HuggingFace token found. Pyannote diarization will fail if model is gated.")
            # We continue anyway, maybe user has cached credentials
            
        try:
            from pyannote.audio import Pipeline
            logger.info("Loading Pyannote pipeline (pyannote/speaker-diarization-3.1)...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
            elif torch.backends.mps.is_available():
                 self.pipeline.to(torch.device("mps"))
                 
        except Exception as e:
            logger.error(f"Failed to load Pyannote pipeline: {e}")
            self.pipeline = None

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Returns list of segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
        """
        if not self.pipeline:
            logger.error("Pyannote pipeline not loaded.")
            return [{"speaker": "speaker_unknown", "start": 0.0, "end": 0.0}]

        if not os.path.exists(audio_path):
             logger.error(f"Audio file not found: {audio_path}")
             return []

        try:
            # Run inference
            diarization = self.pipeline(audio_path)
            
            segments = []
            # iterate over turns
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return [{"speaker": "speaker_error", "start": 0.0, "end": 0.0}]
