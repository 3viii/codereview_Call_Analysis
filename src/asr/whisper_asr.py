import os
from typing import Tuple, List, Dict, Any
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from .interface import ASRInterface
from src.utils import get_logger

logger = get_logger(__name__)

class WhisperASR(ASRInterface):
    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "float32"):
        self.model = None
        if WhisperModel:
            try:
                logger.info(f"Loading Whisper model '{model_size}' on {device}...")
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
        else:
             logger.error("faster_whisper not installed.")

    def transcribe(self, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Returns (full_transcript, segments_with_timestamps).
        Note: The 'speaker' field in segments will be None or 'unknown'.
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded.")

        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments_raw, info = self.model.transcribe(audio_path)
        
        segments = []
        full_text_parts = []
        
        # segments_raw is a generator
        for seg in segments_raw:
            text = seg.text.strip()
            if not text:
                continue
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
                "speaker": "unknown" 
            })
            full_text_parts.append(text)
            
        full_transcript = " ".join(full_text_parts)
        return full_transcript, segments
