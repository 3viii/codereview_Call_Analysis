import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from src.utils import get_logger

logger = get_logger(__name__)

class ToneAnalyzer:
    EMO_MAP = {
        "ang": "Angry",
        "hap": "Happy",
        "neu": "Neutral",
        "sad": "Sad",
        "fea": "Fear",
        "exc": "Excited"
    }

    def __init__(self, model_name: str = "superb/hubert-large-superb-er"):
        self.model_name = model_name
        self.fe = None
        self.model = None

    def load(self):
        try:
            logger.info(f"Loading tone analysis model: {self.model_name}")
            self.fe = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        except Exception as e:
            logger.warning(f"Tone model load failed: {e}")

    def analyze(self, audio_path: str) -> dict:
        """
        Returns: {"label": "angry", "label_pretty": "Angry", "score": float}
        """
        if not self.fe or not self.model:
             return {"label": "UNKNOWN", "label_pretty": "UNKNOWN", "score": 0.0}

        try:
            # Load 16k mono
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            # Cut to 10s
            max_len = 10 * 16000
            if len(audio) > max_len:
                audio = audio[:max_len]
            
            inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model(**inputs)
                probs = out.logits.softmax(dim=-1)[0].numpy()
            
            id2label = self.model.config.id2label
            best_id = int(np.argmax(probs))
            label = id2label[best_id]
            score = float(probs[best_id])
            
            pretty = self.EMO_MAP.get(label.lower(), label)
            return {"label": label, "label_pretty": pretty, "score": score}
        except Exception as e:
            logger.error(f"Tone analysis failed: {e}")
            return {"label": "UNKNOWN", "label_pretty": "UNKNOWN", "score": 0.0}
