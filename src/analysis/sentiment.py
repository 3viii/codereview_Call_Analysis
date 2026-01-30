from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
from src.utils import get_logger

logger = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        self.model_name = model_name
        self.pipeline = None
        self.vader = VaderAnalyzer()
        
    def load(self):
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            self.pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model_name,
                tokenizer=self.model_name
            )
        except Exception as e:
            logger.warning(f"Sentiment model load failed ({e}), using VADER fallback.")
            self.pipeline = None

    def analyze(self, text: str) -> dict:
        """
        Returns: {"label": "POSITIVE"/"NEUTRAL"/"NEGATIVE", "score": float}
        """
        if not text:
            return {"label": "NEUTRAL", "score": 0.0}
            
        # 1. Try Transformer
        if self.pipeline:
            try:
                res = self.pipeline(text[:512])[0] # truncation
                label = res['label'].upper()
                score = float(res['score'])
                
                # Model specific mapping
                if "LABEL_0" in label: label = "NEGATIVE"
                elif "LABEL_1" in label: label = "NEUTRAL" 
                elif "LABEL_2" in label: label = "POSITIVE"
                
                return {"label": label, "score": score}
            except Exception as e:
                logger.warning(f"Transformer inference failed: {e}")
        
        # 2. VADER fallback
        scores = self.vader.polarity_scores(text)
        comp = scores['compound']
        if comp >= 0.05:
            label = "POSITIVE"
        elif comp <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return {"label": label, "score": abs(comp)}
