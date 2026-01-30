from transformers import pipeline
from src.utils import get_logger

logger = get_logger(__name__)

class IntentClassifier:
    INTENT_LABELS = [
        "Payment Discussion",
        "Dispute",
        "Confirmation",
        "Arrangement",
        "Full Promise to Pay",
        "Partial Promise",
        "Refusal",
        "General Inquiry"
    ]

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None

    def load(self):
        try:
            logger.info(f"Loading intent classifier: {self.model_name}")
            self.classifier = pipeline("zero-shot-classification", model=self.model_name)
        except Exception as e:
            logger.warning(f"Intent classifier load failed: {e}")
            self.classifier = None

    def classify(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Ambiguous"

        debt_keywords = ["pay", "payment", "emi", "due", "overdue", "amount", "rupees", "rs", "₹", "upi", "bank", "transfer"]
        has_debt_terms = any(w in text.lower() for w in debt_keywords)

        if not self.classifier:
             return "Payment Discussion" if has_debt_terms else "General Inquiry"

        try:
            result = self.classifier(text, candidate_labels=self.INTENT_LABELS, multi_label=False)
            best_label = result["labels"][0]
            best_score = float(result["scores"][0])
            
            if best_score < 0.5 and not has_debt_terms:
                return "General Inquiry"
            return best_label
        except Exception as e:
            logger.warning(f"Intent inference failed: {e}")
            return "Payment Discussion" if has_debt_terms else "General Inquiry"
