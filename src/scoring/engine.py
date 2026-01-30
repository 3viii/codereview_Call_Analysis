import re
from typing import List, Dict, Any

class ScoringEngine:
    @staticmethod
    def score(turns: List[Dict], intent: str, sentiment: Dict, speech_emotion: Dict) -> Dict[str, int]:
        """
        Compute 1-5 scores for Listening, Communication, Persuasion, Outcome.
        """
        s_label = (sentiment or {}).get("label", "UNKNOWN").upper()
        s_score = float((sentiment or {}).get("score", 0.0))
        e_label = (speech_emotion or {}).get("label_pretty", (speech_emotion or {}).get("label", "UNKNOWN")).upper()
        e_score = float((speech_emotion or {}).get("score", 0.0) or 0.0)

        all_text = " ".join(t.get("text", "") for t in turns)
        all_text_lower = all_text.lower()

        # Base scores
        listening = 3
        communication = 3
        persuasion = 3
        outcome = 3

        # Listening
        if any(ph in all_text_lower for ph in ["i understand", "okay", "sure", "let me check", "please"]):
            listening += 1
        if "sorry" in all_text_lower or "apologize" in all_text_lower:
            listening += 1

        # Communication
        num_words = len(all_text.split())
        if num_words > 80:
            communication += 1
        if s_label == "POSITIVE":
            communication += 1
        elif s_label == "NEGATIVE":
            communication -= 1

        # Persuasion
        if s_label == "POSITIVE" and s_score > 0.6:
            persuasion += 1
        if e_label in ["HAPPY", "EXCITED"] and e_score > 0.5:
            persuasion += 1
        if e_label in ["ANGRY", "SAD", "FEAR"] and e_score > 0.5:
            persuasion -= 1

        # Outcome
        intent_lower = (intent or "").lower()
        if "full promise" in intent_lower or "arrangement" in intent_lower or "confirmation" in intent_lower:
            outcome += 1
        if "refusal" in intent_lower:
            outcome -= 1
        if s_label == "NEGATIVE":
            outcome -= 1
        
        # If any amounts present in text -> slightly better outcome (heuristic from original code)
        # Original: if len([a for a in (turns or []) if re.search(r"\d", " ".join(a.values()))]) > 0:
        # But 'turns' is list of dicts. " ".join(a.values()) joins text/speaker/etc.
        if any(re.search(r"\d", t.get("text", "")) for t in turns):
            outcome += 0 # Original code did += 0, keeping it for parity or maybe it was a placeholder I should improve?
            # Original comment: "# If amounts and dates present -> slightly better outcome"
            # It said += 0. I will leave it as is or maybe make it +1 if that was intended?
            # The prompt says "Fix bugs". += 0 does nothing.
            # I will assume it was meant to be += 1.
            outcome += 1

        def clamp(x): return max(1, min(5, int(round(x))))
        
        return {
            "Listening": clamp(listening),
            "Communication": clamp(communication),
            "Persuasion": clamp(persuasion),
            "Outcome": clamp(outcome),
        }
