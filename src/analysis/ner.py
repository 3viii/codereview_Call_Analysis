import re
from transformers import pipeline
from typing import List, Dict, Any, Union
from src.utils import get_logger

logger = get_logger(__name__)

# --- Regex Helpers ---

def extract_amounts_regex(text: str) -> List[str]:
    text = text or ""
    # remove grouping commas
    normalized = text.replace(",", "").replace("\u200e", " ")
    # remove ordinal suffixes
    normalized = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", normalized, flags=re.IGNORECASE)

    pattern = r"(?:₹|rs\.?|rupees)?\s*([0-9][0-9\.\,]{1,10}|[0-9]{3,10})"
    candidates = []
    for m in re.finditer(pattern, normalized, flags=re.IGNORECASE):
        num_str = m.group(1).strip()
        num_clean = re.sub(r"[^\d]", "", num_str)
        if not num_clean:
            continue
        try:
            v = int(num_clean)
        except:
            continue
        if 1900 <= v <= 2099: continue
        
        # skip dates context
        span_start, span_end = max(0, m.start()-10), min(len(normalized), m.end()+10)
        nearby = normalized[span_start:span_end]
        if re.search(r"\b[0-3]?\d\s*[/\-]\s*[0-1]?\d\s*[/\-]\s*\d{2,4}\b", nearby): continue
        if re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", nearby, flags=re.IGNORECASE): continue
        if v <= 31: continue
        
        candidates.append(str(v))
        
    return list(dict.fromkeys(candidates))

def extract_dates_regex(text: str) -> List[str]:
    text = text or ""
    dates = []
    # dd/mm/yyyy
    for m in re.finditer(r"\b([0-3]?\d)[/\\-]([0-1]?\d)[/\\-](\d{2,4})\b", text):
        dates.append(m.group(0))
    # day month
    for m in re.finditer(r"\b([0-3]?\d)(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b", text, flags=re.IGNORECASE):
        dates.append(f"{m.group(1)} {m.group(2)}")
    # relative
    for m in re.finditer(r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text, flags=re.IGNORECASE):
        dates.append(m.group(0))
    for kw in ["tomorrow", "today", "yesterday"]:
        if re.search(rf"\b{kw}\b", text, flags=re.IGNORECASE):
            dates.append(kw)
    # weekdays
    for m in re.finditer(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text, flags=re.IGNORECASE):
        dates.append(m.group(1))
        
    # clean
    out = []
    seen = set()
    for d in dates:
        d_norm = " ".join(d.split())
        if d_norm.lower() not in seen:
            seen.add(d_norm.lower())
            out.append(d_norm)
    return out

def extract_modes_regex(text: str) -> List[str]:
    modes = []
    payment_modes = ["upi", "bank transfer", "bank", "cheque", "cash", "neft", "rtgs", "net banking", "online", "wallet", "card"]
    text_lower = text.lower()
    for mode in payment_modes:
        if mode in text_lower:
            modes.append(mode)
    return sorted(list(set(modes)))

def postprocess_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    if not entities:
        return {"Amount": [], "Date": [], "Mode": [], "PERSON": [], "ORG": [], "LOC": []}

    # Clean PERSON/ORG/LOC
    for k in ("PERSON", "ORG", "LOC"):
        cleaned = []
        for tok in entities.get(k, []):
            if not tok: continue
            tok_clean = tok.strip().strip(".,;:()[]\"'").strip()
            if len(tok_clean) <= 2: continue
            if re.fullmatch(r"[\d\W_]+", tok_clean): continue
            cleaned.append(tok_clean)
        entities[k] = sorted(list(dict.fromkeys(cleaned)))

    noisy_orgs = {"emi", "up", "upi", "amount", "payment"}
    entities["ORG"] = [o for o in entities.get("ORG", []) if o and o.lower() not in noisy_orgs]

    # Normalize Mode
    norm_map = {"upi": "UPI", "up": "UPI", "bank transfer": "Bank Transfer", "bank": "Bank Transfer", 
                "neft": "NEFT", "rtgs": "RTGS", "net banking": "Net Banking", "cheque": "Cheque", 
                "cash": "Cash", "card": "Card"}
    modes = []
    for m in entities.get("Mode", []):
        m_norm = re.sub(r"[^\w\s]", "", m.lower().strip())
        modes.append(norm_map.get(m_norm, m.strip().title()))
    entities["Mode"] = sorted(list(dict.fromkeys(modes)))

    # Clean Amounts (re-check logic even if regex does most)
    clean_amounts = []
    for a in entities.get("Amount", []):
         try:
            v = int(float(re.sub(r"[^\d]", "", str(a))))
            if 1900 <= v <= 2099: continue
            if v < 100: continue
            clean_amounts.append(str(v))
         except: pass
    entities["Amount"] = sorted(list(dict.fromkeys(clean_amounts)))

    # Dates preference
    dates = entities.get("Date", [])
    numeric_dates = [d for d in dates if re.search(r"\d", d) or re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", d, flags=re.I)]
    entities["Date"] = sorted(list(dict.fromkeys(numeric_dates if numeric_dates else dates)))
    entities["Date"] = [d for d in entities["Date"] if not re.fullmatch(r"\d{2,4}", d)]

    return entities


class NERExtractor:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.model_name = model_name
        self.pipeline = None
    
    def load(self):
        try:
            logger.info(f"Loading NER model: {self.model_name}")
            self.pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"NER model load failed: {e}")
            self.pipeline = None

    def extract(self, text: str) -> Dict[str, List[str]]:
        entities = {"Amount": [], "Date": [], "Mode": [], "PERSON": [], "ORG": [], "LOC": []}
        if not text: return entities
        
        # ML
        if self.pipeline:
            try:
                res = self.pipeline(text)
                for ent in res:
                    label = ent.get("entity_group", "")
                    word = ent.get("word", "")
                    if word.startswith("##"): word = word[2:]
                    if label in entities and word:
                        entities[label].append(word)
            except Exception as e:
                logger.warning(f"NER inference failed: {e}")

        # Regex
        entities["Amount"].extend(extract_amounts_regex(text))
        entities["Date"].extend(extract_dates_regex(text))
        entities["Mode"].extend(extract_modes_regex(text))
        
        return postprocess_entities(entities)
