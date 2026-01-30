
from src.analysis.ner import extract_amounts_regex, extract_dates_regex, extract_modes_regex
from src.analysis.sentiment import SentimentAnalyzer

def test_amount_extraction():
    text = "I will pay 5000 rupees tomorrow. Also 10,000 next week."
    expected = ["5000", "10000"]
    assert extract_amounts_regex(text) == expected

def test_date_extraction():
    text = "Call me on 10th October or next Friday."
    dates = extract_dates_regex(text)
    assert "10 October" in dates
    assert "next Friday" in dates

def test_mode_extraction():
    text = "I will do a bank transfer or UPI."
    modes = extract_modes_regex(text)
    assert "bank transfer" in modes
    assert "upi" in modes

def test_sentiment_analyzer_vader_fallback():
    # Test without loading the heavy model (should use VADER)
    analyzer = SentimentAnalyzer() 
    # Not calling load() so pipeline is None, triggers fallback
    res = analyzer.analyze("I am very happy with this service.")
    assert res["label"] == "POSITIVE"
    assert res["score"] > 0
    
    res = analyzer.analyze("This is terrible.")
    assert res["label"] == "NEGATIVE"
