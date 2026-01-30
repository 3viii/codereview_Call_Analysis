from src.pipeline import AudioPipeline
from src.config import CONFIG
import os

def test_pipeline_mock():
    # Force mock mode
    CONFIG.use_api = "mock"
    
    pipeline = AudioPipeline()
    # Mock doesn't really check file existence, but let's be safe
    # We can pass any string
    res = pipeline.process_file("dummy.wav", out_dir="tests/outputs")
    
    assert res["transcript"]
    assert len(res["turns"]) > 0
    
    # Check new fields
    first_turn = res["turns"][0]
    assert "confidence" in first_turn
    assert "role" in first_turn
    assert first_turn["role"] in ["collector", "debtor"]
    
    assert res["intent"]
    
    # Check if files were created
    assert os.path.exists("tests/outputs/analysis.json")
    assert os.path.exists("tests/outputs/transcript.txt")
    
    # Clean up
    import shutil
    shutil.rmtree("tests/outputs")
