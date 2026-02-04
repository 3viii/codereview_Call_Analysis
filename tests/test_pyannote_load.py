#!/usr/bin/env python3
"""
Quick test to verify Pyannote loads correctly.
"""
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diarization.pyannote_diarizer import PyannoteDiarizer

def test_pyannote_load():
    print("\n" + "="*60)
    print("PYANNOTE LOAD TEST")
    print("="*60 + "\n")
    
    try:
        diarizer = PyannoteDiarizer()
        diarizer.load()
        
        if diarizer.pipeline is None:
            print("✗ FAILED: Pipeline is None")
            return False
        
        print("\n" + "="*60)
        print("✓ PYANNOTE LOADED SUCCESSFULLY")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_pyannote_load()
    sys.exit(0 if success else 1)
