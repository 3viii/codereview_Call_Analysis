#!/usr/bin/env python3
"""
Test deterministic fallback when all scores are tied.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AudioPipeline

def test_deterministic_fallback():
    """Test that first speaker becomes COLLECTOR when scores tie."""
    print("\n" + "="*60)
    print("DETERMINISTIC FALLBACK TEST")
    print("="*60 + "\n")
    
    pipeline = AudioPipeline()
    
    # Test with no keywords - all scores = 0 (tied)
    print("[Test] No keywords - scores tied at 0")
    print("-" * 60)
    
    turns = [
        {"speaker": "SPEAKER_00", "text": "Hello there.", "start": 0.0, "end": 1.0, "confidence": 0.95},
        {"speaker": "SPEAKER_01", "text": "Hi, how are you?", "start": 1.5, "end": 2.5, "confidence": 0.92},
        {"speaker": "SPEAKER_00", "text": "I am fine.", "start": 3.0, "end": 4.0, "confidence": 0.94},
        {"speaker": "SPEAKER_01", "text": "That's good.", "start": 4.5, "end": 5.0, "confidence": 0.91}
    ]
    
    result = pipeline._assign_roles(turns)
    
    # Verify roles are assigned (not None)
    for i, turn in enumerate(result):
        role = turn.get("role")
        speaker = turn.get("speaker")
        print(f"Turn {i}: speaker={speaker}, role={role}")
        
        if role is None:
            print(f"✗ FAIL: role is None for turn {i}")
            return False
    
    # Verify first speaker (SPEAKER_00) is COLLECTOR
    speaker_00_role = result[0].get("role")
    speaker_01_role = result[1].get("role")
    
    print()
    if speaker_00_role == "COLLECTOR":
        print(f"✓ SPEAKER_00 (first speaker) -> COLLECTOR (correct)")
    else:
        print(f"✗ SPEAKER_00 -> {speaker_00_role} (expected COLLECTOR)")
        return False
    
    if speaker_01_role == "DEBTOR":
        print(f"✓ SPEAKER_01 (second speaker) -> DEBTOR (correct)")
    else:
        print(f"✗ SPEAKER_01 -> {speaker_01_role} (expected DEBTOR)")
        return False
    
    print("\n" + "="*60)
    print("✓ DETERMINISTIC FALLBACK WORKS")
    print("  First speaker -> COLLECTOR when scores tie")
    return True


if __name__ == "__main__":
    success = test_deterministic_fallback()
    sys.exit(0 if success else 1)
