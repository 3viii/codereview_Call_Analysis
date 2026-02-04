#!/usr/bin/env python3
"""
Test single-speaker handling in pipeline.
Verifies that the pipeline doesn't crash with single-speaker diarization.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AudioPipeline

def test_single_speaker():
    """Test that pipeline handles single-speaker gracefully."""
    print("\n" + "="*60)
    print("SINGLE-SPEAKER HANDLING TEST")
    print("="*60 + "\n")
    
    pipeline = AudioPipeline()
    
    # Test with single speaker
    print("[Test] Single speaker - should not crash")
    print("-" * 60)
    
    turns = [
        {"speaker": "SPEAKER_00", "text": "Hello, this is a test.", "start": 0.0, "end": 2.0, "confidence": 0.95},
        {"speaker": "SPEAKER_00", "text": "Only one speaker here.", "start": 2.5, "end": 4.0, "confidence": 0.92}
    ]
    
    try:
        result = pipeline._assign_roles(turns)
        
        # Verify no crash
        assert result is not None, "Result should not be None"
        assert len(result) == 2, f"Expected 2 turns, got {len(result)}"
        
        # Verify role is None for single speaker
        for i, turn in enumerate(result):
            role = turn.get("role")
            speaker = turn.get("speaker")
            speaker_id = turn.get("speaker_id")
            
            print(f"Turn {i}: speaker={speaker}, speaker_id={speaker_id}, role={role}")
            
            assert role is None, f"Expected role=None for single speaker, got {role}"
            assert speaker == "SPEAKER_00", f"Speaker label changed: {speaker}"
            assert speaker_id == "SPEAKER_00", f"Speaker ID changed: {speaker_id}"
        
        print("\n✓ Test PASSED - Pipeline handles single speaker gracefully")
        print("  - No crash")
        print("  - role = None")
        print("  - Speaker labels preserved")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_speaker()
    
    print("\n" + "="*60)
    if success:
        print("✓ SINGLE-SPEAKER HANDLING WORKS")
        sys.exit(0)
    else:
        print("✗ SINGLE-SPEAKER HANDLING FAILED")
        sys.exit(1)
