#!/usr/bin/env python3
"""
Role Assignment Validation Test
Verifies that role assignment works correctly as post-processing
without affecting diarization output.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AudioPipeline

def test_role_assignment():
    """
    Test role assignment with mock data to verify:
    1. Diarization speaker labels are preserved
    2. Roles are assigned via post-processing only
    3. Weighted keyword scoring works correctly
    """
    print(f"\n{'='*60}")
    print("ROLE ASSIGNMENT VALIDATION TEST")
    print(f"{'='*60}\n")
    
    pipeline = AudioPipeline()
    
    # Test Case 1: Clear collector vs debtor
    print("[Test 1] Clear collector vs debtor keywords")
    print("-" * 60)
    
    turns = [
        {"speaker": "SPEAKER_00", "text": "Hello, I am calling from HDFC bank regarding your loan EMI payment.", "start": 0.0, "end": 3.0, "confidence": 0.95},
        {"speaker": "SPEAKER_01", "text": "Yes, I understand. I will pay tomorrow when I get my salary.", "start": 3.5, "end": 6.0, "confidence": 0.92},
        {"speaker": "SPEAKER_00", "text": "This call is recorded for our records. Your payment is due.", "start": 6.5, "end": 9.0, "confidence": 0.94},
        {"speaker": "SPEAKER_01", "text": "I cannot pay today. Please give me time until next week.", "start": 9.5, "end": 12.0, "confidence": 0.91}
    ]
    
    result = pipeline._assign_roles(turns)
    
    # Verify speaker labels preserved
    assert result[0]["speaker"] == "SPEAKER_00", "Speaker label changed!"
    assert result[1]["speaker"] == "SPEAKER_01", "Speaker label changed!"
    
    # Verify roles assigned
    speaker_00_role = result[0].get("role")
    speaker_01_role = result[1].get("role")
    
    print(f"SPEAKER_00 role: {speaker_00_role}")
    print(f"SPEAKER_01 role: {speaker_01_role}")
    
    assert speaker_00_role == "COLLECTOR", f"Expected COLLECTOR, got {speaker_00_role}"
    assert speaker_01_role == "DEBTOR", f"Expected DEBTOR, got {speaker_01_role}"
    
    print("✓ Test 1 PASSED\n")
    
    # Test Case 2: No clear keywords - should leave unassigned
    print("[Test 2] No clear keywords")
    print("-" * 60)
    
    turns_no_keywords = [
        {"speaker": "SPEAKER_00", "text": "Hello, how are you?", "start": 0.0, "end": 2.0, "confidence": 0.95},
        {"speaker": "SPEAKER_01", "text": "I am fine, thank you.", "start": 2.5, "end": 4.0, "confidence": 0.92}
    ]
    
    result2 = pipeline._assign_roles(turns_no_keywords)
    
    role_0 = result2[0].get("role")
    role_1 = result2[1].get("role")
    
    print(f"SPEAKER_00 role: {role_0}")
    print(f"SPEAKER_01 role: {role_1}")
    
    assert role_0 is None, f"Expected None, got {role_0}"
    assert role_1 is None, f"Expected None, got {role_1}"
    
    print("✓ Test 2 PASSED\n")
    
    # Test Case 3: Verify speaker_id preservation
    print("[Test 3] Speaker ID preservation")
    print("-" * 60)
    
    for i, turn in enumerate(result):
        assert "speaker_id" in turn, f"Turn {i} missing speaker_id"
        assert turn["speaker_id"] == turn["speaker"], f"speaker_id doesn't match speaker"
        print(f"Turn {i}: speaker={turn['speaker']}, speaker_id={turn['speaker_id']}, role={turn.get('role')}")
    
    print("✓ Test 3 PASSED\n")
    
    print(f"{'='*60}")
    print("✓ ALL TESTS PASSED")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_role_assignment()
        if success:
            print("✓ VALIDATION SUCCESSFUL")
            sys.exit(0)
        else:
            print("✗ VALIDATION FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
