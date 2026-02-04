#!/usr/bin/env python3
"""
End-to-end role assignment wiring test.
Verifies that roles flow from pipeline to dashboard correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AudioPipeline

def test_role_wiring():
    """Test that role assignment wiring works end-to-end."""
    print("\n" + "="*60)
    print("ROLE ASSIGNMENT WIRING TEST")
    print("="*60 + "\n")
    
    pipeline = AudioPipeline()
    
    # Simulate diarized + aligned turns with collector/debtor keywords
    print("[Setup] Creating test turns with clear collector/debtor keywords")
    print("-" * 60)
    
    turns = [
        {
            "speaker": "SPEAKER_00",
            "text": "Hello, I am calling from HDFC bank regarding your loan EMI payment.",
            "start": 0.0,
            "end": 3.0,
            "confidence": 0.95
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Yes, I will pay tomorrow when I get my salary.",
            "start": 3.5,
            "end": 6.0,
            "confidence": 0.92
        },
        {
            "speaker": "SPEAKER_00",
            "text": "This call is recorded. Your payment is due.",
            "start": 6.5,
            "end": 9.0,
            "confidence": 0.94
        },
        {
            "speaker": "SPEAKER_01",
            "text": "I understand. I cannot pay today but will arrange next week.",
            "start": 9.5,
            "end": 12.0,
            "confidence": 0.91
        }
    ]
    
    print(f"Input: {len(turns)} turns from 2 speakers (SPEAKER_00, SPEAKER_01)\n")
    
    # Run role assignment
    print("[Execution] Running _assign_roles...")
    print("-" * 60)
    result = pipeline._assign_roles(turns)
    print()
    
    # Verify results
    print("[Verification] Checking role propagation...")
    print("-" * 60)
    
    success = True
    
    # Check that roles are in each turn
    for i, turn in enumerate(result):
        speaker = turn.get("speaker")
        role = turn.get("role")
        text_preview = turn.get("text", "")[:50]
        
        print(f"Turn {i}: speaker={speaker}, role={role}")
        print(f"  Text: {text_preview}...")
        
        # Verify role key exists
        if "role" not in turn:
            print(f"  ✗ FAIL: 'role' key missing from turn {i}")
            success = False
        
        # Verify role is set (not None for this test case)
        if role is None:
            print(f"  ✗ FAIL: role is None for turn {i}")
            success = False
        
        # Verify role is correct type
        if role not in ["COLLECTOR", "DEBTOR", None]:
            print(f"  ✗ FAIL: Invalid role value: {role}")
            success = False
    
    print()
    
    # Verify speaker-to-role mapping
    speaker_00_roles = [t["role"] for t in result if t.get("speaker") == "SPEAKER_00"]
    speaker_01_roles = [t["role"] for t in result if t.get("speaker") == "SPEAKER_01"]
    
    # All turns from same speaker should have same role
    if len(set(speaker_00_roles)) > 1:
        print(f"✗ FAIL: SPEAKER_00 has inconsistent roles: {speaker_00_roles}")
        success = False
    
    if len(set(speaker_01_roles)) > 1:
        print(f"✗ FAIL: SPEAKER_01 has inconsistent roles: {speaker_01_roles}")
        success = False
    
    # Verify expected roles (based on keywords)
    expected_00 = "COLLECTOR"  # Has "bank", "loan", "emi", "this call is recorded"
    expected_01 = "DEBTOR"     # Has "i will pay", "salary", "tomorrow", "cannot pay", "next week"
    
    if speaker_00_roles[0] == expected_00:
        print(f"✓ SPEAKER_00 -> {expected_00} (correct)")
    else:
        print(f"✗ SPEAKER_00 -> {speaker_00_roles[0]} (expected {expected_00})")
        success = False
    
    if speaker_01_roles[0] == expected_01:
        print(f"✓ SPEAKER_01 -> {expected_01} (correct)")
    else:
        print(f"✗ SPEAKER_01 -> {speaker_01_roles[0]} (expected {expected_01})")
        success = False
    
    print()
    print("="*60)
    
    if success:
        print("✓ WIRING TEST PASSED")
        print("\nRoles are correctly:")
        print("  1. Assigned in _assign_roles")
        print("  2. Stored in turn['role']")
        print("  3. Ready for dashboard display")
        return True
    else:
        print("✗ WIRING TEST FAILED")
        return False


if __name__ == "__main__":
    success = test_role_wiring()
    sys.exit(0 if success else 1)
