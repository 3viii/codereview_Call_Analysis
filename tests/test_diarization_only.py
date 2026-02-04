#!/usr/bin/env python3
"""
Minimal Pyannote Diarization Test
Tests ONLY diarization without ASR, role assignment, or other logic.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diarization.pyannote_diarizer import PyannoteDiarizer
from src.audio.processing import load_audio
import tempfile
import soundfile as sf

def test_diarization(audio_path):
    """
    Test Pyannote diarization on an audio file.
    Prints: number of speakers, speaker labels, segment times.
    """
    print(f"\n{'='*60}")
    print(f"TESTING DIARIZATION: {audio_path}")
    print(f"{'='*60}\n")
    
    # Initialize diarizer
    print("[1/4] Loading Pyannote model...")
    diarizer = PyannoteDiarizer()
    diarizer.load()
    print("✓ Model loaded\n")
    
    # Preprocess audio (16kHz mono)
    print("[2/4] Preprocessing audio to 16kHz mono...")
    wav, sr = load_audio(audio_path, target_sr=16000)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    sf.write(tmp_path, wav, sr)
    print(f"✓ Preprocessed audio saved to: {tmp_path}\n")
    
    # Run diarization
    print("[3/4] Running Pyannote diarization...")
    try:
        segments = diarizer.diarize(tmp_path)
        print(f"✓ Diarization complete\n")
    except Exception as e:
        print(f"✗ DIARIZATION FAILED: {e}")
        os.remove(tmp_path)
        return False
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Analyze results
    print("[4/4] Analyzing results...")
    print(f"\n{'='*60}")
    print("DIARIZATION RESULTS")
    print(f"{'='*60}\n")
    
    if not segments:
        print("✗ FAILED: No segments returned")
        return False
    
    # Extract unique speakers
    unique_speakers = set(s["speaker"] for s in segments)
    num_speakers = len(unique_speakers)
    
    print(f"Total Segments: {len(segments)}")
    print(f"Unique Speakers: {num_speakers}")
    print(f"Speaker Labels: {sorted(unique_speakers)}\n")
    
    # Show first 10 segments
    print("First 10 Segments:")
    print(f"{'Speaker':<15} {'Start':<10} {'End':<10} {'Duration':<10}")
    print("-" * 50)
    
    for i, seg in enumerate(segments[:10]):
        speaker = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        duration = end - start
        print(f"{speaker:<15} {start:<10.2f} {end:<10.2f} {duration:<10.2f}")
    
    if len(segments) > 10:
        print(f"... ({len(segments) - 10} more segments)")
    
    print(f"\n{'='*60}")
    
    # Validation
    if num_speakers < 2:
        print(f"\n✗ VALIDATION FAILED: Only {num_speakers} speaker(s) detected")
        print("Expected: >= 2 speakers")
        return False
    
    print(f"\n✓ VALIDATION PASSED: {num_speakers} speakers detected")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_diarization_only.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    success = test_diarization(audio_file)
    
    if success:
        print("\n✓ TEST PASSED")
        sys.exit(0)
    else:
        print("\n✗ TEST FAILED")
        sys.exit(1)
