import sys
import os
from src.pipeline import AudioPipeline
from src.config import CONFIG
from src.utils import setup_logging

def main():
    setup_logging()
    
    audio_files = sys.argv[1:]
    
    if not audio_files:
        print("[MAIN] No audio provided. Usage: python main.py <file.wav>")
        if CONFIG.use_api == "mock":
             # Demo mode for mock
             print("[MAIN] Running mock demo with dummy path...")
             pipeline = AudioPipeline()
             pipeline.process_file("dummy.wav")
        return

    pipeline = AudioPipeline() # Lazy loads resources

    for audio_file in audio_files:
        print(f"\n=== Processing {audio_file} ===")
        try:
            res = pipeline.process_file(audio_file)
            print("Done!")
            print("Intent:", res.get("intent"))
            scores = res.get("scores", {})
            print("Scores:", scores)
            
            # Print minimal summary
            print("\n--- Summary ---")
            print(f"Transcript extract: {res.get('transcript', '')[:100]}...")
            print(f"Entities: {res.get('entities')}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
