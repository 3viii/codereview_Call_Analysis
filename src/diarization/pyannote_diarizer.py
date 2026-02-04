import os
import torch
from typing import List, Dict, Any
from src.config import CONFIG
from src.utils import get_logger

logger = get_logger(__name__)

class PyannoteDiarizer:
    def __init__(self):
        self.pipeline = None
        
    def load(self):
        """Load Pyannote pipeline. Fails loudly if unsuccessful."""
        logger.info("=" * 60)
        logger.info("INITIALIZING PYANNOTE DIARIZATION")
        logger.info("=" * 60)
        
        # Import required modules
        try:
            from pyannote.audio import Pipeline
            from huggingface_hub import login, HfFolder
            logger.info("✓ pyannote.audio imported successfully")
        except ImportError as e:
            error_msg = (
                f"CRITICAL: Failed to import required modules: {e}\n"
                "Install with: pip install pyannote.audio huggingface_hub"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Authenticate with HuggingFace
        # Try multiple token sources in order
        token = None
        
        # 1. Check if already logged in via huggingface-cli
        try:
            existing_token = HfFolder.get_token()
            if existing_token:
                token = existing_token
                logger.info("✓ Using existing HuggingFace CLI login")
        except:
            pass
        
        # 2. Check environment/config
        if not token:
            token = CONFIG.hf_token or os.environ.get("INTELLISCORE_HF_TOKEN")
            if token:
                logger.info(f"✓ HuggingFace token found: {token[:10]}...")
                try:
                    login(token=token, add_to_git_credential=False)
                    logger.info("✓ Authenticated with HuggingFace")
                except Exception as e:
                    logger.warning(f"Token login failed: {e}, continuing anyway...")
        
        if not token:
            error_msg = (
                "CRITICAL: No HuggingFace authentication found!\n"
                "Pyannote requires authentication. Options:\n"
                "  1. Run: huggingface-cli login\n"
                "  2. Set: export INTELLISCORE_HF_TOKEN=hf_...\n"
                "  3. Add to config_example.json: {\"hf_token\": \"hf_...\"}\n"
                "\n"
                "Get token from: https://huggingface.co/settings/tokens\n"
                "IMPORTANT: Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Load the pipeline
        try:
            logger.info("Loading Pyannote pipeline: pyannote/speaker-diarization-3.1")
            logger.info("This may take a few moments on first run...")
            logger.info("NOTE: You must accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            
            # Verify pipeline loaded
            if self.pipeline is None:
                raise RuntimeError("Pipeline.from_pretrained returned None - model not accessible")
            
            logger.info("[PYANNOTE_OK] Pipeline loaded successfully")
            
            # Move to appropriate device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                self.pipeline.to(device)
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("✓ Using Apple MPS (Metal)")
                self.pipeline.to(device)
            else:
                logger.info("✓ Using CPU (no GPU detected)")
            
            logger.info("=" * 60)
            logger.info("PYANNOTE INITIALIZATION COMPLETE")
            logger.info("=" * 60)
            
        except Exception as e:
            self.pipeline = None
            error_msg = (
                f"CRITICAL: Failed to load Pyannote pipeline: {e}\n"
                f"Error type: {type(e).__name__}\n"
                "\n"
                "Common causes:\n"
                "  1. Model terms not accepted - visit: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  2. Invalid/expired HuggingFace token\n"
                "  3. No internet connection\n"
                "  4. Insufficient disk space for model cache\n"
                "\n"
                "Troubleshooting:\n"
                "  - Verify login: huggingface-cli whoami\n"
                "  - Accept model terms at the URL above\n"
                "  - Check internet connection\n"
                "  - Clear cache: rm -rf ~/.cache/huggingface"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Returns list of segments: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
        Fails loudly if pipeline not loaded or diarization fails.
        """
        if not self.pipeline:
            error_msg = "CRITICAL: Pyannote pipeline not loaded. Cannot perform diarization."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if not os.path.exists(audio_path):
            error_msg = f"CRITICAL: Audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Running diarization on: {audio_path}")
            
            # Run inference
            diarization = self.pipeline(audio_path)
            
            segments = []
            # iterate over turns
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
            
            logger.info(f"✓ Diarization complete: {len(segments)} segments found")
            unique_speakers = set(s["speaker"] for s in segments)
            logger.info(f"✓ Unique speakers: {unique_speakers}")
            
            return segments
            
        except Exception as e:
            error_msg = f"CRITICAL: Diarization failed: {e}\nError type: {type(e).__name__}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
