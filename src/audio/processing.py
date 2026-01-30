import soundfile as sf
import logging
from src.utils import get_logger

logger = get_logger(__name__)

def load_audio(path: str, target_sr: int = 16000):
    """
    Load audio file and resample to target_sr. 
    Returns: (audio_array: np.ndarray [float32], sr: int)
    """
    try:
        wav, sr = sf.read(path, dtype='int16')
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype('int16')
        audio = wav.astype('float32') / 32768.0
        
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except ImportError:
                logger.warning("librosa not installed, cannot resample. Returning original sr.")
            except Exception as e:
                logger.warning(f"Resampling failed: {e}. Returning original sr.")
        
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio {path}: {e}")
        raise

