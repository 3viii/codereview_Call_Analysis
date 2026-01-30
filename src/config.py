import os
from pydantic import BaseModel, Field
from typing import Optional

class AppConfig(BaseModel):
    use_api: str = Field(default="whisper_local", description="ASR method: mock, whisper_local, whisper, google")
    openai_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    google_credentials_path: Optional[str] = Field(default=None, description="Path to Google Credentials JSON")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace Token for Pyannote")
    
    # Model paths / names (can be overriden via env vars if we wanted)
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment"
    ner_model: str = "dslim/bert-base-NER"
    speech_emotion_model: str = "superb/hubert-large-superb-er"
    
    class Config:
        env_prefix = "INTELLISCORE_"

def load_config(json_path: str = "config_example.json") -> AppConfig:
    import json
    config_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"[CONFIG] Warning: Could not load {json_path}: {e}")
    
    # Environment variables can override (using pydantic's automatic env binding if we instantiated directly, 
    # but here we mix json + env. For now, simple binding)
    # Environment variables override config file
    env_use_api = os.environ.get("INTELLISCORE_USE_API")
    if env_use_api:
        config_data["use_api"] = env_use_api
        
    env_hf_token = os.environ.get("INTELLISCORE_HF_TOKEN")
    if env_hf_token:
        config_data["hf_token"] = env_hf_token

    return AppConfig(**config_data)

CONFIG = load_config()
