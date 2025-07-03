from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    WANDB_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: Optional[str] = "http://localhost:11434"

    model_config = SettingsConfigDict(env_file=".env")

config = Config()