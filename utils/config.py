"""
Agnostos Lab - Configuration Management
Centralized settings with environment variable support
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings,SettingsConfigDict
from dotenv import load_dotenv
from pathlib import Path


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """
    base_dir: Path= Path(__file__).resolve().parent.parent
    temp_dir: Path = base_dir / "temp"
    artifacts_dir: Path = base_dir / "artifacts"
    #--------------setting up the DB-------
    database_url: str = os.getenv("DATABASE_URL", "")
    # ============ LLM Configuration ============
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    
    # ============ Sandbox Configuration ============
    sandbox_timeout: int = 300  # 5 minutes default
    max_retries: int = 3  # Max self-heal attempts
    
    # ============ Modal Configuration (Optional GPU) ============
    modal_token_id: str = os.getenv("MODAL_TOKEN_ID", "")
    modal_token_secret: str = os.getenv("MODAL_TOKEN_SECRET", "")
    use_gpu: bool = False  # Enable GPU execution via Modal
    
    # ============ Paths ============ this is for the parallel exection
    max_parallel_attempts: int = 3
    execution_timeout: int = 600
    
    # ============ Logging ============
    log_level: str = "INFO"
    enable_tracing: bool = False  # LangSmith tracing

    
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore"    
    )
    
    def validate_keys(self) -> dict[str, bool]:
        """
        Validate that required API keys are configured.
        
        Returns:
            Dict with validation status for each key
        """
        return {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "modal": bool(self.modal_token_id and self.modal_token_secret)
        }
    
    def get_llm_provider(self) -> str:
        """
        Determine which LLM provider to use based on available keys.
        
        Returns:
            Provider name: 'openai' or 'anthropic'
        """
        if self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        else:
            raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


# Global settings instance
settings = Settings()
