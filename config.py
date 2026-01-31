"""
Agnostos Lab - Configuration Management
Centralized settings with environment variable support
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    All settings can be overridden via environment variables.
    """
    
    # ============ LLM Configuration ============
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    
    # ============ Sandbox Configuration ============
    e2b_api_key: str = os.getenv("E2B_API_KEY", "")
    sandbox_timeout: int = 300  # 5 minutes default
    max_retries: int = 3  # Max self-heal attempts
    
    # ============ Modal Configuration (Optional GPU) ============
    modal_token_id: str = os.getenv("MODAL_TOKEN_ID", "")
    modal_token_secret: str = os.getenv("MODAL_TOKEN_SECRET", "")
    use_gpu: bool = False  # Enable GPU execution via Modal
    
    # ============ Paths ============
    temp_dir: str = "./temp"
    artifacts_dir: str = "./artifacts"
    
    # ============ Logging ============
    log_level: str = "INFO"
    enable_tracing: bool = False  # LangSmith tracing
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def validate_keys(self) -> dict[str, bool]:
        """
        Validate that required API keys are configured.
        
        Returns:
            Dict with validation status for each key
        """
        return {
            "openai": bool(self.openai_api_key),
            "e2b": bool(self.e2b_api_key),
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
