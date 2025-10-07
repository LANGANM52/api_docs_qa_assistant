from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # API Configuration
    app_name: str = "API Documentation Q&A Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Vector Store Configuration
    vector_store_directory: str = "./vector_db"
    collection_name: str = "api_documentation"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    # API Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()