"""
Server configuration settings.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database Configuration
    database_url: str = "postgresql://snitch:password@localhost:5432/snitchsystem"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "snitchsystem"
    database_user: str = "snitch"
    database_password: str = "password"
    
    # API Configuration
    api_title: str = "SnitchSystem Server API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Authentication
    api_key: str = "your-api-key-here"
    secret_key: str = "your-secret-key-here"
    
    # File Storage
    snapshots_dir: str = "snapshots"
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = ["image/jpeg", "image/png"]
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/server.log"
    
    # Data Retention
    event_retention_days: int = 90
    snapshot_retention_days: int = 30
    
    # Performance
    max_events_per_request: int = 1000
    
    class Config:
        env_file = ".env"
        env_prefix = "SNITCH_"


# Global settings instance
settings = Settings()
