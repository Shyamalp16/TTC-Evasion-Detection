"""
Configuration settings for the edge device.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Gate Configuration
    gate_id: str = "gate_001"
    station_id: str = "station_001"
    
    # Camera Configuration
    camera_urls: List[str] = [
        "rtsp://admin:password@192.168.1.100:554/stream1",
        "rtsp://admin:password@192.168.1.101:554/stream1"
    ]
    camera_timeout: int = 10
    
    # Detection Settings
    detection_confidence: float = 0.5
    detection_iou_threshold: float = 0.45
    detection_classes: List[int] = [0]  # Person class in COCO dataset
    
    # Server Configuration
    server_url: str = "http://192.168.1.50:8000"
    server_api_key: str = "your-api-key-here"
    server_timeout: int = 30
    
    # Processing Settings
    frame_skip: int = 5  # Process every 5th frame
    max_queue_size: int = 100
    detection_interval: float = 0.1  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/edge_device.log"
    
    # Privacy Settings
    snapshot_retention_days: int = 30
    max_snapshot_size_mb: int = 5
    
    class Config:
        env_file = ".env"
        env_prefix = "SNITCH_"


# Global settings instance
settings = Settings()
