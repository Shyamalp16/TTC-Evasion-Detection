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
    #rtsp:// URLs.
    camera_urls: List[str] = [
        "0",  # Laptop webcam index 0
        # "1",  # Uncomment if you have a second webcam
        # "rtsp://admin:password@192.168.1.100:554/stream1",
        # "rtsp://admin:password@192.168.1.101:554/stream1"
    ]
    camera_timeout: int = 10
    
    # Detection Settings
    detection_confidence: float = 0.5
    detection_iou_threshold: float = 0.45
    detection_classes: List[int] = [0]  # Person class in COCO dataset
    yolo_model_path: str = "yolov8m.pt"  # Heavier model for better long-range accuracy
    detection_downscale_ratio: float = 1.0  # Keep full resolution so distant subjects stay visible
    reference_person_height_m: float = 1.7  # Average adult height for depth estimation
    camera_focal_length_mm: float = 3.6  # Typical webcam focal length
    camera_sensor_height_mm: float = 2.76  # Sensor height (mm) for focal length conversion
    max_detection_distance_m: float = 25.0  # Clamp distance estimates to a reasonable range
    
    # Server Configuration
    server_url: str = "http://127.0.0.1:8000"  # Dev: local server
    server_api_key: str = "your-api-key-here"
    server_timeout: int = 30
    
    # Processing Settings
    frame_skip: int = 5  # Process every 5th frame (dev: smoother UI)
    max_queue_size: int = 100
    detection_interval: float = 0.15  # seconds
    preview_interval: float = 0.03  # seconds between UI updates (~33 FPS)
    
    # Event Batching Settings
    detection_batch_interval: float = 60.0  # seconds - batch regular detections
    evasion_send_immediately: bool = True  # send evasion events immediately
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/edge_device.log"
    enable_console_logs: bool = False  # reduce terminal overhead
    console_log_level: str = "WARNING"
    file_log_level: str = "INFO"
    
    # Privacy Settings
    snapshot_retention_days: int = 30
    max_snapshot_size_mb: int = 5
    
    class Config:
        env_file = ".env"
        env_prefix = "SNITCH_"


# Global settings instance
settings = Settings()
