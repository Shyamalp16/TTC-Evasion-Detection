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
    primary_camera_id: int = 0  # Primary camera for snapshots and events
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

    # Gate Crossing Settings
    enable_gate_crossing: bool = True
    gate_crossing_line_x: float = 0.5  # Horizontal position (0-1) where gate crossing line is drawn
    gate_crossing_direction: str = "right"  # "left" or "right" - direction of crossing (person moving from left to right or right to left)
    gate_crossing_hysteresis: int = 10  # Pixels of hysteresis to prevent bouncing
    person_tracking_max_age: int = 15  # Frames before forgetting a tracked person
    person_tracking_iou_threshold: float = 0.1  # IOU threshold for matching detections to tracks

    # Server Configuration
    server_url: str = "http://127.0.0.1:8000"  # Dev: local server
    server_api_key: str = "your-api-key-here"
    server_timeout: int = 30
    
    # Processing Settings
    frame_skip: int = 5  # Process every 5th frame (dev: smoother UI)
    max_queue_size: int = 100
    detection_interval: float = 0.15  # seconds
    preview_interval: float = 0.03  # seconds between UI updates (~33 FPS)
    
    # Gate Crossing Settings
    enable_gate_crossing: bool = True
    gate_crossing_line_x: float = 0.5  # Horizontal position (0-1) where gate crossing line is drawn
    gate_crossing_direction: str = "right"  # "left" or "right"
    gate_crossing_hysteresis: int = 10  # Pixels of hysteresis to prevent bouncing
    person_tracking_max_age: int = 15  # Frames before forgetting a tracked person
    person_tracking_iou_threshold: float = 0.1  # IOU threshold for matching detections to tracks
    person_exit_edge_margin: int = 15  # Pixels from frame edge to consider as potentially exiting
    person_exit_edge_frames: int = 3  # Frames at edge before considering person exited
    person_exit_margin: int = 30  # Pixels outside frame to immediately consider as exited

    # Pose/Gesture Settings
    # Backend: "mediapipe" only (movenet removed)
    pose_backend: str = "mediapipe"
    pose_score_threshold: float = 0.3

    # MediaPipe Pose parameters
    mediapipe_model_complexity: int = 1
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5

    # Gesture detection parameters (used by tap gesture)
    gesture_chest_tolerance_px: int = 40
    gesture_proximity_px: int = 60
    gesture_dwell_min_s: float = 0.25
    gesture_dwell_max_s: float = 1.0
    pose_full_frame: bool = False

    # Overlay/Visualization
    pose_overlay_enabled: bool = True
    pose_overlay_min_confidence: float = 0.3
    pose_overlay_infer_if_missing: bool = True
    pose_overlay_draw_angles: bool = True

    # Event Batching Settings
    detection_batch_interval: float = 1.0  # seconds - batch regular detections
    evasion_send_immediately: bool = True  # send evasion events immediately
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/edge_device.log"
    enable_console_logs: bool = False  # disable console logs, use file logging
    console_log_level: str = "WARNING"
    file_log_level: str = "DEBUG"
    
    # Privacy Settings
    snapshot_retention_days: int = 30
    max_snapshot_size_mb: int = 5
    
    class Config:
        env_file = ".env"
        env_prefix = "SNITCH_"


# Global settings instance
settings = Settings()
