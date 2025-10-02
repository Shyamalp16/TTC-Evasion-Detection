"""
Configuration settings for the edge device.
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gate_id: str = "gate_001"
    station_id: str = "station_001"
    
    camera_urls: List[str] = [
        0, # laptop cam
        # "rtsp://Shyamalp16:Shyamalp16@10.0.0.242:554/stream2",
        # "rtsp://admin:password@192.168.1.101:554/stream1"
    ]
    primary_camera_id: int = 0 
    camera_timeout: int = 10
    disable_webcam_fallback: bool = True
    rtsp_transport: str = "tcp"  # tcp (more reliable) or udp (lower latency)
    ffmpeg_capture_options: str = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    detection_confidence: float = 0.6
    detection_iou_threshold: float = 0.45
    detection_classes: List[int] = [0]  #person clasa
    yolo_model_path: str = "yolo11m.pt"  
    detection_downscale_ratio: float = 0.8 
    reference_person_height_m: float = 1.7  
    camera_focal_length_mm: float = 3.6  
    camera_sensor_height_mm: float = 2.76  
    max_detection_distance_m: float = 25.0  

    enable_gate_crossing: bool = True
    gate_crossing_line_y: float = 0.5  
    gate_crossing_direction: str = "up"  
    gate_crossing_hysteresis: int = 10  
    person_tracking_max_age: int = 300  
    person_tracking_iou_threshold: float = 0.05

    # Server Configuration
    server_url: str = "http://127.0.0.1:8000"  
    server_api_key: str = "your-api-key-here"
    server_timeout: int = 30
    
    frame_skip: int = 2  
    max_queue_size: int = 100
    detection_interval: float = 0.2
    preview_interval: float = 0.01
    
    
    person_exit_edge_margin: int = 1
    person_exit_edge_frames: int = 5
    person_exit_margin: int = 30  

    # yolo11m-pose (balanced) or yolo11l-pose (high accuracy)
    yolo_pose_model_path: str = "yolo11m-pose.pt"
    pose_confidence_threshold: float = 0.5 
    

    # "botsort.yaml" (default, more accurate)/"bytetrack.yaml" (faster)
    yolo_tracker: str = "bytetrack.yaml"  
    track_persist: bool = True  
    track_conf: float = 0.5
    track_iou: float = 0.5 

    gesture_chest_tolerance_px: int = 40
    gesture_proximity_px: int = 60
    gesture_dwell_min_s: float = 0.25
    gesture_dwell_max_s: float = 1.0
    pose_full_frame: bool = False
    

    tap_validity_window_seconds: float = 5.0      # How long tap is valid after detection
    tap_max_age_seconds: float = 5.0 
    tap_allow_reuse: bool = False    
    tap_max_attempts: int = 5        
    tap_cooldown_after_use_seconds: float = 2.0
    
    tap_require_arm_extension: bool = False       # Require arm to be extended forward (strict)
    tap_require_torso_level: bool = False         # Require wrist at torso level (strict)
    tap_min_stable_frames: int = 1                # Minimum frames for stability
    tap_min_stable_duration: float = 0.1         # Minimum stable duration in seconds
    tap_variance_threshold: float = 40.0          # Max position variance in pixels
    tap_enable_debug_logging: bool = True         # Log why tap detection fails
    tap_wrist_offset_px: int = 30


    pose_overlay_enabled: bool = True
    pose_overlay_min_confidence: float = 0.3
    pose_overlay_infer_if_missing: bool = True
    pose_overlay_draw_angles: bool = True


    detection_batch_interval: float = 1.0 
    evasion_send_immediately: bool = True 
    

    log_level: str = "INFO"
    log_file: str = "logs/edge_device.log"
    enable_console_logs: bool = False  
    console_log_level: str = "WARNING"
    file_log_level: str = "DEBUG"
    

    snapshot_retention_days: int = 30
    max_snapshot_size_mb: int = 5
    
    class Config:
        env_file = ".env"
        env_prefix = "SNITCH_"


# Global settings instance
settings = Settings()
