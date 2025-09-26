"""
YOLO detection engine for person detection.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
from loguru import logger
from config import settings


class DetectionEngine:
    """YOLO-based person detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize YOLO model."""
        try:
            # Load YOLOv8-tiny model
            self.model = YOLO('yolov8n.pt')
            self.is_initialized = True
            logger.info("YOLO model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    async def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in a frame."""
        if not self.is_initialized:
            logger.error("Detection engine not initialized")
            return []
        
        try:
            # Run inference
            results = self.model(frame, 
                               conf=settings.detection_confidence,
                               iou=settings.detection_iou_threshold,
                               classes=settings.detection_classes,
                               verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": "person"
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    async def detect_multiple_frames(self, frames: List[Tuple[int, np.ndarray]]) -> Dict[int, List[Dict[str, Any]]]:
        """Detect persons in multiple frames from different cameras."""
        results = {}
        
        for camera_id, frame in frames:
            detections = await self.detect_persons(frame)
            if detections:
                results[camera_id] = detections
                
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        return {
            "model_name": "YOLOv8-tiny",
            "input_size": self.model.model[-1].imgsz if hasattr(self.model.model[-1], 'imgsz') else "unknown",
            "classes": len(self.model.names),
            "class_names": list(self.model.names.values())
        }
