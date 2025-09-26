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
        self.downscale_ratio = 0.5  # dev: process smaller frames
        
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
            # Downscale for speed
            small = cv2.resize(frame, None, fx=self.downscale_ratio, fy=self.downscale_ratio, interpolation=cv2.INTER_LINEAR)
            
            # Run inference
            results = self.model(small,
                               conf=settings.detection_confidence,
                               iou=settings.detection_iou_threshold,
                               classes=settings.detection_classes,
                               verbose=False)
            
            detections = []
            scale = 1.0 / self.downscale_ratio
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Scale boxes back to original frame size
                        x1, y1, x2, y2 = [int(v * scale) for v in [x1, y1, x2, y2]]
                        
                        detection = {
                            "bbox": [x1, y1, x2, y2],
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
        
        for detection in detections[:20]:  # avoid drawing too many
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
            "input_size": getattr(self.model.model[-1], 'imgsz', "unknown"),
            "classes": len(self.model.names),
            "class_names": list(self.model.names.values())
        }
