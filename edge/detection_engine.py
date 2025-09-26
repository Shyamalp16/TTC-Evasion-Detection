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
        self.downscale_ratio = settings.detection_downscale_ratio
        self._focal_pixels = None
        
    async def initialize(self) -> bool:
        """Initialize YOLO model."""
        try:
            # Load YOLOv8-tiny model
            self.model = YOLO(settings.yolo_model_path)
            self.is_initialized = True
            self._compute_focal_length_pixels()
            logger.info("YOLO model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            return False
    
    def _compute_focal_length_pixels(self):
        """Pre-compute focal length in pixels from camera intrinsics."""
        try:
            # Sensor height in mm and focal length in mm
            sensor_height = settings.camera_sensor_height_mm
            focal_length_mm = settings.camera_focal_length_mm
            if sensor_height <= 0 or focal_length_mm <= 0:
                return

            # Using a default vertical resolution assumption (since frame height may vary)
            # We'll adjust focal length per-frame based on actual height.
            self._focal_pixels = {
                "focal_length_mm": focal_length_mm,
                "sensor_height_mm": sensor_height
            }
        except Exception as exc:
            logger.warning(f"Failed to pre-compute focal length: {exc}")
            self._focal_pixels = None

    def _estimate_distance(self, bbox: List[int], frame_height: int) -> float:
        """Estimate distance based on bounding box height and camera parameters."""
        if not self._focal_pixels:
            return -1.0

        try:
            y1, y2 = bbox[1], bbox[3]
            bbox_height_pixels = max(1, y2 - y1)
            sensor_height_mm = self._focal_pixels["sensor_height_mm"]
            focal_length_mm = self._focal_pixels["focal_length_mm"]

            # Convert focal length to pixel units using frame height
            focal_length_pixels = (focal_length_mm / sensor_height_mm) * frame_height

            distance_m = (settings.reference_person_height_m * focal_length_pixels) / bbox_height_pixels
            distance_m = float(min(max(distance_m, 0.0), settings.max_detection_distance_m))
            return distance_m
        except Exception as exc:
            logger.debug(f"Distance estimation error: {exc}")
            return -1.0

    async def detect_persons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons in a frame."""
        if not self.is_initialized:
            logger.error("Detection engine not initialized")
            return []
        
        try:
            # Downscale for speed
            if self.downscale_ratio != 1.0:
                small = cv2.resize(frame, None, fx=self.downscale_ratio, fy=self.downscale_ratio, interpolation=cv2.INTER_LINEAR)
            else:
                small = frame
            
            # Run inference
            results = self.model(small,
                               conf=settings.detection_confidence,
                               iou=settings.detection_iou_threshold,
                               classes=settings.detection_classes,
                               verbose=False)
            
            detections = []
            scale = 1.0 / self.downscale_ratio if self.downscale_ratio != 0 else 1.0
            frame_height = frame.shape[0]
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Scale boxes back to original frame size
                        x1, y1, x2, y2 = [int(v * scale) for v in [x1, y1, x2, y2]]
                        
                        bbox = [int(v * scale) for v in [x1, y1, x2, y2]]
                        distance = self._estimate_distance(bbox, frame_height)

                        detection = {
                            "bbox": bbox,
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": "person",
                            "distance_m": distance
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
            distance = detection.get("distance_m", -1.0)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if distance >= 0:
                label = f"Person {confidence:.2f} {distance:.1f}m"
            else:
                label = f"Person {confidence:.2f}"
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
