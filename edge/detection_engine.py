"""
YOLO detection engine for person detection and pose estimation.
"""
import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
from loguru import logger
from config import settings
from ocsort import OCSortTracker, iou as iou_calc
from roi import point_in_roi, crosses_line


class YOLOPoseEstimator:
    """YOLO11-based pose estimation for person tracking.
    
    Uses YOLO11-pose models which provide 17 keypoints:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    """

    def __init__(self):
        """Initialize YOLO pose model."""
        self.model = None
        self.is_initialized = False
        # YOLO keypoint mapping (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        logger.info(f"Initializing YOLO pose estimator with model: {settings.yolo_pose_model_path}")

    def initialize(self) -> bool:
        """Load YOLO pose model."""
        try:
            self.model = YOLO(settings.yolo_pose_model_path)
            self.is_initialized = True
            logger.info(f"YOLO pose model loaded: {settings.yolo_pose_model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO pose model: {e}")
            return False

    def infer_pose(self, image: np.ndarray, bbox: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """
        Run pose estimation on an image or cropped person region.

        Args:
            image: BGR image array (full frame or cropped person)
            bbox: Optional bounding box [x1, y1, x2, y2] if image is full frame

        Returns:
            Dictionary with pose keypoints or None if no pose detected
            Format: {'keypoints': {name: {'x': int, 'y': int, 'z': float, 'visibility': float}}}
        """
        try:
            if not self.is_initialized:
                logger.warning("YOLO pose model not initialized")
                return None

            if image is None or image.size == 0:
                return None

            # Run YOLO pose inference
            results = self.model(image, conf=settings.pose_confidence_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return None

            result = results[0]
            
            # Check if keypoints are available
            if result.keypoints is None or result.keypoints.data is None:
                return None
            
            keypoints_data = result.keypoints.data
            
            if len(keypoints_data) == 0:
                return None
            
            # Use the first person detected (highest confidence)
            kpts = keypoints_data[0].cpu().numpy()  # Shape: (17, 3) -> [x, y, confidence]
            
            if len(kpts) < 17:
                return None
            
            # Convert to our expected format
            keypoints = {}
            h, w = image.shape[:2]
            
            for idx, name in enumerate(self.keypoint_names):
                x, y, conf = kpts[idx]
                keypoints[name] = {
                    'x': int(x),
                    'y': int(y),
                    'z': 0.0,  # YOLO doesn't provide depth
                    'visibility': float(conf)
                }
            
            return {
                'keypoints': keypoints,
                'timestamp': None
            }

        except Exception as e:
            logger.debug(f"YOLO pose estimation failed: {e}")
            return None


class TrackedPerson:
    """Represents a tracked person with ID and crossing state."""

    def __init__(self, person_id: int, bbox: List[int], frame_height: int, frame_width: int, gate_line_y: Optional[int] = None):
        self.person_id = person_id
        self.bbox = bbox
        self.last_seen_frame = 0
        self.crossed_gate = False
        self.crossing_direction = None  # "up" or "down"
        self.exited_frame = False  # Track if person has left the frame
        self.frames_at_edge = 0  # Track how long person has been at frame edge
        # Store gate line position in pixels (from ROI config)
        self.gate_line_y_pixels = gate_line_y
        # Track position (TOP edge y1)
        _, y1, _, _ = bbox
        self.previous_y1 = y1
        self.current_y1 = y1

    def update_position(self, bbox: List[int], frame_height: int, frame_width: int):
        """Update person's position and check for gate crossing."""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        
        # Track if this is the first update (to handle late detections)
        is_first_update = (self.previous_y1 == self.current_y1)
        
        # Update position tracking
        self.previous_y1 = self.current_y1
        self.current_y1 = y1

        # Check if person has left the frame boundaries
        margin = settings.person_exit_margin
        if (x2 < -margin or x1 > frame_width + margin or
            y2 < -margin or y1 > frame_height + margin):
            logger.debug(f"Person {self.person_id} exited frame completely (bbox: {bbox})")
            self.exited_frame = True
            return False  # Don't process gate crossing for exited persons

        # Check if person's bounding box is mostly outside the frame
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width > 0 and bbox_height > 0:
            visible_width = min(x2, frame_width) - max(x1, 0)
            visible_height = min(y2, frame_height) - max(y1, 0)
            visibility_ratio = (visible_width * visible_height) / (bbox_width * bbox_height)

            if visibility_ratio < 0.3:  # Less than 30% of bbox is visible
                logger.debug(f"Person {self.person_id} mostly outside frame (visibility: {visibility_ratio:.2f})")
                self.exited_frame = True
                return False

        # Check if bounding box is very small (person far away)
        if bbox_width > 0 and bbox_height > 0 and (bbox_width < 20 or bbox_height < 20):
            logger.debug(f"Person {self.person_id} bbox too small ({bbox_width}x{bbox_height})")
            self.exited_frame = True
            return False

        # Also check if person is very close to frame edges (might be partially exiting)
        edge_margin = settings.person_exit_edge_margin
        if (x2 < edge_margin or x1 > frame_width - edge_margin or
            y2 < edge_margin or y1 > frame_height - edge_margin):
            # If they've been at the edge for multiple frames, consider them exited
            if not hasattr(self, 'frames_at_edge'):
                self.frames_at_edge = 0
            self.frames_at_edge += 1
            if self.frames_at_edge > settings.person_exit_edge_frames:
                logger.debug(f"Person {self.person_id} exited after {self.frames_at_edge} frames at edge (bbox: {bbox})")
                self.exited_frame = True
                return False
        else:
            # Reset counter if not at edge
            if hasattr(self, 'frames_at_edge'):
                self.frames_at_edge = 0

        # Check for crossing (vertical movement across horizontal gate line)
        # Only detect down->up crossings (top edge crosses upward)
        if self.gate_line_y_pixels is None:
            # Fallback to normalized position if gate line not provided
            gate_line_pixels = frame_height * settings.gate_crossing_line_y
        else:
            # Use actual gate line from ROI config (in pixels)
            gate_line_pixels = self.gate_line_y_pixels
        
        hysteresis_pixels = settings.gate_crossing_hysteresis

        # Person moving from down to up (bottom of frame to top)
        # previous_y1 is higher value (lower on screen), current_y1 is lower value (higher on screen)
        crossed_upward = (self.previous_y1 > gate_line_pixels + hysteresis_pixels and 
                         self.current_y1 < gate_line_pixels - hysteresis_pixels)
        
        # SPECIAL CASE: Person detected when already past gate line (late detection)
        # If first update and already above gate line, trigger immediate crossing
        if is_first_update and not self.crossed_gate:
            if self.current_y1 < gate_line_pixels - hysteresis_pixels:
                self.crossed_gate = True
                self.crossing_direction = "up"
                logger.warning(
                    f"Person {self.person_id} detected AFTER gate line (y={gate_line_pixels}px) "
                    f"at position {self.current_y1}px - triggering immediate crossing validation"
                )
                return True
        
        # Normal crossing detection (transition based)
        if crossed_upward and not self.crossed_gate:
            self.crossed_gate = True
            self.crossing_direction = "up"
            logger.info(
                f"Person {self.person_id} TOP EDGE crossed gate line (y={gate_line_pixels}px) "
                f"from down to up: {self.previous_y1}px → {self.current_y1}px (entering - needs validation)"
            )
            return True
        
        # Ignore up->down crossings (person exiting)
        # We don't track or validate exits
        
        return False



class GateCrossingMonitor:
    """
    Monitors gate crossings and frame exits for tracked persons.
    Does NOT assign IDs - uses IDs from OC-SORT tracker.
    """

    def __init__(self, gate_line: Optional[List[int]] = None):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        # Extract gate line y-coordinate from ROI config (horizontal line: y1 == y2)
        self.gate_line_y = None
        if gate_line and len(gate_line) >= 4:
            # gate_line format: [x1, y1, x2, y2]
            y1, y2 = gate_line[1], gate_line[3]
            if y1 == y2:  # Horizontal line
                self.gate_line_y = y1
                logger.info(f"GateCrossingMonitor initialized with gate line at y={self.gate_line_y}px")

    def update(self, detections: List[Dict[str, Any]], frame_height: int, frame_width: int) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Update crossing state for detections (IDs already assigned by OC-SORT).
        Returns: (detections, crossed_person_ids)
        """
        crossed_person_ids = []

        for detection in detections:
            person_id = detection.get("person_id")
            if person_id is None:
                continue
                
            bbox = detection["bbox"]
            
            # Get or create tracked person (using ID from OC-SORT)
            if person_id not in self.tracked_persons:
                # New person from OC-SORT - create tracking state with gate line position
                self.tracked_persons[person_id] = TrackedPerson(
                    person_id, bbox, frame_height, frame_width, gate_line_y=self.gate_line_y
                )
            
            # Update position and check for crossing
            person = self.tracked_persons[person_id]
            crossed = person.update_position(bbox, frame_height, frame_width)
            
            if crossed:
                crossed_person_ids.append(person_id)

        # Clean up exited persons
        to_remove = []
        for person_id, person in self.tracked_persons.items():
            if person.exited_frame:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.tracked_persons[person_id]

        return detections, crossed_person_ids


class DetectionEngine:
    """YOLO-based person detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.downscale_ratio = settings.detection_downscale_ratio
        self._focal_pixels = None
        # Load ROI configuration first (needed for gate monitor)
        self.validator_roi = None
        self.gate_line = None
        self._load_roi_config()
        
        # Gate crossing monitor - tracks crossing state using IDs from OC-SORT
        self.gate_monitor = GateCrossingMonitor(gate_line=self.gate_line) if settings.enable_gate_crossing else None
        
        # OC-SORT trackers per camera for stable ID assignment
        self.ocsort_trackers: Dict[int, OCSortTracker] = {}

        # Initialize pose estimator
        try:
            self.pose_estimator = YOLOPoseEstimator()
            logger.info("YOLO pose estimator created (will initialize with detection engine)")
        except Exception as e:
            logger.error(f"Failed to create YOLO pose estimator: {e}")
            self.pose_estimator = None

    def _load_roi_config(self):
        """Load ROI configuration from YAML file."""
        roi_config_path = os.path.join(os.path.dirname(__file__), "config", "rois.yaml")
        try:
            if os.path.exists(roi_config_path):
                with open(roi_config_path, 'r') as f:
                    config = yaml.safe_load(f)

                self.validator_roi = config.get('validator_roi')
                self.gate_line = config.get('gate_line')

                if self.validator_roi:
                    logger.info(f"Loaded validator ROI: {self.validator_roi}")
                if self.gate_line:
                    logger.info(f"Loaded gate line: {self.gate_line}")
            else:
                logger.warning(f"ROI config file not found: {roi_config_path}")
        except Exception as e:
            logger.error(f"Failed to load ROI config: {e}")

    async def initialize(self) -> bool:
        """Initialize YOLO model and pose estimator."""
        try:
            # Load YOLOv8 detection model
            self.model = YOLO(settings.yolo_model_path)
            self.is_initialized = True
            self._compute_focal_length_pixels()
            logger.info("YOLO detection model initialized successfully")
            
            # Initialize pose estimator
            if self.pose_estimator and not self.pose_estimator.is_initialized:
                if not self.pose_estimator.initialize():
                    logger.error("Failed to initialize YOLO pose estimator")
                    self.pose_estimator = None
            
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

                        # Filter out boxes that are implausible for a person
                        w = max(1, bbox[2] - bbox[0])
                        h = max(1, bbox[3] - bbox[1])
                        aspect = w / h
                        if aspect < 0.2 or aspect > 4.0:
                            # Extremely tall/skinny or wide/flat - likely a false positive
                            continue
                        if w * h < 500:  # tiny boxes
                            continue

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
    
    async def detect_multiple_frames(self, frames: List[Tuple[int, np.ndarray]]) -> Tuple[Dict[int, List[Dict[str, Any]]], List[int]]:
        """
        Detect persons in multiple frames from different cameras.
        Returns: (detection_results, crossed_person_ids)
        """
        results = {}
        all_crossed_person_ids = []

        for camera_id, frame in frames:
            detections = await self.detect_persons(frame)

            # OC-SORT: assign track IDs for this camera's detections
            if detections:
                det_array = np.array([d["bbox"] for d in detections], dtype=float)
            else:
                det_array = np.empty((0, 4), dtype=float)

            # Get tracker for this camera
            tracker = self.ocsort_trackers.get(camera_id)
            if tracker is None:
                # CRITICAL: Keep tracks alive as long as person is in frame
                # max_age set very high - we'll manually clean up when person exits frame
                tracker = OCSortTracker(
                    max_age=300,     # Keep tracks alive for 300 frames (~10 seconds) - only forget if truly lost
                    min_hits=1,      # Assign ID immediately
                    iou_threshold=0.05,  # Very permissive matching - prevents ID switches
                )
                self.ocsort_trackers[camera_id] = tracker
                logger.info(f"Initialized OC-SORT tracker for camera {camera_id} with persistent params (max_age=300, min_hits=1, iou=0.05)")

            tracked = tracker.update(det_array)
            # Map from bbox to id with a tolerance using IoU to pair back to detection dicts
            id_assigned = [False] * len(detections)
            for trk_bbox, trk_id in tracked:
                # Find best matching detection by IoU
                best_idx = -1
                best_iou = 0.0
                for idx, det in enumerate(detections):
                    if id_assigned[idx]:
                        continue
                    iou_val = iou_calc(np.array([int(v) for v in trk_bbox], dtype=float), np.array(det["bbox"], dtype=float))
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = idx
                # More permissive matching threshold (was 0.1, now 0.05) for stable ID assignment
                if best_idx >= 0 and best_iou >= 0.05:
                    detections[best_idx]["person_id"] = int(trk_id)
                    id_assigned[best_idx] = True

            # Fallback: any unassigned detections get unique incremental IDs beyond current max
            current_max_id = max([int(t[1]) for t in tracked], default=0)
            next_id = current_max_id + 1
            for idx, assigned in enumerate(id_assigned):
                if not assigned and idx < len(detections):
                    detections[idx]["person_id"] = int(next_id)
                    logger.info(f"👤 New person detected - assigned ID {next_id}")
                    next_id += 1

            # Monitor gate crossings and frame exits (using IDs from OC-SORT)
            crossed_person_ids = []
            if self.gate_monitor:
                frame_height, frame_width = frame.shape[:2]
                detections, crossed_person_ids = self.gate_monitor.update(detections, frame_height, frame_width)
                all_crossed_person_ids.extend(crossed_person_ids)
                
                # CRITICAL: Clean up OC-SORT tracks for persons who have exited the frame
                # This ensures track IDs are only forgotten when person leaves, not based on time
                exited_person_ids = []
                for person_id, person in self.gate_monitor.tracked_persons.items():
                    if person.exited_frame:
                        exited_person_ids.append(person_id)
                
                # Force OC-SORT to forget these tracks by checking against active trackers
                if exited_person_ids and hasattr(tracker, 'trackers'):
                    to_remove_indices = []
                    for idx, trk in enumerate(tracker.trackers):
                        trk_id = int(trk.id)
                        if trk_id in exited_person_ids:
                            to_remove_indices.append(idx)
                            logger.info(f"🚪 Person {trk_id} exited frame - releasing track ID")
                    
                    # Remove tracks in reverse order to avoid index issues
                    for idx in sorted(to_remove_indices, reverse=True):
                        del tracker.trackers[idx]
            
            results[camera_id] = detections

        return results, all_crossed_person_ids
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        annotated_frame = frame.copy()


        # Draw validator ROI if loaded
        if self.validator_roi:
            x1, y1, x2, y2 = self.validator_roi
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Cyan color, thicker line
            cv2.putText(annotated_frame, "Validator ROI", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw gate line from ROI config if loaded
        if self.gate_line:
            x1, y1, x2, y2 = self.gate_line
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow color, thicker line
            cv2.putText(annotated_frame, "Gate Line (ROI)", (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw pose keypoints for tracks with pose data
        self._draw_pose_keypoints(annotated_frame, detections)

        for detection in detections[:20]:  # avoid drawing too many
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            distance = detection.get("distance_m", -1.0)
            person_id = detection.get("person_id")

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Build label with person ID if available
            if person_id is not None:
                label_parts = [f"ID:{person_id}"]
            else:
                label_parts = []

            label_parts.append(f"{confidence:.2f}")
            if distance >= 0:
                label_parts.append(f"{distance:.1f}m")

            label = " ".join(label_parts)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame

    def _draw_pose_keypoints(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Draw pose keypoints and skeleton for tracks with pose data."""
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            pose_keypoints = detection.get("pose_keypoints")
            if not pose_keypoints:
                continue

            # Helper function to check if keypoint is visible in frame
            def is_keypoint_visible(kp: Dict[str, Any], margin: int = 10) -> bool:
                """Check if keypoint is within frame boundaries with margin."""
                x, y = int(kp['x']), int(kp['y'])
                return (margin <= x < frame_width - margin and
                        margin <= y < frame_height - margin and
                        kp['visibility'] > 0.5)

            # Define key connections for skeleton (simplified)
            skeleton_connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
                ('nose', 'left_shoulder'),
                ('nose', 'right_shoulder'),
            ]

            # Draw skeleton connections only for visible keypoints
            for start_name, end_name in skeleton_connections:
                if start_name in pose_keypoints and end_name in pose_keypoints:
                    start_kp = pose_keypoints[start_name]
                    end_kp = pose_keypoints[end_name]

                    # Only draw connection if BOTH keypoints are visible in frame
                    if is_keypoint_visible(start_kp) and is_keypoint_visible(end_kp):
                        cv2.line(frame,
                                (int(start_kp['x']), int(start_kp['y'])),
                                (int(end_kp['x']), int(end_kp['y'])),
                                (255, 255, 255), 2)  # White skeleton lines

            # Highlight key points only if they're visible in frame
            key_points = ['left_wrist', 'right_wrist', 'left_elbow', 'right_elbow',
                         'left_shoulder', 'right_shoulder']

            for point_name in key_points:
                if point_name in pose_keypoints:
                    kp = pose_keypoints[point_name]
                    if is_keypoint_visible(kp):
                        # Color code different body parts
                        if 'wrist' in point_name:
                            color = (0, 255, 0)  # Green for wrists
                            radius = 6
                        elif 'elbow' in point_name:
                            color = (0, 165, 255)  # Orange for elbows
                            radius = 5
                        else:  # shoulders
                            color = (255, 0, 0)  # Red for shoulders
                            radius = 4

                        cv2.circle(frame, (int(kp['x']), int(kp['y'])), radius, color, -1)  # Filled circle
                        cv2.circle(frame, (int(kp['x']), int(kp['y'])), radius + 2, (255, 255, 255), 2)  # White outline

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
