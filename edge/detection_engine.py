"""
YOLO detection engine for person detection.
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
import mediapipe as mp


class PoseEstimator:
    """MediaPipe-based pose estimation for person tracking."""

    def __init__(self):
        # Use static image mode for better per-frame detection
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,  # Better for individual detections
            model_complexity=settings.mediapipe_model_complexity,
            min_detection_confidence=settings.mediapipe_min_detection_confidence,
            min_tracking_confidence=settings.mediapipe_min_tracking_confidence,
            enable_segmentation=False,  # Faster without segmentation
            smooth_landmarks=False  # Not needed for static images
        )
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        logger.info("PoseEstimator initialized with static_image_mode=True for robust detection")

    def infer_pose(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Run pose estimation on an image.

        Args:
            image: RGB image array

        Returns:
            Dictionary with pose keypoints or None if no pose detected
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                logger.debug("Pose estimation: Empty or None image")
                return None
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Run pose estimation
            results = self.pose.process(rgb_image)

            if results.pose_landmarks:
                keypoints = {}
                h, w = image.shape[:2]

                # Extract key landmarks (wrists, elbows, shoulders)
                landmark_indices = {
                    'nose': 0,
                    'left_shoulder': 11,
                    'right_shoulder': 12,
                    'left_elbow': 13,
                    'right_elbow': 14,
                    'left_wrist': 15,
                    'right_wrist': 16,
                    'left_hip': 23,
                    'right_hip': 24,
                    'left_knee': 25,
                    'right_knee': 26,
                    'left_ankle': 27,
                    'right_ankle': 28
                }

                for name, idx in landmark_indices.items():
                    landmark = results.pose_landmarks.landmark[idx]
                    keypoints[name] = {
                        'x': int(landmark.x * w),
                        'y': int(landmark.y * h),
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }

                return {
                    'keypoints': keypoints,
                    'landmarks': results.pose_landmarks,
                    'timestamp': None
                }

        except Exception as e:
            logger.debug(f"Pose estimation failed: {e}")

        return None


class TrackedPerson:
    """Represents a tracked person with ID and crossing state."""

    def __init__(self, person_id: int, bbox: List[int], frame_height: int, frame_width: int):
        self.person_id = person_id
        self.bbox = bbox
        self.last_seen_frame = 0
        self.crossed_gate = False
        self.crossing_direction = None  # "up" or "down"
        self.exited_frame = False  # Track if person has left the frame
        self.frames_at_edge = 0  # Track how long person has been at frame edge
        # Track position relative to gate line (use TOP edge y1 for horizontal gate line)
        _, y1, _, _ = bbox
        self.relative_position = y1 / frame_height  # 0-1 normalized (top edge)

    def update_position(self, bbox: List[int], frame_height: int, frame_width: int):
        """Update person's position and check for gate crossing."""
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        # Use TOP edge (y1) for crossing detection
        old_relative = self.relative_position
        self.relative_position = y1 / frame_height

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
        gate_line = settings.gate_crossing_line_y
        hysteresis = settings.gate_crossing_hysteresis / frame_height

        # Person moving from down to up (top edge crosses gate line upward)
        # old_relative is higher value (lower on screen), new is lower value (higher on screen)
        if old_relative > gate_line + hysteresis and self.relative_position < gate_line - hysteresis:
            if not self.crossed_gate:
                self.crossed_gate = True
                self.crossing_direction = "up"
                logger.info(f"Person {self.person_id} TOP EDGE crossed gate from down to up (entering - needs validation)")
                return True
        
        # Ignore up->down crossings (person exiting)
        # We don't track or validate exits
        
        return False



class PersonTracker:
    """Simple person tracker using bounding box overlap."""

    def __init__(self):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.next_person_id = 1
        self.frame_count = 0

    def update(self, detections: List[Dict[str, Any]], frame_height: int, frame_width: int) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Update tracking with new detections.
        Returns: (enriched_detections, crossed_person_ids)
        """
        self.frame_count += 1
        crossed_person_ids = []

        # Do not touch last_seen_frame here. We only update it when a
        # detection is actually matched to a tracked person. This allows
        # tracks to age-out correctly when not seen.

        # Match detections to existing tracks (excluding exited persons)
        matched_track_ids = set()
        enriched_detections = []

        for detection in detections:
            bbox = detection["bbox"]
            best_match_id = None
            best_iou = 0

            # Find best matching existing track
            for person_id, person in self.tracked_persons.items():
                if person_id in matched_track_ids:
                    continue

                # Skip persons who have exited the frame - they should never be matched again
                if person.exited_frame:
                    continue

                iou = self._calculate_iou(bbox, person.bbox)
                frames_since_seen = self.frame_count - person.last_seen_frame

                # Use lower threshold for recently seen tracks (coasting)
                threshold = settings.person_tracking_iou_threshold
                if frames_since_seen <= 3:  # Recently seen
                    threshold = max(0.05, threshold * 0.5)  # Much more permissive

                if iou > threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = person_id

            if best_match_id is not None:
                # Update existing track
                person = self.tracked_persons[best_match_id]
                crossed = person.update_position(bbox, frame_height, frame_width)
                # Mark as seen on this frame
                person.last_seen_frame = self.frame_count
                if crossed:
                    crossed_person_ids.append(best_match_id)
                matched_track_ids.add(best_match_id)
            else:
                # Create new track
                new_person = TrackedPerson(self.next_person_id, bbox, frame_height, frame_width)
                self.tracked_persons[self.next_person_id] = new_person
                matched_track_ids.add(self.next_person_id)
                self.next_person_id += 1

            # Enrich detection with tracking info
            enriched_detection = dict(detection)
            enriched_detection["person_id"] = best_match_id if best_match_id else self.next_person_id - 1

            # Only include non-exited persons in the results
            if best_match_id is not None:
                person = self.tracked_persons[best_match_id]
                if not person.exited_frame:
                    enriched_detections.append(enriched_detection)
            else:
                # New person - include them
                enriched_detections.append(enriched_detection)

        # Clean up old tracks and exited persons
        to_remove = []
        for person_id, person in self.tracked_persons.items():
            frames_since_seen = self.frame_count - person.last_seen_frame

            # Remove persons who have exited the frame
            if person.exited_frame:
                logger.debug(f"Removing exited person {person_id} from tracking")
                to_remove.append(person_id)
                continue

            # Remove persons who haven't been seen for too long
            if frames_since_seen > settings.person_tracking_max_age:
                to_remove.append(person_id)

        for person_id in to_remove:
            del self.tracked_persons[person_id]

        return enriched_detections, crossed_person_ids

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate intersection over union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


class DetectionEngine:
    """YOLO-based person detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.downscale_ratio = settings.detection_downscale_ratio
        self._focal_pixels = None
        # Legacy simple tracker retained for gate-crossing logic if needed
        self.person_tracker = PersonTracker() if settings.enable_gate_crossing else None
        # OC-SORT style trackers per camera for stable IDs
        self.ocsort_trackers: Dict[int, OCSortTracker] = {}

        # Load ROI configuration
        self.validator_roi = None
        self.gate_line = None
        self._load_roi_config()

        # Initialize pose estimator
        try:
            self.pose_estimator = PoseEstimator()
            logger.info("MediaPipe pose estimator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe pose estimator: {e}")
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

            # Maintain gate-crossing logic using legacy tracker state if enabled
            crossed_person_ids = []
            if self.person_tracker:
                frame_height, frame_width = frame.shape[:2]
                tracked_detections, crossed_person_ids = self.person_tracker.update(detections, frame_height, frame_width)
                results[camera_id] = tracked_detections
                all_crossed_person_ids.extend(crossed_person_ids)
                
                # CRITICAL: Clean up OC-SORT tracks for persons who have exited the frame
                # This ensures track IDs are only forgotten when person leaves, not based on time
                exited_person_ids = []
                for person_id, person in self.person_tracker.tracked_persons.items():
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
            else:
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
