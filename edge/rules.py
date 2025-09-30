"""
Rule engine: combines ROI state, pose-based tap detection, and gate crossing.

Maintains per-track state:
- inside_validator_roi
- tap_confirmed
- last_pose_ts
- gesture_history (for temporal detection)

Public API:
    ValidatorAndGateRules(settings_like) with:
        update_track(frame, detection, validator_roi) -> None
        on_crossing(track_id) -> classification dict {"pass"|"evasion"|"unknown"}
        get_status(track_id) -> dict
"""
from typing import Dict, Any, Tuple, Optional, List
import time

from loguru import logger



class ValidatorAndGateRules:
    def __init__(self, config):
        self.cfg = config
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def _get_track(self, track_id: int) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if st is None:
            st = {
                "tap_confirmed": False,
                "inside_validator_roi": False,
                "gesture_history": {},
                "wrist_history": [],  # Store wrist positions over time for stability detection
                "last_tap_check": 0.0,  # Timestamp of last tap gesture check
                
                # Tap lifecycle tracking (security enhancement)
                "tap_timestamp": None,           # When tap was detected
                "tap_expires_at": None,          # When tap expires
                "tap_used": False,               # Whether tap was already used for crossing
                "tap_usage_timestamp": None,     # When tap was consumed
                "tap_attempts": 0,               # Number of tap attempts (fraud detection)
                "crossings_count": 0,            # Total crossings by this person
                "last_crossing_timestamp": None, # Last crossing time
            }
            self._tracks[track_id] = st
        return st


    def update_track(self, frame, detection: Dict[str, Any], validator_roi: Tuple[int, int, int, int]) -> None:
        """Update track state with ROI information."""
        track_id = int(detection.get("person_id", -1))
        if track_id < 0:
            return

        track = self._get_track(track_id)

        # Check if person is inside validator ROI
        bbox = detection.get("bbox", [])
        if len(bbox) >= 4:
            # Use center of bounding box for ROI check
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            inside_roi = self._point_in_roi((int(center_x), int(center_y)), validator_roi)

            track["inside_validator_roi"] = inside_roi

            # Update last seen timestamp
            import time
            track["last_seen_ts"] = time.time()

            # Check for tap gesture if pose data is available
            pose_keypoints = detection.get("pose_keypoints")
            if pose_keypoints:
                self._check_tap_gesture(track_id, pose_keypoints, validator_roi)

    def _check_tap_gesture(self, track_id: int, pose_keypoints: Dict[str, Any], validator_roi: Tuple[int, int, int, int]) -> None:
        """Check for tap gesture based on wrist position and stability."""
        track = self._get_track(track_id)
        current_time = time.time()
        
        # SECURITY: Don't re-detect tap if already confirmed and not expired
        if track.get("tap_confirmed") and not track.get("tap_used"):
            tap_expires = track.get("tap_expires_at", 0)
            if current_time < tap_expires:
                # Tap still valid, don't process new tap
                return
            else:
                # Tap expired, allow new tap detection
                logger.debug(f"Track {track_id}: Previous tap expired, allowing new detection")
                self._reset_tap_state(track)
        
        # SECURITY: Cooldown period after tap usage
        last_usage = track.get("tap_usage_timestamp")
        if last_usage:
            cooldown = getattr(self.cfg, 'tap_cooldown_after_use_seconds', 2.0)
            time_since_usage = current_time - last_usage
            if time_since_usage < cooldown:
                # Too soon after last tap usage
                return

        # Determine dominant wrist (higher confidence)
        left_wrist = pose_keypoints.get('left_wrist')
        right_wrist = pose_keypoints.get('right_wrist')

        if not left_wrist or not right_wrist:
            return  # Need both wrists for comparison

        # Choose dominant wrist based on visibility confidence
        if left_wrist['visibility'] > right_wrist['visibility']:
            dominant_wrist = left_wrist
            dominant_name = 'left_wrist'
            elbow_name = 'left_elbow'
            shoulder_name = 'left_shoulder'
        else:
            dominant_wrist = right_wrist
            dominant_name = 'right_wrist'
            elbow_name = 'right_elbow'
            shoulder_name = 'right_shoulder'

        wrist_x, wrist_y = dominant_wrist['x'], dominant_wrist['y']

        # Rule 1: Wrist must be inside validator_roi
        if not self._point_in_roi((int(wrist_x), int(wrist_y)), validator_roi):
            return

        # Get elbow and shoulder positions for forward extension check
        elbow = pose_keypoints.get(elbow_name)
        shoulder = pose_keypoints.get(shoulder_name)

        if not elbow or not shoulder:
            return

        # Rule 2: Wrist horizontally extended forward (ahead of elbow & shoulder)
        # For right-side dominant wrist, check if wrist_x > elbow_x > shoulder_x (moving right)
        # For left-side dominant wrist, check if wrist_x < elbow_x < shoulder_x (moving left)
        if dominant_name == 'right_wrist':
            extended_forward = wrist_x > elbow['x'] > shoulder['x']
        else:  # left_wrist
            extended_forward = wrist_x < elbow['x'] < shoulder['x']

        if not extended_forward:
            return

        # Rule 3: Wrist vertical position ~ torso/waist level
        # Check if wrist Y is between shoulder Y and approximate hip level
        shoulder_y = shoulder['y']
        # Estimate hip level (roughly 1.5x shoulder-to-elbow distance below shoulder)
        elbow_to_shoulder_dist = abs(elbow['y'] - shoulder_y)
        estimated_hip_y = shoulder_y + (elbow_to_shoulder_dist * 1.5)

        # Wrist should be between shoulder level and hip level (with some tolerance)
        torso_level = shoulder_y <= wrist_y <= estimated_hip_y

        if not torso_level:
            return

        # Rule 4: Wrist remains stable for 0.3-0.5s
        # Add current wrist position to history
        wrist_position = {'x': wrist_x, 'y': wrist_y, 'timestamp': current_time}
        track['wrist_history'].append(wrist_position)

        # Keep only last 1 second of wrist history
        track['wrist_history'] = [pos for pos in track['wrist_history']
                                 if current_time - pos['timestamp'] < 1.0]

        # Need at least 0.3 seconds of history for stability check
        if len(track['wrist_history']) < 3:  # At least 3 frames
            return

        # Calculate position variance over the last 0.3-0.5 seconds
        recent_positions = [pos for pos in track['wrist_history']
                           if current_time - pos['timestamp'] <= 0.5]

        if len(recent_positions) < 3:
            return

        # Calculate variance in wrist position
        x_positions = [pos['x'] for pos in recent_positions]
        y_positions = [pos['y'] for pos in recent_positions]

        x_variance = max(x_positions) - min(x_positions)
        y_variance = max(y_positions) - min(y_positions)
        total_variance = (x_variance ** 2 + y_variance ** 2) ** 0.5

        # Stability threshold (pixels)
        stability_threshold = 15.0  # Adjust as needed

        wrist_stable = total_variance < stability_threshold

        # Check if stable for at least 0.3 seconds
        stability_duration = current_time - recent_positions[0]['timestamp']
        stable_duration_met = stability_duration >= 0.3

        # If all conditions met, confirm tap
        if wrist_stable and stable_duration_met:
            # SECURITY: Set tap with expiration and lifecycle tracking
            TAP_VALIDITY_WINDOW = getattr(self.cfg, 'tap_validity_window_seconds', 5.0)
            
            track["tap_confirmed"] = True
            track["tap_timestamp"] = current_time
            track["tap_expires_at"] = current_time + TAP_VALIDITY_WINDOW
            track["tap_used"] = False  # Mark as unused
            track["tap_attempts"] += 1  # Increment attempt counter
            
            logger.info(
                f"✓ Tap DETECTED for track {track_id} "
                f"(expires in {TAP_VALIDITY_WINDOW}s, attempt #{track['tap_attempts']})"
            )
            logger.debug(
                f"Tap details: wrist stable for {stability_duration:.2f}s, "
                f"variance: {total_variance:.1f}px"
            )

    def _point_in_roi(self, point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        """Check if a point is within a bounding box ROI."""
        x, y = point
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_crossing(self, track_id: int) -> Dict[str, Any]:
        """Classify crossing based on tap gesture confirmation with multi-layer security validation."""
        track = self._get_track(track_id)
        current_time = time.time()
        
        # Get tap state
        tap_confirmed = track.get("tap_confirmed", False)
        tap_timestamp = track.get("tap_timestamp")
        tap_expires_at = track.get("tap_expires_at")
        tap_used = track.get("tap_used", False)
        
        # Update crossing metadata
        track["crossings_count"] += 1
        track["last_crossing_timestamp"] = current_time
        
        # === MULTI-LAYER SECURITY VALIDATION ===
        
        # LAYER 1: Was there a tap?
        if not tap_confirmed or tap_timestamp is None:
            logger.warning(
                f"❌ Track {track_id} EVASION: No tap detected "
                f"(crossing #{track['crossings_count']})"
            )
            return {
                "classification": "evasion",
                "tap_confirmed": False,
                "reason": "no_tap",
                "crossing_count": track["crossings_count"],
                "fraud_indicators": self._check_fraud_indicators(track_id, track)
            }
        
        # LAYER 2: Has tap already been used?
        if tap_used:
            tap_usage_time = track.get("tap_usage_timestamp")
            time_since_usage = current_time - tap_usage_time if tap_usage_time else 0
            
            logger.warning(
                f"❌ Track {track_id} EVASION: Tap already used "
                f"({time_since_usage:.1f}s ago, crossing #{track['crossings_count']})"
            )
            return {
                "classification": "evasion",
                "tap_confirmed": False,
                "reason": "tap_already_used",
                "tap_reuse_attempt": True,
                "crossing_count": track["crossings_count"],
                "fraud_indicators": self._check_fraud_indicators(track_id, track)
            }
        
        # LAYER 3: Has tap expired?
        if current_time > tap_expires_at:
            time_expired = current_time - tap_expires_at
            
            logger.warning(
                f"❌ Track {track_id} EVASION: Tap expired "
                f"({time_expired:.1f}s ago, crossing #{track['crossings_count']})"
            )
            
            # Clean up expired tap
            self._reset_tap_state(track)
            
            return {
                "classification": "evasion",
                "tap_confirmed": False,
                "reason": "tap_expired",
                "expired_by_seconds": time_expired,
                "crossing_count": track["crossings_count"],
                "fraud_indicators": self._check_fraud_indicators(track_id, track)
            }
        
        # LAYER 4: Temporal ordering - tap must happen BEFORE crossing
        tap_age = current_time - tap_timestamp
        if tap_age < 0:
            # This should never happen (clock issue)
            logger.error(f"⚠️ Track {track_id}: Tap timestamp in future! Possible clock issue")
            return {
                "classification": "evasion",
                "tap_confirmed": False,
                "reason": "invalid_timestamp",
                "crossing_count": track["crossings_count"],
                "fraud_indicators": self._check_fraud_indicators(track_id, track)
            }
        
        # LAYER 5: Tap too old? (redundant with expiration, but extra safety)
        MAX_TAP_AGE = getattr(self.cfg, 'tap_max_age_seconds', 5.0)
        if tap_age > MAX_TAP_AGE:
            logger.warning(
                f"❌ Track {track_id} EVASION: Tap too old "
                f"({tap_age:.1f}s, max {MAX_TAP_AGE}s)"
            )
            self._reset_tap_state(track)
            return {
                "classification": "evasion",
                "tap_confirmed": False,
                "reason": "tap_too_old",
                "tap_age_seconds": tap_age,
                "crossing_count": track["crossings_count"],
                "fraud_indicators": self._check_fraud_indicators(track_id, track)
            }
        
        # === ALL SECURITY CHECKS PASSED - VALID TAP ===
        
        # CRITICAL: Mark tap as USED (prevents reuse)
        track["tap_used"] = True
        track["tap_usage_timestamp"] = current_time
        
        fraud_flags = self._check_fraud_indicators(track_id, track)
        
        logger.info(
            f"✅ Track {track_id} VALID PASS: Tap confirmed and consumed "
            f"(tap age: {tap_age:.2f}s, crossing #{track['crossings_count']})"
        )
        
        # Optional: Reset tap state after use (belt-and-suspenders approach)
        # This ensures tap can't be accidentally reused even if logic fails
        self._reset_tap_state(track)
        
        return {
            "classification": "pass",
            "tap_confirmed": True,
            "tap_age_seconds": tap_age,
            "crossing_count": track["crossings_count"],
            "reason": "valid_tap",
            "fraud_indicators": fraud_flags
        }

    def _reset_tap_state(self, track: Dict[str, Any]) -> None:
        """Reset tap-related state for a track."""
        track["tap_confirmed"] = False
        track["tap_timestamp"] = None
        track["tap_expires_at"] = None
        track["tap_used"] = False
        track["wrist_history"].clear()
    
    def _check_fraud_indicators(self, track_id: int, track: Dict[str, Any]) -> List[str]:
        """Check for potential fraud/gaming attempts."""
        fraud_flags = []
        
        # Flag 1: Too many tap attempts
        max_attempts = getattr(self.cfg, 'tap_max_attempts', 5)
        if track.get("tap_attempts", 0) > max_attempts:
            fraud_flags.append("excessive_tap_attempts")
        
        # Flag 2: Multiple crossings with single tap attempt
        if track.get("crossings_count", 0) > 1 and track.get("tap_attempts", 0) == 1:
            fraud_flags.append("multiple_crossings_single_tap")
        
        # Flag 3: Tap reuse attempt detected
        if track.get("tap_used") and track.get("tap_confirmed"):
            fraud_flags.append("tap_reuse_attempt")
        
        if fraud_flags:
            logger.warning(
                f"🚨 Track {track_id} FRAUD INDICATORS: {', '.join(fraud_flags)}"
            )
        
        return fraud_flags

    def get_status(self, track_id: int) -> Dict[str, Any]:
        return self._get_track(track_id).copy()

    def cleanup_old_tracks(self, max_age_seconds: float = 30.0) -> None:
        """Remove tracks that haven't been seen for max_age_seconds."""
        import time
        current_time = time.time()
        to_remove = []

        for track_id, track in self._tracks.items():
            last_seen = track.get("last_seen_ts", 0)
            if current_time - last_seen > max_age_seconds:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self._tracks[track_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tracks from rules engine")




