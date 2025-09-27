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

    def _point_in_roi(self, point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        """Check if a point is within a bounding box ROI."""
        x, y = point
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_crossing(self, track_id: int) -> Dict[str, Any]:
        # Without pose/gesture detection, always classify as evasion
        return {"classification": "evasion", "tap_confirmed": False}

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




