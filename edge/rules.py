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
from typing import Dict, Any, Tuple, Optional

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
        # Simplified method - no ROI or pose processing
        track_id = int(detection.get("person_id", -1))
        if track_id < 0:
            return
        # Ensure track exists but don't update any state
        self._get_track(track_id)

    def on_crossing(self, track_id: int) -> Dict[str, Any]:
        # Without pose/gesture detection, always classify as evasion
        return {"classification": "evasion", "tap_confirmed": False}

    def get_status(self, track_id: int) -> Dict[str, Any]:
        return self._get_track(track_id).copy()




