"""
Rule-based tap gesture detection using wrist/elbow/shoulder keypoints and ROI.

Expose detect_tap(keypoints, validator_roi, history) -> bool, metrics

validator_roi: (x1, y1, x2, y2) in frame pixels
keypoints: dict from PoseEstimator.infer_pose()
history: mutable dict per track_id to maintain temporal evidence
"""
from typing import Dict, Tuple, Optional, Any



def detect_tap(pose: Dict[str, Any], validator_roi: Tuple[int, int, int, int], history: Dict[str, Any], now_ts: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Gesture detection disabled - always returns False.
    """
    return False, {"reason": "gesture_detection_disabled"}


