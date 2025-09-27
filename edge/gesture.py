"""
Rule-based tap gesture detection using wrist/elbow/shoulder keypoints and ROI.

Expose detect_tap(keypoints, validator_roi, history) -> bool, metrics

validator_roi: (x1, y1, x2, y2) in frame pixels
keypoints: dict from PoseEstimator.infer_pose()
history: mutable dict per track_id to maintain temporal evidence
"""
from typing import Dict, Tuple, Optional, Any
import time
import math

from config import settings

def _get_point(kps, name: str) -> Optional[Tuple[int, int, float]]:
    for kp in kps:
        if kp.get("name") == name:
            return int(kp.get("x", 0)), int(kp.get("y", 0)), float(kp.get("confidence", 0.0))
    return None


def _point_in_roi(x: int, y: int, roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def _horizontal_distance_to_roi(x: int, roi: Tuple[int, int, int, int]) -> int:
    x1, _, x2, _ = roi
    if x < x1:
        return x1 - x
    if x > x2:
        return x - x2
    return 0


def detect_tap(pose: Dict[str, Any], validator_roi: Tuple[int, int, int, int], history: Dict[str, Any], now_ts: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect a tap gesture.

    Conditions:
    - Hand moves towards validator ROI (decreasing horizontal distance)
    - Wrist y approximately at chest (midpoint between shoulders)
    - Short pause 0.3-1.0s near/in ROI
    """
    if not pose or not pose.get("keypoints"):
        return False, {"reason": "no_pose"}

    now = now_ts or time.time()
    kps = pose["keypoints"]
    
    ls = _get_point(kps, "left_shoulder")
    rs = _get_point(kps, "right_shoulder")
    le = _get_point(kps, "left_elbow")
    re = _get_point(kps, "right_elbow")
    lw = _get_point(kps, "left_wrist")
    rw = _get_point(kps, "right_wrist")

    if not ls or not rs or (not lw and not rw):
        return False, {"reason": "missing_joints"}

    chest_y = int((ls[1] + rs[1]) / 2)
    y_tol = int(max(settings.gesture_chest_tolerance_px, abs(ls[1] - rs[1]) * 0.3))

    # pick best wrist by confidence
    wrists = []
    if lw: wrists.append(("left", lw))
    if rw: wrists.append(("right", rw))
    wrists.sort(key=lambda w: w[1][2], reverse=True)
    if not wrists:
        return False, {"reason": "no_wrist"}

    hand_side, (wx, wy, wconf) = wrists[0]
    if wconf < 0.3:
        return False, {"reason": "low_conf"}

    # Distance to validator ROI horizontally
    dist_x = _horizontal_distance_to_roi(wx, validator_roi)
    in_roi = _point_in_roi(wx, wy, validator_roi)

    # Track temporal evolution
    prev = history.get("prev", None)
    history.setdefault("first_seen", now)
    history["last_seen"] = now
    history.setdefault("min_dist", dist_x)
    history["min_dist"] = min(history["min_dist"], dist_x)

    if prev is not None:
        prev_dist = prev.get("dist_x", dist_x)
        moving_towards = dist_x < prev_dist - 2  # at least 2 px closer
    else:
        moving_towards = False

    near_chest = abs(wy - chest_y) <= y_tol
    pause_window = history.get("pause_window", None)

    # Update pause window if within ROI proximity
    proximity_px = history.get("proximity_px", settings.gesture_proximity_px)
    close_enough = dist_x <= proximity_px or in_roi

    if close_enough and near_chest:
        if pause_window is None:
            pause_window = {"start": now, "accum": 0.0, "last": now}
        else:
            # accumulate dwell time
            pause_window["accum"] += now - pause_window["last"]
            pause_window["last"] = now
    else:
        # preserve window but don't accumulate, so brief gaps don't reset
        if pause_window is not None:
            pause_window["last"] = now

    history["pause_window"] = pause_window
    history["prev"] = {"dist_x": dist_x, "wx": wx, "wy": wy, "ts": now, "in_roi": in_roi}

    dwell = pause_window["accum"] if pause_window else 0.0
    dwell_min = history.get("dwell_min_s", settings.gesture_dwell_min_s)
    dwell_max = history.get("dwell_max_s", settings.gesture_dwell_max_s)

    # Decision
    tap_detected = (near_chest and close_enough and (dwell_min <= dwell <= dwell_max)) or (
        settings.gesture_relaxed_in_roi and in_roi and (dwell_min <= dwell <= dwell_max)
    )

    metrics = {
        "hand_side": hand_side,
        "dist_x": dist_x,
        "near_chest": near_chest,
        "in_roi": in_roi,
        "moving_towards": moving_towards,
        "dwell": dwell,
        "dwell_min": dwell_min,
        "dwell_max": dwell_max,
    }
    return tap_detected, metrics


