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
import time

from loguru import logger

from pose import PoseEstimator
from gesture import detect_tap
import numpy as np
import cv2
import math


class ValidatorAndGateRules:
    def __init__(self, config):
        self.cfg = config
        self._pose = PoseEstimator()
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def _get_track(self, track_id: int) -> Dict[str, Any]:
        st = self._tracks.get(track_id)
        if st is None:
            st = {
                "tap_confirmed": False,
                "inside_validator_roi": False,
                "gesture_history": {},
                "last_pose_ts": 0.0,
                "last_metrics": {},
                "last_pose": None,
            }
            self._tracks[track_id] = st
        return st

    def _point_in_roi(self, bbox, roi) -> bool:
        if roi is None:
            return False
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        # check bbox center inside ROI
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _overlaps_roi(self, bbox, roi) -> bool:
        if roi is None:
            return False
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        ix1 = max(x1, rx1)
        iy1 = max(y1, ry1)
        ix2 = min(x2, rx2)
        iy2 = min(y2, ry2)
        return ix2 > ix1 and iy2 > iy1

    def _horizontal_distance_bbox_to_roi(self, bbox, roi) -> int:
        if roi is None:
            return 1_000_000
        x1, _, x2, _ = bbox
        rx1, _, rx2, _ = roi
        if x2 < rx1:
            return rx1 - x2
        if x1 > rx2:
            return x1 - rx2
        return 0

    def update_track(self, frame, detection: Dict[str, Any], validator_roi: Tuple[int, int, int, int]) -> None:
        now = time.time()
        track_id = int(detection.get("person_id", -1))
        if track_id < 0:
            return
        state = self._get_track(track_id)

        # ROI occupancy
        inside = self._point_in_roi(detection["bbox"], validator_roi)
        overlaps = self._overlaps_roi(detection["bbox"], validator_roi)
        hdist = self._horizontal_distance_bbox_to_roi(detection["bbox"], validator_roi)
        near_or_inside = inside or overlaps or hdist <= self.cfg.gesture_bbox_roi_hdist_px
        state["inside_validator_roi"] = inside

        # If near/inside ROI OR full-frame pose requested, run pose (if available)
        should_pose = near_or_inside or bool(getattr(self.cfg, "pose_full_frame", False))
        if should_pose and (now - state["last_pose_ts"]) >= self.cfg.movenet_min_interval_s and self._pose is not None:
            pose = self._pose.infer_pose(frame, detection["bbox"])  # may be None
            state["last_pose_ts"] = now
            if pose is not None:
                tap, metrics = detect_tap(pose, validator_roi, state["gesture_history"], now_ts=now)
                state["last_metrics"] = metrics
                state["last_pose"] = pose
                if tap:
                    state["tap_confirmed"] = True
                    logger.info(f"Track {track_id}: tap confirmed")
            else:
                if getattr(self.cfg, "gesture_debug_logs", False):
                    logger.debug(f"Track {track_id}: pose unavailable or below threshold (near={near_or_inside}, hdist={hdist})")

    def on_crossing(self, track_id: int) -> Dict[str, Any]:
        state = self._get_track(track_id)
        if state.get("tap_confirmed"):
            result = {"classification": "pass", "tap_confirmed": True}
        else:
            result = {"classification": "evasion", "tap_confirmed": False}
        # Reset per-crossing
        # Keep a short memory so immediate subsequent crossings (tailgate) don't reuse tap
        state["tap_confirmed"] = False
        state["gesture_history"] = {}
        state["inside_validator_roi"] = False
        return result

    def get_status(self, track_id: int) -> Dict[str, Any]:
        return self._get_track(track_id).copy()

    def preload(self) -> None:
        """Initialize and warm up MoveNet so first inference does not stall."""
        try:
            logger.info("Preloading MoveNet pose model...")
            ok = self._pose.initialize()
            if not ok:
                logger.warning("MoveNet preload failed (model not initialized)")
                return
            # Warm-up run with a tiny dummy frame and bbox
            size = max(16, int(getattr(self.cfg, "movenet_input_size", 192)))
            dummy = np.zeros((size, size, 3), dtype=np.uint8)
            bbox = [1, 1, min(8, size - 2), min(8, size - 2)]
            try:
                _ = self._pose.infer_pose(dummy, bbox)
            except Exception:
                # Ignore warm-up inference errors
                pass
            logger.info("MoveNet preload complete")
        except Exception as exc:
            logger.warning(f"MoveNet preload encountered an error: {exc}")



    def _kp_map(self, pose: Dict[str, Any]) -> Dict[str, Tuple[int, int, float]]:
        mp: Dict[str, Tuple[int, int, float]] = {}
        for kp in pose.get("keypoints", []) or []:
            try:
                name = str(kp.get("name"))
                x = int(kp.get("x", 0))
                y = int(kp.get("y", 0))
                c = float(kp.get("confidence", 0.0))
                mp[name] = (x, y, c)
            except Exception:
                continue
        return mp

    def _angle_deg(self, a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> Optional[float]:
        try:
            ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
            bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
            nba = np.linalg.norm(ba)
            nbc = np.linalg.norm(bc)
            if nba < 1e-6 or nbc < 1e-6:
                return None
            cosang = float(np.dot(ba, bc) / (nba * nbc))
            cosang = max(-1.0, min(1.0, cosang))
            return float(math.degrees(math.acos(cosang)))
        except Exception:
            return None

    def draw_pose_overlays(self, frame, detections: Any) -> Any:
        """Draw keypoints, skeleton, and joint angles for tracks with a recent pose.

        Returns the same frame (modified in place) for convenience.
        """
        try:
            if not getattr(self.cfg, "pose_overlay_enabled", False):
                return frame
            min_conf = float(getattr(self.cfg, "pose_overlay_min_confidence", 0.3))
            # Optionally run pose for overlay even if not near ROI
            infer_if_missing = bool(getattr(self.cfg, "pose_overlay_infer_if_missing", True))

            # Common skeleton pairs for MoveNet 17
            pairs = [
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),
            ]

            for det in detections or []:
                track_id = int(det.get("person_id", -1)) if isinstance(det, dict) else -1
                if track_id < 0:
                    continue
                st = self._tracks.get(track_id)
                if not st:
                    st = self._get_track(track_id)
                pose = st.get("last_pose")
                if (pose is None) and infer_if_missing and self._pose is not None:
                    # Run a quick on-demand pose inference for overlay
                    try:
                        pose_try = self._pose.infer_pose(frame, det.get("bbox"))
                        if pose_try is not None:
                            st["last_pose"] = pose_try
                            pose = pose_try
                    except Exception:
                        pass
                if not pose or not pose.get("keypoints"):
                    continue
                mp = self._kp_map(pose)

                # Draw skeleton lines
                for a, b in pairs:
                    pa = mp.get(a)
                    pb = mp.get(b)
                    if pa and pb and pa[2] >= min_conf and pb[2] >= min_conf:
                        cv2.line(frame, (pa[0], pa[1]), (pb[0], pb[1]), (0, 255, 255), 2)

                # Draw keypoints
                for name, (x, y, conf) in mp.items():
                    if conf >= min_conf:
                        cv2.circle(frame, (x, y), 3, (0, 165, 255), -1)

                # Draw joint angles if enabled
                if getattr(self.cfg, "pose_overlay_draw_angles", True):
                    # Elbows
                    le, ls, lw = mp.get("left_elbow"), mp.get("left_shoulder"), mp.get("left_wrist")
                    re, rs, rw = mp.get("right_elbow"), mp.get("right_shoulder"), mp.get("right_wrist")
                    # Knees
                    lk, lh, la = mp.get("left_knee"), mp.get("left_hip"), mp.get("left_ankle")
                    rk, rh, ra = mp.get("right_knee"), mp.get("right_hip"), mp.get("right_ankle")

                    if le and ls and lw and min(le[2], ls[2], lw[2]) >= min_conf:
                        ang = self._angle_deg((ls[0], ls[1]), (le[0], le[1]), (lw[0], lw[1]))
                        if ang is not None:
                            cv2.putText(frame, f"{int(ang)}\u00B0", (le[0] + 6, le[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)
                    if re and rs and rw and min(re[2], rs[2], rw[2]) >= min_conf:
                        ang = self._angle_deg((rs[0], rs[1]), (re[0], re[1]), (rw[0], rw[1]))
                        if ang is not None:
                            cv2.putText(frame, f"{int(ang)}\u00B0", (re[0] + 6, re[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)
                    if lk and lh and la and min(lk[2], lh[2], la[2]) >= min_conf:
                        ang = self._angle_deg((lh[0], lh[1]), (lk[0], lk[1]), (la[0], la[1]))
                        if ang is not None:
                            cv2.putText(frame, f"{int(ang)}\u00B0", (lk[0] + 6, lk[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)
                    if rk and rh and ra and min(rk[2], rh[2], ra[2]) >= min_conf:
                        ang = self._angle_deg((rh[0], rh[1]), (rk[0], rk[1]), (ra[0], ra[1]))
                        if ang is not None:
                            cv2.putText(frame, f"{int(ang)}\u00B0", (rk[0] + 6, rk[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 50), 2)

        except Exception:
            # Non-fatal drawing errors should not crash the pipeline
            pass
        return frame

