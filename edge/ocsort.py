"""
Minimal OC-SORT/SORT-style multi-object tracker.

This implementation provides stable IDs per object using a constant-velocity
Kalman filter and Hungarian assignment with IoU cost. It is designed for
runtime use and does not include dataset-specific evaluation utilities.
"""
from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] box to z = [x, y, s, r].
    x,y is center; s is scale (area); r is aspect ratio w/h.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / h if h != 0 else 0.0
    return np.array([x, y, s, r], dtype=float).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float = None) -> np.ndarray:
    """Convert filter state vector x to [x1,y1,x2,y2] box."""
    x_center, y_center, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(max(s * r, 0.0))
    h = s / w if w != 0 else 0.0
    x1 = x_center - w / 2.0
    y1 = y_center - h / 2.0
    x2 = x_center + w / 2.0
    y2 = y_center + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area1 = max(0.0, (bb_test[2] - bb_test[0])) * max(0.0, (bb_test[3] - bb_test[1]))
    area2 = max(0.0, (bb_gt[2] - bb_gt[0])) * max(0.0, (bb_gt[3] - bb_gt[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_cost_matrix(detections: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    """Return cost matrix (1 - IoU) for Hungarian assignment."""
    if detections.size == 0 or trackers.size == 0:
        return np.zeros((detections.shape[0], trackers.shape[0]), dtype=float)
    cost = np.zeros((detections.shape[0], trackers.shape[0]), dtype=float)
    for d in range(detections.shape[0]):
        for t in range(trackers.shape[0]):
            cost[d, t] = 1.0 - iou(detections[d], trackers[t])
    return cost


class KalmanBoxTracker:
    """Track a single object with a Kalman filter."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        # Create a Kalman filter with 7D state: [x, y, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Initial covariance
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty for velocities
        self.kf.P *= 10.0

        # Process and measurement noise
        self.kf.R[2:, 2:] *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker._next_id()
        self.history: List[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def _next_id() -> int:
        KalmanBoxTracker.count += 1
        return KalmanBoxTracker.count

    def update(self, bbox: np.ndarray) -> None:
        """Kalman correction with observed bbox."""
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Kalman prediction step; returns predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            # prevent s from becoming negative
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


class OCSortTracker:
    """Multiple object tracker managing KalmanBoxTracker instances."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.2):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections_xyxy: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Update tracker with current frame detections in [x1,y1,x2,y2].
        Returns a list of (bbox, track_id) for active tracks.
        """
        self.frame_count += 1

        # Predict all tracker positions
        predicted_boxes = []
        for tracker in self.trackers:
            pred = tracker.predict()
            predicted_boxes.append(pred.reshape(-1))
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # Assign detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections_xyxy, predicted_boxes
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections_xyxy[det_idx])

        # Create and initialize new trackers for unmatched detections
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections_xyxy[det_idx]))

        # Remove dead trackers
        alive_trackers: List[KalmanBoxTracker] = []
        for trk_idx, tracker in enumerate(self.trackers):
            if tracker.time_since_update < self.max_age:
                alive_trackers.append(tracker)
        self.trackers = alive_trackers

        # Prepare outputs for confirmed/active tracks
        outputs: List[Tuple[np.ndarray, int]] = []
        for tracker in self.trackers:
            # Only report tracks with sufficient hits or in the first few frames
            if tracker.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_state().reshape(-1)
                outputs.append((bbox, tracker.id))
        return outputs

    def _associate_detections_to_trackers(
        self, detections: np.ndarray, trackers: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Assign detections to trackers using Hungarian algorithm with IoU gating."""
        if trackers.size == 0:
            return [], list(range(len(detections))), []

        cost_matrix = iou_cost_matrix(detections, trackers)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = []
        unmatched_trks = []

        for d, t in zip(row_ind, col_ind):
            if 1.0 - cost_matrix[d, t] >= self.iou_threshold:
                matched.append((int(d), int(t)))
            else:
                unmatched_dets.append(int(d))
                unmatched_trks.append(int(t))

        # Detections and trackers not in any assignment
        for d in range(detections.shape[0]):
            if d not in row_ind:
                unmatched_dets.append(d)
        for t in range(trackers.shape[0]):
            if t not in col_ind:
                unmatched_trks.append(t)

        return matched, unmatched_dets, unmatched_trks


