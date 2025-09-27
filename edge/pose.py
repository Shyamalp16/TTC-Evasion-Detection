"""
Pose estimation using MoveNet Lite (TFLite).

- Tries to import tflite_runtime first, then falls back to tensorflow.lite
- Provides PoseEstimator with infer_pose(frame, bbox) -> keypoints dict

Keypoints schema (MoveNet Lightning, 17 points):
{
    "keypoints": [
        {"name": str, "x": int, "y": int, "confidence": float},
        ... 17 entries ...
    ],
    "score": float
}
Coordinates are in full-frame pixel space.
"""
from typing import Dict, List, Tuple, Optional
import math
import time

import numpy as np
import cv2
from loguru import logger

from config import settings
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # optional backend
import os


def _resolve_model_path(path_in_settings: str) -> str:
    """Resolve model path; try absolute, then relative to edge/models."""
    if os.path.isabs(path_in_settings) and os.path.exists(path_in_settings):
        return path_in_settings
    # Try relative to project root
    if os.path.exists(path_in_settings):
        return path_in_settings
    # Try edge/models/<filename>
    candidate = os.path.join(os.path.dirname(__file__), "models", os.path.basename(path_in_settings))
    return candidate


def _load_tflite_interpreter(model_path: str):
    """Load a TFLite interpreter from tflite_runtime or tensorflow.lite."""
    interpreter = None
    try:
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=model_path, num_threads=max(1, settings.movenet_num_threads))
        logger.info("Loaded tflite_runtime.Interpreter for MoveNet")
        return interpreter
    except Exception as e:
        logger.debug(f"tflite_runtime not available or failed to load: {e}")

    # Try ai_edge_litert (newer TF LiteRT)
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        interpreter = Interpreter(model_path=model_path, num_threads=max(1, settings.movenet_num_threads))
        logger.info("Loaded ai_edge_litert.Interpreter for MoveNet")
        return interpreter
    except Exception as e:
        logger.debug(f"ai_edge_litert not available or failed to load: {e}")

    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        interpreter = Interpreter(model_path=model_path, num_threads=max(1, settings.movenet_num_threads))
        logger.info("Loaded tensorflow.lite Interpreter for MoveNet")
        return interpreter
    except Exception as e:
        logger.error(f"Failed to load TFLite interpreter: {e}")
        return None


MOVENET_KEYPOINT_NAMES_17 = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class PoseEstimator:
    """Pose estimator supporting MediaPipe and MoveNet backends (configurable)."""

    def __init__(self):
        self.backend: str = getattr(settings, "pose_backend", "mediapipe")
        # MoveNet
        self.model_path: str = settings.movenet_model_path
        self.input_size: int = int(settings.movenet_input_size)
        self.score_threshold: float = float(settings.pose_score_threshold)
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._use_tfhub = bool(getattr(settings, "movenet_use_tfhub", False))
        self._tfhub_model = None
        self._tf = None
        # MediaPipe
        self._mp_pose = None
        self._mp_instance = None
        self._mp_connections = None
        self._initialized = False

    def initialize(self) -> bool:
        try:
            if self.backend == "mediapipe" and mp is not None:
                # Initialize MediaPipe Pose once
                self._mp_pose = mp.solutions.pose
                self._mp_connections = self._mp_pose.POSE_CONNECTIONS
                self._mp_instance = self._mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=int(getattr(settings, "mediapipe_model_complexity", 1)),
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=float(getattr(settings, "mediapipe_min_detection_confidence", 0.5)),
                    min_tracking_confidence=float(getattr(settings, "mediapipe_min_tracking_confidence", 0.5)),
                )
                self._initialized = True
                logger.info("MediaPipe Pose initialized")
                return True
            if self._use_tfhub:
                # Lazy import TF/Hub to avoid overhead if not used
                # Reduce TensorFlow/TF Hub warning noise
                try:
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                except Exception:
                    pass
                import tensorflow as tf  # type: ignore
                try:
                    tf.get_logger().setLevel('ERROR')
                    # Also try to reduce absl logging if present
                    try:
                        from absl import logging as absl_logging  # type: ignore
                        absl_logging.set_verbosity(absl_logging.ERROR)
                    except Exception:
                        pass
                except Exception:
                    pass
                import tensorflow_hub as hub  # type: ignore
                url = getattr(settings, "movenet_tfhub_url", "https://tfhub.dev/google/movenet/singlepose/lightning/4")
                self._tfhub_model = hub.load(url)
                # Signature: serving_default, input expects [1,192,192,3] float32 0..1
                self._interpreter = None
                self._tf = tf
                self._initialized = True
                logger.info(f"MoveNet TF Hub model loaded: {url}")
                return True
            else:
                resolved = _resolve_model_path(self.model_path)
                if not os.path.exists(resolved):
                    logger.error(f"MoveNet model not found at: {resolved}")
                    return False
                self._interpreter = _load_tflite_interpreter(resolved)
                if self._interpreter is None:
                    return False
                self._interpreter.allocate_tensors()
                self._input_details = self._interpreter.get_input_details()
                self._output_details = self._interpreter.get_output_details()
                self._initialized = True
                logger.info(f"MoveNet TFLite initialized: {resolved}")
                return True
        except Exception as e:
            logger.error(f"PoseEstimator initialization failed: {e}")
            return False

    def _preprocess(self, frame: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop person bbox with small margin, resize to input, return tensor and crop bbox used."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Add margin to include hands that may extend beyond bbox
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.2)
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(w, x2 + margin_x)
        cy2 = min(h, y2 + margin_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # Keep aspect ratio by padding to square before resize
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        pad_top = (side - ch) // 2
        pad_bottom = side - ch - pad_top
        pad_left = (side - cw) // 2
        pad_right = side - cw - pad_left
        crop_sq = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        input_img = cv2.resize(crop_sq, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0,1]
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        return input_img, (cx1 - pad_left, cy1 - pad_top, cx1 - pad_left + side, cy1 - pad_top + side)

    def _postprocess(self, raw_output: np.ndarray, crop_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Dict[str, any]:
        """Map normalized keypoints back to frame coordinates."""
        h, w = frame_shape[:2]
        cx1, cy1, cx2, cy2 = crop_bbox
        side = max(1, cx2 - cx1)

        # Expected output shape (TFLite): (1,1,17,3) -> y, x, score
        # TF Hub returns a dict with 'output_0' tensor shape [1,1,17,3]
        if isinstance(raw_output, dict):
            arr = None
            for v in raw_output.values():
                arr = v
                break
            raw = np.array(arr)
        else:
            raw = np.array(raw_output)
        if raw.ndim == 4:
            kp = raw[0, 0]
        elif raw.ndim == 3:
            kp = raw[0]
        else:
            kp = raw

        keypoints = []
        total_score = 0.0
        for idx, (y_norm, x_norm, score) in enumerate(kp):
            x = int(cx1 + float(x_norm) * side)
            y = int(cy1 + float(y_norm) * side)
            score_f = float(score)
            total_score += max(0.0, score_f)
            name = MOVENET_KEYPOINT_NAMES_17[idx] if idx < len(MOVENET_KEYPOINT_NAMES_17) else f"kp_{idx}"
            keypoints.append({"name": name, "x": int(np.clip(x, 0, w - 1)), "y": int(np.clip(y, 0, h - 1)), "confidence": score_f})

        avg_score = total_score / max(1, len(keypoints))
        return {"keypoints": keypoints, "score": avg_score}

    def infer_pose(self, frame: np.ndarray, bbox: List[int]) -> Optional[Dict[str, any]]:
        """Run pose and return keypoints dict. If backend is MediaPipe, bbox is optional."""
        if not self._initialized:
            # Throttle init attempts: only every 2 seconds
            last_try = getattr(self, "_last_init_try", 0.0)
            now = time.time()
            if now - last_try < 2.0:
                return None
            self._last_init_try = now
            ok = self.initialize()
            if not ok:
                return None

        try:
            if self.backend == "mediapipe" and self._mp_instance is not None:
                # Run on full frame for robustness
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._mp_instance.process(rgb)
                if not res.pose_landmarks:
                    return None
                h, w = frame.shape[:2]
                keypoints = []
                total = 0.0
                lm = res.pose_landmarks.landmark
                # MediaPipe has 33 landmarks; select a subset mapping to the 17-keypoint names
                # We'll use approximate indices: shoulders(11,12), elbows(13,14), wrists(15,16), hips(23,24), knees(25,26), ankles(27,28), plus nose(0)
                idx_map = {
                    "nose": 0,
                    "left_shoulder": 11,
                    "right_shoulder": 12,
                    "left_elbow": 13,
                    "right_elbow": 14,
                    "left_wrist": 15,
                    "right_wrist": 16,
                    "left_hip": 23,
                    "right_hip": 24,
                    "left_knee": 25,
                    "right_knee": 26,
                    "left_ankle": 27,
                    "right_ankle": 28,
                }
                for name, i in idx_map.items():
                    p = lm[i]
                    x = int(max(0, min(w - 1, p.x * w)))
                    y = int(max(0, min(h - 1, p.y * h)))
                    c = float(p.visibility)
                    total += max(0.0, c)
                    keypoints.append({"name": name, "x": x, "y": y, "confidence": c})
                score = total / max(1, len(keypoints))
                return {"keypoints": keypoints, "score": score}

            inp, crop_bbox = self._preprocess(frame, bbox)
            if self._tfhub_model is not None:
                # TF Hub model: call signature
                output = self._tfhub_model.signatures['serving_default'](self._tf.constant(inp))  # type: ignore
                result = self._postprocess(output, crop_bbox, frame.shape)
            else:
                # TFLite: set tensor and invoke
                self._interpreter.set_tensor(self._input_details[0]["index"], inp)
                self._interpreter.invoke()
                output = self._interpreter.get_tensor(self._output_details[0]["index"])  # type: ignore
                result = self._postprocess(output, crop_bbox, frame.shape)
            if result["score"] < self.score_threshold:
                return None
            return result
        except Exception as e:
            logger.debug(f"Pose inference failed: {e}")
            return None


