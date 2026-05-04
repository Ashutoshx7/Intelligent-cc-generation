"""Multi-person pose analysis for flinch and head-turn detection using MediaPipe Tasks API."""
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Default model path
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "pose_landmarker_lite.task")


class PoseAnalyzer:
    """
    Detect physical reactions using MediaPipe PoseLandmarker (33 landmarks per person).

    Reactions detected:
    1. Flinch/Startle — sudden shoulder asymmetry (one shoulder raised)
    2. Head Turn — nose position deviates from center between ears

    Key landmark indices (PoseLandmark enum):
        0  = NOSE
        7  = LEFT_EAR
        8  = RIGHT_EAR
        11 = LEFT_SHOULDER
        12 = RIGHT_SHOULDER

    Uses MediaPipe Tasks API (v0.10+).
    """

    def __init__(self, config: dict, model_path: str = None):
        if model_path is None:
            model_path = _MODEL_PATH

        if not os.path.exists(model_path):
            logger.warning(f"Pose model not found at {model_path}. Pose analysis will be disabled.")
            self.detector = None
        else:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=config['visual'].get('max_num_poses', 4),
                min_pose_detection_confidence=config['visual']['min_detection_confidence'],
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            logger.info("PoseLandmarker initialized (Tasks API)")

        self.flinch_threshold = config['visual']['flinch_threshold']      # 0.05
        self.flinch_ceiling = config['visual']['flinch_ceiling']          # 0.15
        self.head_turn_threshold = config['visual']['head_turn_threshold']  # 0.15
        self.head_turn_ceiling = config['visual']['head_turn_ceiling']      # 0.35

    def analyze(self, frame: np.ndarray) -> dict:
        """
        Analyze a single frame for pose-based reactions.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            dict with pose_score, detected, num_persons, flinch_score, head_turn_score
        """
        if self.detector is None:
            return {"pose_score": 0.0, "detected": False, "num_persons": 0,
                    "flinch_score": 0.0, "head_turn_score": 0.0}

        # Convert BGR to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return {"pose_score": 0.0, "detected": False, "num_persons": 0,
                    "flinch_score": 0.0, "head_turn_score": 0.0}

        # Score each detected person, take the maximum reaction
        best_flinch = 0.0
        best_head_turn = 0.0

        for person_landmarks in results.pose_landmarks:
            flinch = self._score_flinch(person_landmarks)
            head_turn = self._score_head_turn(person_landmarks)
            best_flinch = max(best_flinch, flinch)
            best_head_turn = max(best_head_turn, head_turn)

        pose_score = max(best_flinch, best_head_turn)
        num_persons = len(results.pose_landmarks)

        return {
            "pose_score": pose_score,
            "detected": True,
            "num_persons": num_persons,
            "flinch_score": best_flinch,
            "head_turn_score": best_head_turn,
        }

    def _score_flinch(self, landmarks) -> float:
        """
        Score flinch/startle from shoulder asymmetry.

        Normal standing: shoulders at roughly same Y -> diff ~ 0.01-0.03
        Flinch/startle: one shoulder raised -> diff ~ 0.05-0.15+
        """
        lm = landmarks
        shoulder_diff = abs(lm[11].y - lm[12].y)

        if shoulder_diff < self.flinch_threshold:
            return 0.0

        score = (shoulder_diff - self.flinch_threshold) / (self.flinch_ceiling - self.flinch_threshold)
        return min(score, 1.0)

    def _score_head_turn(self, landmarks) -> float:
        """
        Score head turn from nose position relative to ears.

        Forward-facing: nose.x ~ midpoint(ears), ratio ~ 0.5
        Turned: ratio deviates from 0.5
        """
        lm = landmarks
        ear_span = abs(lm[7].x - lm[8].x)

        if ear_span < 0.01:
            return 0.0

        nose_ratio = (lm[0].x - min(lm[7].x, lm[8].x)) / ear_span
        deviation = abs(nose_ratio - 0.5)

        if deviation < self.head_turn_threshold:
            return 0.0

        score = (deviation - self.head_turn_threshold) / (self.head_turn_ceiling - self.head_turn_threshold)
        return min(score, 1.0)

    def close(self):
        """Release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()
