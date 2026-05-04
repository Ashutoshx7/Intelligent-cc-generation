"""Multi-face analysis for surprise/gasp detection using MediaPipe Tasks API."""
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Default model path
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "face_landmarker.task")


class FaceAnalyzer:
    """
    Detect facial reactions using MediaPipe FaceLandmarker (478 landmarks per face).

    Primary signal: mouth openness (surprise/gasp).
        Landmarks 13 = upper inner lip center
        Landmarks 14 = lower inner lip center
        Closed mouth: gap ~ 0.01-0.02 normalized
        Open mouth (gasp): gap ~ 0.05-0.10+

    Supports multi-face detection — takes the MAX reaction score across
    all detected faces, because in a group scene it only matters that
    at least one person reacted.

    Uses MediaPipe Tasks API (v0.10+).
    """

    def __init__(self, config: dict, model_path: str = None):
        if model_path is None:
            model_path = _MODEL_PATH

        if not os.path.exists(model_path):
            logger.warning(f"Face model not found at {model_path}. Face analysis will be disabled.")
            self.detector = None
        else:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=config['visual'].get('max_num_faces', 4),
                min_face_detection_confidence=config['visual']['min_detection_confidence'],
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker initialized (Tasks API)")

        self.mouth_threshold = config['visual']['mouth_open_threshold']  # 0.02
        self.mouth_ceiling = config['visual']['mouth_open_ceiling']      # 0.08

    def analyze(self, frame: np.ndarray) -> dict:
        """
        Analyze frame for facial reactions across all detected faces.

        Args:
            frame: BGR image (numpy array from OpenCV).

        Returns:
            dict with face_score (max across faces), detected, num_faces, mouth_scores
        """
        if self.detector is None:
            return {"face_score": 0.0, "detected": False, "num_faces": 0, "mouth_scores": []}

        # Convert BGR to RGB and create MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return {"face_score": 0.0, "detected": False, "num_faces": 0, "mouth_scores": []}

        mouth_scores = []
        for face_landmarks in results.face_landmarks:
            score = self._score_mouth_open(face_landmarks)
            mouth_scores.append(score)

        max_score = max(mouth_scores) if mouth_scores else 0.0

        return {
            "face_score": max_score,
            "detected": True,
            "num_faces": len(results.face_landmarks),
            "mouth_scores": mouth_scores,
        }

    def _score_mouth_open(self, landmarks) -> float:
        """
        Score mouth openness from lip landmarks.

        Landmark 13 = upper inner lip center
        Landmark 14 = lower inner lip center
        The Y-distance indicates how open the mouth is (normalized 0-1 coords).
        """
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        mouth_gap = abs(upper_lip.y - lower_lip.y)

        if mouth_gap < self.mouth_threshold:
            return 0.0

        score = (mouth_gap - self.mouth_threshold) / (self.mouth_ceiling - self.mouth_threshold)
        return min(score, 1.0)

    def close(self):
        """Release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()
