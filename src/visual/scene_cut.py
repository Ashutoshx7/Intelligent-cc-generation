"""Detect scene cuts via frame histogram comparison."""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SceneCutDetector:
    """
    Detect scene cuts by comparing color histograms of consecutive frames.
    Uses Bhattacharyya distance — values > threshold indicate a cut.

    Why this matters for CC: A scene cut at the same timestamp as an audio
    event causes MediaPipe to see a completely different person/scene,
    which looks like a "reaction" but is actually just an edit.
    Events on scene cuts should skip visual analysis entirely.
    """

    def __init__(self, threshold: float = 0.4):
        """
        Args:
            threshold: Bhattacharyya distance threshold (0-1).
                       0.3-0.5 works well for most content.
        """
        self.threshold = threshold

    def detect_cuts(self, video_path: str) -> list:
        """
        Scan entire video and return timestamps of scene cuts.

        Samples every 3rd frame for speed while still catching all cuts.

        Args:
            video_path: Path to video file.

        Returns:
            List of cut timestamps in seconds.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            logger.warning("Could not read FPS, defaulting to 24")
            fps = 24.0

        cuts = []
        prev_hist = None
        frame_idx = 0
        sample_interval = 3  # check every 3rd frame for speed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # HSV histogram is more robust to lighting changes than grayscale
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist, hist)

                if prev_hist is not None:
                    dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if dist > self.threshold:
                        cut_time = frame_idx / fps
                        cuts.append(cut_time)
                        logger.debug(f"Scene cut at {cut_time:.2f}s (dist={dist:.3f})")

                prev_hist = hist

            frame_idx += 1

        cap.release()
        total_duration = frame_idx / fps if fps > 0 else 0
        logger.info(f"Detected {len(cuts)} scene cuts in {total_duration:.1f}s video")
        return cuts

    def is_on_scene_cut(self, timestamp: float, cuts: list, tolerance: float = 0.5) -> bool:
        """
        Check if a timestamp falls within 'tolerance' seconds of any scene cut.

        Args:
            timestamp: Time in seconds to check.
            cuts: List of cut timestamps from detect_cuts().
            tolerance: Window around each cut (seconds).

        Returns:
            True if the timestamp is near a scene cut.
        """
        for cut_time in cuts:
            if abs(timestamp - cut_time) <= tolerance:
                return True
        return False
