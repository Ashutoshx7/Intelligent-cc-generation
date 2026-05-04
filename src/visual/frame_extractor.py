"""Extract video frames at specific timestamps using temporal reaction windows."""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extract frames from video at precise timestamps.

    Key innovation: instead of extracting at the midpoint of the audio event
    (when the actor hasn't reacted yet), we extract frames in the REACTION
    WINDOW — 300ms to 1500ms AFTER the sound onset.

    Human reaction times:
    - Startle reflex: ~50ms (too fast for camera)
    - Head turn / flinch: 200-400ms
    - Conscious reaction: 500-1500ms
    - Actor dramatic timing: 300-2000ms
    """

    def __init__(self, config: dict):
        self.reaction_start = config['visual']['reaction_window_start']  # 0.3s
        self.reaction_end = config['visual']['reaction_window_end']      # 1.5s
        self.num_frames = config['visual']['num_reaction_frames']        # 5

    def extract_reaction_frames(self, video_path: str, event_start: float) -> list:
        """
        Extract frames in the biological reaction window AFTER an audio event.

        Samples num_frames evenly across [event_start + 0.3s, event_start + 1.5s].

        Args:
            video_path: Path to video file.
            event_start: Start time of the audio event in seconds.

        Returns:
            List of (timestamp, frame) tuples. Frame is BGR numpy array.
            May return fewer than num_frames if video is too short.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            cap.release()
            logger.warning("Could not read FPS")
            return []

        video_duration = total_frames / fps

        # Calculate reaction window boundaries
        window_start = event_start + self.reaction_start
        window_end = min(event_start + self.reaction_end, video_duration - 0.05)

        if window_start >= video_duration:
            logger.debug(f"Reaction window beyond video end for event at {event_start:.2f}s")
            cap.release()
            return []

        # Evenly space frames across the window
        if window_end <= window_start:
            frame_times = [window_start]
        else:
            frame_times = np.linspace(window_start, window_end, num=self.num_frames).tolist()

        frames = []
        for t in frame_times:
            frame_num = int(t * fps)
            if frame_num >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                frames.append((round(t, 3), frame))

        cap.release()
        logger.debug(f"Extracted {len(frames)} reaction frames for event at {event_start:.2f}s "
                     f"(window: {window_start:.2f}s - {window_end:.2f}s)")
        return frames

    def extract_single_frame(self, video_path: str, timestamp: float):
        """Extract a single frame at exact timestamp. Fallback method."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            return None
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
