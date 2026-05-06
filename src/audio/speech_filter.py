"""Filter speech segments using energy-based VAD (pure Python fallback) + YAMNet class indices."""
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import webrtcvad; if not available, use energy-based fallback
try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False
    logger.info("webrtcvad not available — using energy-based VAD fallback")


class SpeechFilter:
    """
    Two-layer speech filtering:
    1. VAD: frame-level voice activity detection (WebRTC if available, else energy-based)
    2. YAMNet class filter: remove events classified as speech classes (indices 0-6)

    The VAD is particularly important for Hindi/regional content where dialogue
    is dense and continuous — without it, many non-speech events would be
    detected during speech segments.
    """

    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000):
        """
        Args:
            aggressiveness: 0-3. Higher = more aggressive at filtering speech.
                           Use 3 for dense Hindi dialogue.
            sample_rate: Audio sample rate.
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30  # 30ms frames
        self.aggressiveness = aggressiveness

        if HAS_WEBRTCVAD:
            assert sample_rate in (8000, 16000, 32000, 48000), \
                f"WebRTC VAD requires sample rate in (8000, 16000, 32000, 48000), got {sample_rate}"
            self.vad = webrtcvad.Vad(aggressiveness)
            logger.info(f"Using WebRTC VAD (aggressiveness={aggressiveness})")
        else:
            self.vad = None
            # Energy threshold for speech — tuned per aggressiveness
            # These thresholds are set to distinguish speech energy from
            # non-speech sounds. Speech has sustained medium energy; sounds
            # like sirens/alarms have different spectral characteristics but
            # similar RMS — so we keep thresholds moderate.
            self._energy_thresholds = {0: 0.04, 1: 0.03, 2: 0.02, 3: 0.015}
            self._energy_threshold = self._energy_thresholds.get(aggressiveness, 0.02)
            logger.info(f"Using energy-based VAD (threshold={self._energy_threshold})")

    def get_speech_segments(self, waveform: np.ndarray) -> list:
        """
        Run VAD on entire waveform to find speech regions.

        Args:
            waveform: float32 array in [-1.0, 1.0]

        Returns:
            List of (start_time, end_time) tuples marking speech regions in seconds.
        """
        if HAS_WEBRTCVAD and self.vad is not None:
            return self._get_speech_segments_webrtc(waveform)
        else:
            return self._get_speech_segments_energy(waveform)

    def _get_speech_segments_webrtc(self, waveform: np.ndarray) -> list:
        """WebRTC VAD-based speech detection."""
        int16_audio = (waveform * 32767).astype(np.int16)
        raw_bytes = int16_audio.tobytes()

        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        frame_bytes = frame_size * 2

        num_frames = len(raw_bytes) // frame_bytes
        speech_frames = []

        for i in range(num_frames):
            start_byte = i * frame_bytes
            frame = raw_bytes[start_byte:start_byte + frame_bytes]

            if len(frame) < frame_bytes:
                break

            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                is_speech = False

            start_time = i * self.frame_duration_ms / 1000.0
            speech_frames.append((start_time, is_speech))

        return self._merge_speech_frames(speech_frames)

    def _get_speech_segments_energy(self, waveform: np.ndarray) -> list:
        """
        Energy-based VAD fallback (pure Python, no C dependencies).

        Computes RMS energy per frame. Frames above the threshold
        are considered speech. Works reasonably well for separating
        speech from silence/ambient, though less accurate than WebRTC.
        """
        frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        num_frames = len(waveform) // frame_size
        speech_frames = []

        for i in range(num_frames):
            start_idx = i * frame_size
            frame = waveform[start_idx:start_idx + frame_size]

            # RMS energy
            rms = np.sqrt(np.mean(frame ** 2))
            is_speech = rms > self._energy_threshold

            start_time = i * self.frame_duration_ms / 1000.0
            speech_frames.append((start_time, is_speech))

        return self._merge_speech_frames(speech_frames)

    def _merge_speech_frames(self, speech_frames: list) -> list:
        """Merge consecutive speech frames into segments."""
        segments = []
        in_speech = False
        seg_start = 0.0

        for time_val, is_speech in speech_frames:
            if is_speech and not in_speech:
                seg_start = time_val
                in_speech = True
            elif not is_speech and in_speech:
                segments.append((seg_start, time_val))
                in_speech = False

        # Close final segment
        if in_speech and speech_frames:
            segments.append((seg_start, speech_frames[-1][0] + self.frame_duration_ms / 1000.0))

        logger.info(f"VAD found {len(segments)} speech segments "
                    f"({sum(e-s for s, e in segments):.1f}s total speech)")
        return segments

    def was_speech_before(self, timestamp: float, speech_segments: list, window: float = 1.0) -> bool:
        """
        Check if speech was active in the [timestamp - window, timestamp] range.

        If speech was happening just before this event and stopped,
        it indicates the speaker paused in reaction to the sound.

        Args:
            timestamp: The audio event's start time.
            speech_segments: Output of get_speech_segments().
            window: How far back to look (seconds).

        Returns:
            True if speech was active in the lookback window.
        """
        check_start = max(0, timestamp - window)
        for seg_start, seg_end in speech_segments:
            if seg_end > check_start and seg_start < timestamp:
                return True
        return False

    def is_during_speech(self, start_time: float, end_time: float,
                         speech_segments: list, overlap_ratio: float = 0.5) -> bool:
        """
        Check if an event overlaps significantly with a speech segment.

        Args:
            start_time: Event start time.
            end_time: Event end time.
            speech_segments: Output of get_speech_segments().
            overlap_ratio: Minimum overlap fraction to flag as during-speech.

        Returns:
            True if event overlaps with speech more than overlap_ratio.
        """
        event_duration = end_time - start_time
        if event_duration <= 0:
            return False

        total_overlap = 0.0
        for seg_start, seg_end in speech_segments:
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start

        return (total_overlap / event_duration) >= overlap_ratio
