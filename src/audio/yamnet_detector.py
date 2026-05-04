"""YAMNet-based non-speech sound event detection."""
import tensorflow_hub as hub
import numpy as np
import csv
import logging

logger = logging.getLogger(__name__)


class YAMNetDetector:
    """
    Detect non-speech audio events using YAMNet (521 AudioSet classes).

    Pipeline:
    1. Feed waveform through YAMNet -> (frames, 521) score matrix
    2. For each frame: get top class + confidence
    3. Filter out speech classes (indices 0-6)
    4. Filter below confidence threshold
    5. Merge consecutive same-label frames into single events
    """

    MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict from default.yaml
        """
        logger.info("Loading YAMNet model...")
        self.model = hub.load(self.MODEL_URL)
        self.class_names = self._load_class_names()
        self.confidence_threshold = config['audio']['confidence_threshold']
        self.speech_indices = set(config['audio']['speech_class_indices'])
        self.merge_gap = config['audio']['merge_gap_seconds']
        self.hop_size = 0.48  # YAMNet: 0.96s window, 0.48s hop
        logger.info(f"YAMNet loaded. {len(self.class_names)} classes, "
                    f"threshold={self.confidence_threshold}")

    def _load_class_names(self) -> list:
        """Load YAMNet class names from the model's asset file."""
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        class_names = []
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        return class_names

    def detect(self, waveform: np.ndarray) -> list:
        """
        Run YAMNet on waveform, return detected non-speech events.

        Args:
            waveform: float32 array, 16kHz mono, range [-1.0, 1.0]

        Returns:
            List of event dicts with id, label, confidence, start_time, end_time.
        """
        # Run inference
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()  # shape: (num_frames, 521)

        logger.info(f"YAMNet produced {scores_np.shape[0]} frames "
                    f"({scores_np.shape[0] * self.hop_size:.1f}s of audio)")

        # Extract per-frame top class (skip speech + low confidence)
        raw_events = []
        for i, frame_scores in enumerate(scores_np):
            top_class_idx = int(np.argmax(frame_scores))
            confidence = float(frame_scores[top_class_idx])

            # Skip speech classes (YAMNet indices 0-6)
            if top_class_idx in self.speech_indices:
                continue

            # Skip low confidence
            if confidence < self.confidence_threshold:
                continue

            raw_events.append({
                "label": self.class_names[top_class_idx],
                "class_index": top_class_idx,
                "confidence": confidence,
                "start_time": round(i * self.hop_size, 3),
                "end_time": round((i + 1) * self.hop_size, 3),
            })

        logger.info(f"Raw detections (after speech filter): {len(raw_events)}")

        # Merge consecutive same-label events
        merged = self._merge_consecutive(raw_events)
        logger.info(f"After merging: {len(merged)} events")

        # Assign sequential IDs
        for i, event in enumerate(merged, 1):
            event["id"] = i

        return merged

    def _merge_consecutive(self, events: list) -> list:
        """
        Merge consecutive frames with the same label into single events.

        Uses peak confidence (not average) — the loudest moment matters most.
        Two events merge if they have the same label and the gap between them
        is <= merge_gap_seconds (default 0.1s).
        """
        if not events:
            return []

        merged = [events[0].copy()]

        for ev in events[1:]:
            prev = merged[-1]

            same_label = ev["label"] == prev["label"]
            close_enough = ev["start_time"] <= prev["end_time"] + self.merge_gap

            if same_label and close_enough:
                prev["end_time"] = ev["end_time"]
                prev["confidence"] = max(prev["confidence"], ev["confidence"])
            else:
                merged.append(ev.copy())

        return merged
