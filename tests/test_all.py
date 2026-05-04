"""Tests for config loading, audio pipeline, fusion logic, and output."""
import os
import sys
import tempfile
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_load_config(self):
        from src.config_loader import load_config
        config = load_config("config/default.yaml")
        assert "audio" in config
        assert "visual" in config
        assert "fusion" in config
        assert config["audio"]["sample_rate"] == 16000
        assert config["audio"]["vad_aggressiveness"] == 3

    def test_load_categories(self):
        from src.config_loader import load_sound_categories
        lookup, default = load_sound_categories("config/sound_categories.yaml")
        assert "Gunshot, gunfire" in lookup
        assert lookup["Gunshot, gunfire"]["category"] == "high_impact"
        assert lookup["Gunshot, gunfire"]["audio_weight"] == 0.85
        assert default["audio_weight"] == 0.6
        assert default["category"] == "default"


# ============================================================
# Speech Filter Tests
# ============================================================

class TestSpeechFilter:
    def test_speech_pause_detection(self):
        from src.audio.speech_filter import SpeechFilter
        sf = SpeechFilter(aggressiveness=2)
        speech_segments = [(0.0, 5.0), (10.0, 15.0)]

        # Event at t=5.5 — speech just ended at t=5.0
        assert sf.was_speech_before(5.5, speech_segments, window=1.0) is True

        # Event at t=8.0 — no speech in lookback window
        assert sf.was_speech_before(8.0, speech_segments, window=1.0) is False

        # Event at t=10.5 — speech is currently happening
        assert sf.was_speech_before(10.5, speech_segments, window=1.0) is True

    def test_during_speech_overlap(self):
        from src.audio.speech_filter import SpeechFilter
        sf = SpeechFilter(aggressiveness=2)
        speech_segments = [(0.0, 5.0)]

        # Event fully inside speech
        assert sf.is_during_speech(1.0, 3.0, speech_segments, 0.5) is True

        # Event fully outside speech
        assert sf.is_during_speech(6.0, 8.0, speech_segments, 0.5) is False

        # Event partially overlapping (less than 50%)
        assert sf.is_during_speech(4.0, 8.0, speech_segments, 0.5) is False


# ============================================================
# Event Merging Tests
# ============================================================

class TestEventMerging:
    def test_merge_consecutive_same_label(self):
        """Consecutive events with same label should merge, taking peak confidence."""
        events = [
            {"label": "Gunshot", "confidence": 0.7, "start_time": 0.0, "end_time": 0.48},
            {"label": "Gunshot", "confidence": 0.9, "start_time": 0.48, "end_time": 0.96},
            {"label": "Glass", "confidence": 0.5, "start_time": 1.44, "end_time": 1.92},
        ]

        # Simulate merging logic
        merge_gap = 0.1
        merged = [events[0].copy()]
        for ev in events[1:]:
            prev = merged[-1]
            if ev["label"] == prev["label"] and ev["start_time"] <= prev["end_time"] + merge_gap:
                prev["end_time"] = ev["end_time"]
                prev["confidence"] = max(prev["confidence"], ev["confidence"])
            else:
                merged.append(ev.copy())

        assert len(merged) == 2
        assert merged[0]["confidence"] == 0.9   # peak confidence
        assert merged[0]["end_time"] == 0.96     # extended
        assert merged[1]["label"] == "Glass"

    def test_no_merge_different_labels(self):
        events = [
            {"label": "Gunshot", "confidence": 0.7, "start_time": 0.0, "end_time": 0.48},
            {"label": "Glass", "confidence": 0.5, "start_time": 0.48, "end_time": 0.96},
        ]

        merge_gap = 0.1
        merged = [events[0].copy()]
        for ev in events[1:]:
            prev = merged[-1]
            if ev["label"] == prev["label"] and ev["start_time"] <= prev["end_time"] + merge_gap:
                prev["end_time"] = ev["end_time"]
                prev["confidence"] = max(prev["confidence"], ev["confidence"])
            else:
                merged.append(ev.copy())

        assert len(merged) == 2  # should NOT merge


# ============================================================
# Fusion / Decision Engine Tests
# ============================================================

class TestDecisionEngine:
    def _get_engine(self):
        import yaml
        from src.fusion.category_mapper import CategoryMapper
        from src.fusion.decision_engine import DecisionEngine
        config = yaml.safe_load(open("config/default.yaml"))
        mapper = CategoryMapper("config/sound_categories.yaml")
        return DecisionEngine(config, mapper)

    def test_high_impact_accepted_without_visual(self):
        """Explosion with no visual reaction should still be captioned."""
        engine = self._get_engine()
        events = [{
            "id": 1, "label": "Explosion", "confidence": 0.7,
            "start_time": 5.0, "end_time": 5.5,
            "reaction_score": 0.0, "on_scene_cut": False, "speech_paused": False,
        }]
        result = engine.decide(events)
        # high_impact: alpha=0.85, threshold=0.30
        # combined = 0.85 * 0.7 = 0.595 >= 0.30 -> ACCEPT
        assert len(result) == 1
        assert result[0]["accepted"] is True

    def test_ambient_rejected_without_visual(self):
        """Background music with no reaction should be rejected."""
        engine = self._get_engine()
        events = [{
            "id": 1, "label": "Music", "confidence": 0.6,
            "start_time": 10.0, "end_time": 12.0,
            "reaction_score": 0.0, "on_scene_cut": False, "speech_paused": False,
        }]
        result = engine.decide(events)
        # ambient: alpha=0.25, threshold=0.70
        # combined = 0.25 * 0.6 = 0.15 < 0.70 -> REJECT
        assert len(result) == 0

    def test_interactive_needs_reaction(self):
        """Doorbell without reaction -> rejected. With reaction -> accepted."""
        engine = self._get_engine()

        # Without reaction
        events = [{
            "id": 1, "label": "Doorbell", "confidence": 0.7,
            "start_time": 5.0, "end_time": 5.5,
            "reaction_score": 0.0, "on_scene_cut": False, "speech_paused": False,
        }]
        result = engine.decide(events)
        assert len(result) == 0  # REJECT

        # With reaction
        events[0]["reaction_score"] = 0.6
        events[0]["accepted"] = False
        result = engine.decide(events)
        assert len(result) == 1  # ACCEPT

    def test_scene_cut_audio_only(self):
        """Events on scene cuts should ignore visual score."""
        engine = self._get_engine()
        events = [{
            "id": 1, "label": "Gunshot, gunfire", "confidence": 0.8,
            "start_time": 5.0, "end_time": 5.5,
            "reaction_score": 0.9,  # should be IGNORED
            "on_scene_cut": True, "speech_paused": False,
        }]
        result = engine.decide(events)
        assert len(result) == 1
        # Combined = 0.8 (audio only), threshold = max(0.30, 0.50) = 0.50
        assert result[0]["combined_score"] == 0.8

    def test_speech_pause_bonus(self):
        """Speech pause should add bonus to combined score."""
        engine = self._get_engine()
        events = [{
            "id": 1, "label": "Knock", "confidence": 0.5,
            "start_time": 5.0, "end_time": 5.5,
            "reaction_score": 0.3, "on_scene_cut": False, "speech_paused": True,
        }]
        result = engine.decide(events)
        # interactive: alpha=0.40, beta=0.60, threshold=0.50
        # combined = 0.40*0.5 + 0.60*0.3 + 0.15 = 0.20 + 0.18 + 0.15 = 0.53
        assert len(result) == 1


# ============================================================
# Output Tests
# ============================================================

class TestOutput:
    def test_srt_timestamp_format(self):
        from src.output.srt_writer import format_timestamp
        assert format_timestamp(0) == "00:00:00,000"
        assert format_timestamp(65.5) == "00:01:05,500"
        assert format_timestamp(3723.123) == "01:02:03,123"

    def test_srt_file_structure(self):
        from src.output.srt_writer import write_srt

        events = [
            {"start_time": 12.48, "end_time": 13.44, "cc_text": "[gunshot]"},
            {"start_time": 5.0, "end_time": 6.0, "cc_text": "[explosion]"},
        ]

        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = f.name

        write_srt(events, path)

        with open(path, 'r') as f:
            content = f.read()

        # Should be sorted by start_time
        assert content.index("[explosion]") < content.index("[gunshot]")
        # Should have correct SRT structure
        assert "00:00:05,000 --> 00:00:06,000" in content
        assert "00:00:12,480 --> 00:00:13,440" in content
        os.remove(path)

    def test_label_mapping(self):
        from src.output.label_mapper import map_label
        assert map_label("Gunshot, gunfire") == "[gunshot]"
        assert map_label("Explosion") == "[explosion]"
        assert map_label("Doorbell") == "[doorbell]"
        assert map_label("Drum") == "[drums]"  # India-specific
        assert map_label("Fireworks") == "[firecrackers]"  # India-specific

    def test_label_fallback(self):
        from src.output.label_mapper import map_label
        # Unknown class should fallback to first word
        result = map_label("SomeUnknownClass")
        assert result.startswith("[")
        assert result.endswith("]")


# ============================================================
# Evaluator Tests
# ============================================================

class TestEvaluator:
    def test_perfect_predictions(self):
        from eval.evaluator import evaluate
        pred = [{"start_time": 5.0, "end_time": 6.0}]
        gt = [{"start_time": 5.0, "end_time": 6.0}]
        result = evaluate(pred, gt)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["overcaption_rate"] == 0.0

    def test_overcaption_detection(self):
        from eval.evaluator import evaluate
        pred = [
            {"start_time": 5.0, "end_time": 6.0},
            {"start_time": 10.0, "end_time": 11.0},  # false positive
            {"start_time": 15.0, "end_time": 16.0},  # false positive
        ]
        gt = [{"start_time": 5.0, "end_time": 6.0}]
        result = evaluate(pred, gt)
        assert result["tp"] == 1
        assert result["fp"] == 2
        assert result["overcaption_rate"] == round(2/3, 4)

    def test_no_predictions(self):
        from eval.evaluator import evaluate
        result = evaluate([], [{"start_time": 1.0, "end_time": 2.0}])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["fn"] == 1

    def test_temporal_iou(self):
        from eval.evaluator import compute_temporal_iou
        # Perfect overlap
        assert compute_temporal_iou(
            {"start_time": 0, "end_time": 1},
            {"start_time": 0, "end_time": 1}
        ) == 1.0
        # No overlap
        assert compute_temporal_iou(
            {"start_time": 0, "end_time": 1},
            {"start_time": 2, "end_time": 3}
        ) == 0.0
        # 50% overlap
        iou = compute_temporal_iou(
            {"start_time": 0, "end_time": 2},
            {"start_time": 1, "end_time": 3}
        )
        assert abs(iou - 1/3) < 0.01  # overlap=1, union=3
