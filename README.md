# Intelligent CC Suggestion Tool

> **DMP 2026 · PlanetRead · C4GT**

AI-powered tool that intelligently identifies moments in a video where a Closed Caption (CC) annotation is genuinely necessary — such as when a non-speech audio event meaningfully affects the speakers or the scene — and suggests contextually relevant CC text, without over-captioning routine or low-impact sounds.

## Architecture

```
Video → Audio Extraction → YAMNet Detection → Speech Filtering
     → Scene Cut Detection → Reaction Window Frame Extraction
     → Pose Analysis (flinch, head turn) + Face Analysis (surprise)
     → Category-Aware Fusion Engine → SRT Output
```

### Key Innovations

1. **Temporal Reaction Windows** — Extracts frames 300ms–1500ms *after* the sound (when reactions actually happen), not at the midpoint
2. **Category-Aware Fusion** — Different sound types use different weights (explosions don't need visual confirmation; doorbells do)
3. **Scene Cut Detection** — Skips visual analysis at edit points to prevent false positive reactions
4. **Overcaption Prevention** — Primary design goal is to filter ambient/insignificant sounds, not just detect everything

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
sudo apt install ffmpeg

# Run on a video
python main.py video.mp4

# Run with options
python main.py video.mp4 -o captions.srt --verbose

# Run with custom threshold
python main.py video.mp4 --threshold 0.35

# Run evaluation
python main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json
```

## Output

The tool produces:
- `<video>_cc.srt` — Standard SRT subtitle file with CC annotations
- `<video>_cc_summary.txt` — Human-readable summary showing accepted/rejected events

### Example SRT Output

```
1
00:00:12,480 --> 00:00:13,440
[gunshot]

2
00:00:28,320 --> 00:00:28,800
[glass breaking]
```

## Configuration

All thresholds are tunable via YAML config files — zero hardcoded magic numbers.

- `config/default.yaml` — Pipeline settings (confidence thresholds, reaction window timing, fusion weights)
- `config/sound_categories.yaml` — Category-aware weights per sound type

### Sound Categories

| Category | Examples | Behavior |
|---|---|---|
| **high_impact** | Gunshot, Explosion, Scream | Caption even without visual reaction |
| **interactive** | Doorbell, Knock, Dog bark | Only caption if someone visibly reacts |
| **social** | Laughter, Applause, Crying | Context-dependent |
| **ambient** | Music, Rain, Traffic | Almost never caption |

## Project Structure

```
├── config/
│   ├── default.yaml            # Pipeline settings
│   └── sound_categories.yaml   # Category-aware weights
├── src/
│   ├── pipeline.py             # Full orchestrator
│   ├── config_loader.py        # YAML config loading
│   ├── audio/
│   │   ├── extractor.py        # ffmpeg audio extraction
│   │   ├── yamnet_detector.py  # YAMNet sound event detection
│   │   └── speech_filter.py    # WebRTC VAD speech filtering
│   ├── visual/
│   │   ├── scene_cut.py        # Histogram-based cut detection
│   │   ├── frame_extractor.py  # Temporal reaction window
│   │   ├── pose_analyzer.py    # MediaPipe Pose (flinch, head turn)
│   │   └── face_analyzer.py    # MediaPipe FaceMesh (surprise)
│   ├── fusion/
│   │   ├── category_mapper.py  # Sound category lookup
│   │   └── decision_engine.py  # Category-aware score fusion
│   └── output/
│       ├── srt_writer.py       # SRT file generation
│       └── label_mapper.py     # YAMNet class → CC label
├── eval/
│   └── evaluator.py            # IoU-based P/R/F1 evaluation
├── tests/
│   └── test_all.py             # Test suite
├── main.py                     # CLI entry point
└── requirements.txt
```

## Tech Stack

| Component | Tool |
|---|---|
| Audio extraction | ffmpeg |
| Sound detection | YAMNet (TensorFlow Hub) |
| Speech filtering | WebRTC VAD |
| Pose detection | MediaPipe Pose |
| Face analysis | MediaPipe Face Mesh |
| Scene cuts | OpenCV histogram comparison |
| Config | YAML |
| Output | Standard SRT |

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| Precision | ≥ 0.75 | Fraction of suggestions that are correct |
| Recall | ≥ 0.65 | Fraction of important events caught |
| Overcaption Rate | ≤ 0.15 | Fraction of suggestions that are false |

## License

MIT
