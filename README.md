# Intelligent CC Suggestion Tool

> **DMP 2026 · PlanetRead · C4GT**

AI-powered tool that identifies moments in a video where a Closed Caption (CC) annotation is genuinely necessary — such as when a non-speech audio event meaningfully affects the speakers or the scene — and suggests contextually relevant CC text, without over-captioning routine or low-impact sounds.

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
4. **Multi-Person Detection** — Analyzes up to 4 people per frame, takes peak reaction score
5. **Overcaption Prevention** — Primary design goal is to filter ambient/insignificant sounds, not just detect everything

## Setup

```bash
# One-command setup (installs deps + downloads models)
chmod +x setup.sh && ./setup.sh

# Or manually:
pip install -r requirements.txt
sudo apt install ffmpeg            # optional but recommended
```

The setup script downloads MediaPipe model files to `models/`.

## Usage

### CLI (Command Line)

```bash
# Basic — produces <video>_cc.srt
python main.py video.mp4

# With options
python main.py video.mp4 -o captions.srt --verbose

# Override fusion threshold
python main.py video.mp4 --threshold 0.35

# Evaluation mode — compares output against ground truth
python main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json
```

### Web UI

```bash
python web/app.py
# Open http://localhost:8000
```

The web interface provides:
- **Upload** — Drag-and-drop video files
- **Processing** — Real-time progress with pipeline stage updates
- **Review** — Video player, interactive timeline, event cards with accept/reject toggles
- **Export** — Download the final SRT file with only accepted captions

## Output

The CLI produces:
- `<video>_cc.srt` — Standard SRT subtitle file with CC annotations
- `<video>_cc_summary.txt` — Human-readable report showing accepted/rejected events with scores

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

All thresholds are tunable via YAML config — zero hardcoded magic numbers.

- `config/default.yaml` — Pipeline settings (confidence thresholds, reaction window timing, fusion weights)
- `config/sound_categories.yaml` — Category-aware weights per sound type

### Sound Categories

| Category | Examples | Behavior |
|---|---|---|
| **high_impact** | Gunshot, Explosion, Scream | Caption even without visual reaction (α=0.85) |
| **interactive** | Doorbell, Knock, Dog bark | Only caption if someone visibly reacts (β=0.60) |
| **social** | Laughter, Applause, Crying | Context-dependent (balanced weights) |
| **ambient** | Music, Rain, Traffic | Almost never caption (threshold=0.70) |

## Project Structure

```
├── config/
│   ├── default.yaml             # Pipeline settings
│   └── sound_categories.yaml    # Category-aware weights
├── src/
│   ├── pipeline.py              # Full orchestrator
│   ├── config_loader.py         # YAML config loading
│   ├── audio/
│   │   ├── extractor.py         # ffmpeg audio extraction (+ OpenCV fallback)
│   │   ├── yamnet_detector.py   # YAMNet sound event detection (521 classes)
│   │   └── speech_filter.py     # WebRTC VAD + energy-based fallback
│   ├── visual/
│   │   ├── scene_cut.py         # Histogram-based cut detection
│   │   ├── frame_extractor.py   # Temporal reaction window (300-1500ms)
│   │   ├── pose_analyzer.py     # MediaPipe Pose (flinch, head turn, multi-person)
│   │   └── face_analyzer.py     # MediaPipe Face (surprise/gasp, multi-face)
│   ├── fusion/
│   │   ├── category_mapper.py   # YAMNet class → behavioral category
│   │   └── decision_engine.py   # Category-aware score fusion + CC decision
│   └── output/
│       ├── srt_writer.py        # SRT file generation
│       └── label_mapper.py      # YAMNet class → CC label (India-specific)
├── eval/
│   ├── evaluator.py             # IoU-based P/R/F1 + overcaption rate
│   └── ground_truth/            # Manual annotations (JSON)
├── web/
│   ├── app.py                   # FastAPI backend
│   └── static/                  # Monochrome web UI
├── tests/
│   ├── test_all.py              # 19-test suite
│   └── generate_test_data.py    # Synthetic video/audio generator
├── main.py                      # CLI entry point
├── setup.sh                     # One-command setup
└── requirements.txt
```

## Testing

```bash
# Run all tests (19 tests)
python -m pytest tests/test_all.py -v

# Generate synthetic test data
python tests/generate_test_data.py

# Full end-to-end pipeline test
python main.py samples/test_clip.avi --verbose

# Evaluation test
python main.py samples/test_clip.avi --evaluate --ground-truth eval/ground_truth/test_clip.json
```

## Tech Stack

| Component | Tool |
|---|---|
| Audio extraction | ffmpeg (with OpenCV fallback) |
| Sound detection | YAMNet (TensorFlow Hub, 521 classes) |
| Speech filtering | WebRTC VAD (with energy-based fallback) |
| Pose detection | MediaPipe PoseLandmarker (Tasks API) |
| Face analysis | MediaPipe FaceLandmarker (Tasks API) |
| Scene cuts | OpenCV histogram comparison |
| Config | YAML (all thresholds tunable) |
| Output | Standard SRT |
| Web UI | FastAPI + Vanilla JS |

## Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| Precision | ≥ 0.75 | Fraction of suggestions that are correct |
| Recall | ≥ 0.65 | Fraction of important events caught |
| Overcaption Rate | ≤ 0.15 | Fraction of suggestions that are unnecessary |

## Hindi/Regional Content Support

- **Dense dialogue handling** — WebRTC VAD at aggressiveness=3 for Hindi speech
- **India-specific sounds** — Fireworks→[firecrackers], Drum→[drums], Bell→[bell]
- **SLS workflow compatible** — Standard SRT format overlays with karaoke subtitles

## License

MIT
