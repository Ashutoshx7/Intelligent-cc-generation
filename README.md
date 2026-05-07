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
4. **Top-3 High-Impact Priority** — When a dangerous sound (gunshot, explosion) appears in YAMNet's top 3 predictions, it's selected even if not the #1 class
5. **Multi-Person Detection** — Analyzes up to 4 people per frame, takes peak reaction score
6. **Overcaption Prevention** — Primary design goal is to filter ambient/insignificant sounds, not just detect everything (90% filter rate on real content)

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
- **Live CC Overlay** — Captions appear on the video player in real-time during playback, styled by category

## Output

The CLI produces:
- `<video>_cc.srt` — Standard SRT subtitle file with CC annotations
- `<video>_cc.sls` — SLS (Same Language Subtitling) format with score metadata
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
│   ├── test_all.py              # 30-test suite
│   └── generate_test_data.py    # Synthetic video/audio generator
├── main.py                      # CLI entry point
├── setup.sh                     # One-command setup
└── requirements.txt
```

## Testing

```bash
# Run all tests (30 tests)
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
| Audio extraction | ffmpeg (with moviepy fallback) |
| Sound detection | YAMNet (TensorFlow Hub, 521 classes) |
| Speech filtering | WebRTC VAD (with energy-based fallback) |
| Pose detection | MediaPipe PoseLandmarker (Tasks API) |
| Face analysis | MediaPipe FaceLandmarker (Tasks API) |
| Scene cuts | OpenCV histogram comparison |
| Config | YAML (all thresholds tunable) |
| Output | Standard SRT + SLS (PlanetRead) |
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

## Known Limitations

1. **YAMNet is AudioSet-trained (English/Western-centric)** — Indian-specific sounds (dhol, pressure cooker whistle, temple bells) may classify under generic labels. Mitigation: substring-based label mapper handles this, and PANNs can be swapped in via the fixed data contract.
2. **Single-frame vs. multi-frame tradeoff** — We extract 5 frames in the reaction window (300–1500ms). For very fast reactions (<300ms) or slow dramatic reactions (>1500ms), the window may miss. The window is configurable in `default.yaml`.
3. **No GPU required but slower on CPU** — YAMNet + MediaPipe run on CPU. A 10s video processes in ~4s. Longer videos scale linearly.
4. **ffmpeg preferred for audio** — Without system ffmpeg, moviepy (bundled ffmpeg) handles extraction. Both produce full-fidelity audio.
5. **WebRTC VAD may not install on all platforms** — Falls back to energy-based VAD automatically, which is less accurate for dense Hindi dialogue.
6. **Confidence calibration** — YAMNet softmax scores are not true probabilities. Per-class calibration on representative Hindi content would improve threshold accuracy.

## What I'd Improve Next

1. **Benchmark on real PlanetRead content** — Tune thresholds and category weights on actual Hindi/regional videos with editor feedback
2. **PANNs backend** — Swap in PANNs for finer-grained classification (the data contract makes this a drop-in)
3. **Confidence calibration** — Per-class percentile normalization on a representative sample
5. **Persistent job storage** — Move from in-memory to SQLite/Redis for multi-user web deployment
6. **VTT output format** — Trivially derivable from SRT, not yet implemented
7. **Threshold tuning UI** — Expose category weights in the web interface for real-time editor adjustment

## License

MIT
