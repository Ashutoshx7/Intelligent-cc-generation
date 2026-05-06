# [DMP 2026] Intelligent CC Suggestion Pipeline — Complete Implementation (Goals 1, 2 & 3)

Closes #2

## 🎬 Demo Video

📹 **[Watch the full demo](PASTE_YOUR_LINK_HERE)** — screen recording showing CLI pipeline + HTML report + Web UI running end-to-end on a 15-second test clip.

---

## Why This Approach

The core problem isn't detecting sounds — it's deciding which sounds **matter**. A car horn in traffic is ambient noise; the same horn causing a speaker to flinch on camera demands a CC. This requires **category-aware multi-modal reasoning**, not just flat confidence thresholds.

I built the entire pipeline before submitting because I wanted to prove the architecture works, not just describe it.

---

## Architecture

```
Video → ffmpeg Audio Extraction (+ OpenCV fallback)
     → YAMNet (521 AudioSet classes) + Speech Filtering (WebRTC VAD)
     → Scene Cut Detection (Bhattacharyya histogram distance)
     → Temporal Reaction Window (300–1500ms after event onset)
     → Multi-Person Pose Analysis (flinch, head turn) + Face Analysis (surprise)
     → Category-Aware Fusion Engine (4 behavioral categories)
     → SRT + JSON + HTML Report Output
```

```
src/
├── audio/              ← Goal 1: Sound Event Detection
│   ├── extractor.py         ffmpeg + OpenCV fallback
│   ├── yamnet_detector.py   YAMNet 521-class detection
│   └── speech_filter.py     WebRTC VAD + energy fallback
├── visual/             ← Goal 2: Visual Reaction Detection
│   ├── scene_cut.py         Bhattacharyya histogram
│   ├── frame_extractor.py   Temporal reaction window
│   ├── pose_analyzer.py     MediaPipe PoseLandmarker (4 people)
│   └── face_analyzer.py     MediaPipe FaceLandmarker (4 people)
├── fusion/             ← Goal 3: CC Decision Engine
│   ├── category_mapper.py   4 behavioral categories
│   └── decision_engine.py   Category-aware fusion
├── output/             ← Multi-format output
│   ├── srt_writer.py        Standard SRT
│   ├── label_mapper.py      120+ AudioSet class mappings
│   └── report_generator.py  JSON + HTML reports
├── pipeline.py         ← Orchestrator
config/
├── default.yaml             All thresholds (zero hardcoded values)
└── sound_categories.yaml    Category weights & thresholds
web/                    ← Bonus: Editorial Review UI
├── app.py                   FastAPI backend
└── static/                  Monochrome interface
eval/                   ← Evaluation Framework
└── evaluator.py             IoU-based P/R/F1
tests/
└── test_all.py              30 passing tests
```

---

## Goal 1 — Sound Event Detection (`src/audio/`)

- **YAMNet via TensorFlow Hub** — 521 AudioSet classes, speech classes (indices 0-6) filtered out
- **WebRTC VAD** speech pre-filter (aggressiveness 3 for dense Hindi dialogue) with energy-based fallback for environments where WebRTC can't install
- **Consecutive event merging** — adjacent events with the same label merge, keeping peak confidence
- **120+ label mappings** — YAMNet class names → human-readable CC labels including India-specific sounds (Drum→[drums], Bell→[bell], Fireworks→[firecrackers])
- All thresholds in `config/default.yaml` — zero hardcoded values

## Goal 2 — Visual Reaction Detection (`src/visual/`)

- **Temporal reaction windows** — frames extracted at **300–1500ms after** the sound onset (not at midpoint), because human reactions have latency
- **Multi-person detection** — `PoseLandmarker(num_poses=4)` + `FaceLandmarker(num_faces=4)`, peak score across all detected persons
- **Scene cut detection** — Bhattacharyya distance on HSV histograms; events on cuts skip visual analysis entirely and use raised audio-only thresholds
- **Peak reaction scoring** — `max(frame_scores)` not average, because reactions are spiky not sustained

## Goal 3 — CC Decision Engine (`src/fusion/`)

**Category-aware fusion** — different sounds need different evidence thresholds:

| Category | Example | Audio Weight (α) | Visual Weight (β) | Threshold |
|---|---|---|---|---|
| `high_impact` | Gunshot, Explosion | 0.85 | 0.15 | 0.30 |
| `interactive` | Doorbell, Knock | 0.40 | 0.60 | 0.45 |
| `social` | Laughter, Applause | 0.55 | 0.45 | 0.40 |
| `ambient` | Traffic, Rain | 0.25 | 0.75 | 0.70 |

Additional signals used in fusion:
- **Speech-pause bonus** (+0.15) — if someone was speaking just before the event and stopped, the sound likely caused a reaction
- **Scene-cut audio-only mode** — threshold raised to ≥0.50 when visual analysis is unreliable
- **Duration splitting** — events >3s split into subtitle-standard chunks

---

## Output Formats

| Format | File | Purpose |
|---|---|---|
| SRT | `*_cc.srt` | Standard subtitle file for any video editor |
| Summary | `*_cc_summary.txt` | Human-readable accept/reject decisions |
| JSON | `*_cc_report.json` | Machine-readable, for downstream integration |
| HTML | `*_cc_report.html` | Professional visual report for editor review |

---

## Hindi/Regional Content Support

- WebRTC VAD at aggressiveness=3 for dense Hindi speech
- India-specific label mappings: Fireworks→[firecrackers], Drum→[drums], Bell→[bell], Tabla→[tabla], Flute→[flute], Gong→[gong]
- SRT format compatible with SLS karaoke subtitle workflows

---

## Web UI (Bonus — not required by issue)

Built a full editorial review interface with FastAPI:
- Drag-and-drop video upload
- Real-time processing with stage updates
- Interactive event review — video player + timeline + accept/reject toggles
- One-click SRT download with only accepted captions
- Minimalist monochrome design

---

## Testing

- **30 unit/integration tests** covering:
  - Config loading and sound category validation
  - Speech filter (VAD pause detection, overlap calculation)
  - Event merging logic
  - Fusion decisions (high impact, ambient, interactive, scene cuts, speech pause bonus)
  - SRT formatting and file structure
  - Label mapping (120+ classes including India-specific)
  - Report generation (JSON structure, HTML elements, filter rate)
  - Energy VAD thresholds
- **IoU-based evaluation** with precision/recall/F1/overcaption rate
- Run: `python3 -m pytest tests/test_all.py -v`

---

## Sample Output

```
python3 demo.py samples/demo_clip.avi
```

Pipeline detects 15 events, filters to 6 accepted CCs:
```
#  1  [white]    [ambient]  → REJECT  (background noise)
#  6  [rustle]   [default]  → ACCEPT  (speech paused before event)
# 10  [beep]     [default]  → ACCEPT  (high audio confidence 0.87)
# 15  [music]    [ambient]  → REJECT  (ambient, no visual reaction)
```

Output SRT:
```
1
00:00:03,840 --> 00:00:04,800
[rustle]

2
00:00:07,680 --> 00:00:08,160
[beep]
```

---

## Known Limitations

1. **YAMNet is AudioSet-trained (English/Western-centric)** — Indian-specific sounds may classify generically. Mitigation: substring label mapper + PANNs can be swapped in.
2. **Reaction window (300–1500ms)** is configurable but may miss very fast or very slow reactions.
3. **ffmpeg preferred** — without it, OpenCV fallback generates silent WAV (visual-only mode).
4. **WebRTC VAD** may not install on all platforms — falls back to energy-based VAD automatically.
5. **Confidence calibration** — YAMNet softmax scores are not true probabilities.

## What I'd Improve During the Coding Period

1. Benchmark on real PlanetRead Hindi content with editor feedback
2. Swap in PANNs for finer-grained Indian sound classification
3. Per-class confidence calibration on representative samples
4. Expose category weights in the Web UI for real-time editor tuning
5. Full 521-class label taxonomy mapping
6. Persistent job storage (SQLite) for multi-user deployment

---

## How to Run

```bash
# One-command setup
chmod +x setup.sh && ./setup.sh

# CLI — process a video
python3 main.py video.mp4 --verbose

# Formatted demo
python3 demo.py samples/demo_clip.avi

# Web UI
python3 web/app.py   # → http://localhost:8000

# Tests
python3 -m pytest tests/test_all.py -v

# Evaluation against ground truth
python3 main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json
```
