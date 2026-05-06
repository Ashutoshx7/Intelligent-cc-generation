# DMP 2026
# Project Proposal for PlanetRead

---

## Project Summary

**Project:** Intelligent Closed Caption (CC) Suggestion Tool
**Mentors:** @keerthiseelan-planetread, @abinash-sketch
**Issue:** [DMP 2026]: Create Intelligent Closed Caption (CC) Suggestion Tool #2

This project builds an AI-powered tool that watches a video — both what you **hear** and what you **see** — and suggests only the closed captions that genuinely matter. A car horn in traffic is ambient noise; the same horn causing a speaker to flinch on camera demands a CC. The system uses a YAMNet-based audio classifier, a MediaPipe-based visual reaction detector, and a **category-aware fusion engine** that weights different types of sounds differently, because explosions and doorbells need fundamentally different evidence thresholds to justify a caption.

The tool produces standard SRT files ready for any subtitle editor, plus JSON and HTML reports for editorial review.

### 🎬 Demo Video
📹 **[Watch the full demo](PASTE_YOUR_LINK_HERE)** — screen recording showing CLI pipeline + HTML report + Web UI running end-to-end.

---

## Project Vision and Motivation

### Vision

My vision is to make closed captioning **intelligent, not exhaustive**. Current CC workflows are either fully manual (editors watch hours of footage and decide caption-by-caption) or fully automated (detect every sound and caption everything). The first approach doesn't scale. The second produces overcaptioned content that fatigues viewers and buries the important moments under noise.

This tool sits in the middle: detect all sounds, but **only suggest captions for sounds that narratively matter**. A siren in a Bollywood action scene where the protagonist freezes? Caption it. Background traffic on a highway interview? Filter it. The difference between these two decisions requires understanding both the **audio signal** and the **visual reaction** — and that's exactly what this tool does.

### Motivation

My motivation comes from the intersection of AI, accessibility, and real-world content.

PlanetRead's Same Language Subtitling (SLS) program has subtitled over 40,000 hours of Bollywood content, reaching 800 million people across India. But SLS is primarily about subtitling speech. The question this project answers is: **what about the sounds between the words?** A door slamming, a phone ringing, firecrackers during Diwali — these sounds carry narrative weight, especially for deaf and hard-of-hearing viewers who can't hear them at all.

I've spent the past year building AI systems — semantic search at Beckn, LLM evaluation at SuperKalam, retrieval pipelines with Qdrant and Elasticsearch. But what drew me to this project specifically is that it's not just a technical challenge. It's about making content **accessible** to millions of people who experience video without sound. Every correctly placed `[gunshot]` or `[doorbell]` caption gives a deaf viewer information they would otherwise miss entirely.

The technical challenge is also genuinely interesting: this isn't a classification problem (that's solved — YAMNet handles it). This is a **decision** problem. Given that a sound exists, should it be captioned? That requires reasoning about context, category, and visual evidence — exactly the kind of multi-modal fusion problem I wanted to build.

---

## Architecture Overview

The system is organized as a three-goal pipeline, matching the structure outlined in the issue. Each goal is a self-contained module with a fixed data contract, meaning any component can be replaced (e.g., swapping YAMNet for PANNs) without changing the orchestration logic.

```
Video File
  │
  ├── ffmpeg Audio Extraction (+ OpenCV silent-WAV fallback)
  │
  ▼
┌──────────────────────────────────────────────────┐
│  GOAL 1: Sound Event Detection  (src/audio/)     │
│  ├── YAMNet — 521 AudioSet classes               │
│  ├── WebRTC VAD speech filter (aggressiveness=3)  │
│  ├── Energy-based VAD fallback                    │
│  └── Consecutive event merging (peak confidence)  │
└──────────────────────┬───────────────────────────┘
                       │ events[]
                       ▼
┌──────────────────────────────────────────────────┐
│  GOAL 2: Visual Reaction Detection (src/visual/) │
│  ├── Scene cut detection (Bhattacharyya distance) │
│  ├── Temporal frame extraction (300–1500ms after) │
│  ├── Multi-person pose analysis (4 people/frame)  │
│  └── Multi-person face analysis (4 faces/frame)   │
└──────────────────────┬───────────────────────────┘
                       │ events[] + reaction scores
                       ▼
┌──────────────────────────────────────────────────┐
│  GOAL 3: CC Decision Engine  (src/fusion/)       │
│  ├── Category-aware fusion (4 categories)        │
│  ├── Speech-pause bonus (+0.15)                  │
│  ├── Scene-cut audio-only mode                   │
│  └── Duration splitting (≤3s per CC)             │
└──────────────────────┬───────────────────────────┘
                       │ accepted events[]
                       ▼
┌──────────────────────────────────────────────────┐
│  OUTPUT  (src/output/)                           │
│  ├── SRT subtitle file (standard format)         │
│  ├── JSON report (machine-readable)              │
│  ├── HTML report (editor review)                 │
│  └── TXT summary (human-readable)                │
└──────────────────────────────────────────────────┘
```

### Module Map

| Module | Role | Layer |
|---|---|---|
| `src/audio/extractor.py` | ffmpeg audio extraction + OpenCV fallback | Audio I/O |
| `src/audio/yamnet_detector.py` | YAMNet 521-class sound detection | Goal 1 |
| `src/audio/speech_filter.py` | WebRTC VAD + energy fallback | Goal 1 |
| `src/visual/scene_cut.py` | Bhattacharyya histogram scene cut detection | Goal 2 |
| `src/visual/frame_extractor.py` | Temporal reaction window (300–1500ms) | Goal 2 |
| `src/visual/pose_analyzer.py` | MediaPipe PoseLandmarker (multi-person) | Goal 2 |
| `src/visual/face_analyzer.py` | MediaPipe FaceLandmarker (multi-person) | Goal 2 |
| `src/fusion/category_mapper.py` | Sound → behavioral category mapping | Goal 3 |
| `src/fusion/decision_engine.py` | Category-aware fusion + accept/reject | Goal 3 |
| `src/output/srt_writer.py` | Standard SRT generation | Output |
| `src/output/label_mapper.py` | 114 YAMNet → CC label mappings | Output |
| `src/output/report_generator.py` | JSON + HTML report generation | Output |
| `src/config_loader.py` | YAML config loading (zero hardcoded values) | Config |
| `src/pipeline.py` | End-to-end orchestrator | Core |
| `web/app.py` | FastAPI editorial review interface | Web UI |
| `eval/evaluator.py` | IoU-based precision/recall/F1 evaluation | Evaluation |
| `demo.py` | Formatted CLI demo for screen recording | Demo |
| `tests/test_all.py` | 30 unit/integration tests | Testing |

---

## Detailed Proposal Description

### Goal 1 — Sound Event Detection (`src/audio/`)

The first goal is to detect **non-speech audio events** in the video. The key challenge here is not detection (YAMNet handles that) — it's **filtering**. In Hindi content, dialogue is dense and continuous. Without proper speech filtering, the detector would flag every moment where a sound overlaps with speech.

#### Audio Extraction (`extractor.py`)

```
Video → ffmpeg → 16kHz mono WAV
         │
         └── [fallback] OpenCV → silent WAV (visual-only mode)
```

- Primary: `ffmpeg -i video.mp4 -ar 16000 -ac 1 -f wav output.wav`
- Fallback: If ffmpeg is not installed, generates a silent WAV file using OpenCV. The pipeline continues in **visual-only mode** — it can still detect scene cuts and reactions, just without audio events.
- This fallback ensures the tool runs on any system, even without ffmpeg.

#### YAMNet Sound Classifier (`yamnet_detector.py`)

- **Model:** YAMNet via TensorFlow Hub — trained on AudioSet (2.1M YouTube clips, 521 event classes)
- **Window:** Processes audio in 0.48s windows with 50% overlap
- **Speech filtering:** YAMNet classes 0–6 are speech classes (`Speech`, `Male speech`, `Female speech`, `Child speech`, `Conversation`, `Narration`, `Babbling`). These are **hard-filtered** — any detection window where the top class is speech is discarded entirely.
- **Confidence threshold:** Configurable via `config/default.yaml` (default: 0.35). Events below this confidence are discarded.
- **Event merging:** Consecutive windows with the same label are merged into a single event. The merged event keeps the **peak confidence** (not average), because a strong 0.9 detection followed by a weak 0.4 detection is one continuous event at 0.9 confidence.

#### Speech Filter (`speech_filter.py`)

A two-layer speech detection system designed for dense Hindi dialogue:

**Layer 1: WebRTC VAD (primary)**
- Google's Voice Activity Detection library, running at aggressiveness level 3 (most aggressive — marks more frames as speech)
- Processes audio in 30ms frames
- Outputs speech segments as `(start_time, end_time)` pairs

**Layer 2: Energy-based VAD (fallback)**
- Pure Python — no C dependencies, runs anywhere
- Computes RMS energy per 30ms frame
- Frames above threshold → speech
- Threshold tuned per aggressiveness level:

| Aggressiveness | Energy Threshold | Use Case |
|---|---|---|
| 0 | 0.04 | Light filtering |
| 1 | 0.03 | Moderate |
| 2 | 0.02 | Aggressive |
| 3 | 0.015 | Dense Hindi dialogue |

**Speech-pause detection:** `was_speech_before(timestamp, segments, window=1.0)` checks if speech was active in the 1-second window before an event. If speech was happening and stopped right before a sound, it's a strong signal that the speaker reacted to the sound. This feeds into the fusion engine as a `+0.15` bonus.

**Overlap detection:** `is_during_speech(start, end, segments, overlap_ratio=0.5)` checks if an event overlaps with speech by more than 50%. Events heavily overlapping with speech are deprioritized.

---

### Goal 2 — Visual Reaction Detection (`src/visual/`)

The second goal is to determine whether anyone on screen **reacted** to a detected sound. This is the key differentiator between sounds that matter and sounds that don't.

#### Scene Cut Detection (`scene_cut.py`)

Before analyzing reactions, the system detects **scene cuts** (hard transitions between shots):

- Extracts frames from the video at configurable intervals
- Computes HSV histograms for consecutive frames
- Calculates **Bhattacharyya distance** between histograms
- Distance above threshold (default: 0.55) → scene cut detected

**Why this matters:** If a sound event falls on a scene cut, the "before" and "after" frames show completely different scenes. Any pose/face changes are due to the cut, not a reaction. Events on scene cuts skip visual analysis entirely and use **raised audio-only thresholds** (≥0.50 instead of category-specific).

#### Temporal Reaction Window (`frame_extractor.py`)

This is one of the most important design decisions in the project:

**The problem:** If a loud sound occurs at t=5.0s, when should we look for a reaction?

**Naive approach (what competitors do):** Extract a frame at the midpoint of the event. This misses the reaction entirely because:
1. Reactions have latency — a human flinch takes 300–500ms to appear on camera
2. The midpoint frame might show the person *before* they react

**Our approach:** Extract **5 frames** from a window starting **300ms after** the event onset and ending **1500ms after**:

```
Sound event:  |████|
              t=5.0  t=5.5

Reaction window:        |· · · · ·|
                     t=5.3      t=6.5
                     (300ms)    (1500ms)
```

- Frame 1: t + 300ms (earliest possible reaction)
- Frame 2: t + 600ms
- Frame 3: t + 900ms
- Frame 4: t + 1200ms
- Frame 5: t + 1500ms (latest typical reaction)

The reaction score is `max(frame_scores)`, not average — because a reaction is a spike, not a sustained state. A 0.7 flinch at frame 3 followed by 0.1 at frames 4–5 is still a strong reaction.

#### Multi-Person Pose Analysis (`pose_analyzer.py`)

Uses MediaPipe PoseLandmarker to detect body-level reactions:

- **Multi-person:** `num_poses=4` — detects up to 4 people per frame (competitors use single-person detection)
- **Reaction signals detected:**
  - Shoulder flinch (sudden vertical displacement of shoulder landmarks)
  - Head turn (lateral displacement of nose landmark relative to shoulders)
  - Body lean (torso angle change)
- **Scoring:** Peak score across all detected persons — if *any* person reacts, the event gets credit

#### Multi-Person Face Analysis (`face_analyzer.py`)

Uses MediaPipe FaceLandmarker to detect facial reactions:

- **Multi-person:** `num_faces=4`
- **Reaction signals detected:**
  - Eye widening (increased distance between upper and lower eyelid landmarks)
  - Eyebrow raise (upward displacement of eyebrow landmarks)
  - Mouth opening (surprise expression — jaw drop)
- **Composite score:** Weighted combination of eye, eyebrow, and mouth signals

The final reaction score for each event is `max(pose_score, face_score)` across all frames in the temporal window and all detected persons.

---

### Goal 3 — CC Decision Engine (`src/fusion/`)

The third goal is the core innovation: a **category-aware fusion engine** that makes the final accept/reject decision for each event.

#### The Problem with Flat Thresholds

A naive approach uses the same threshold for every sound:

```
combined = 0.6 * audio + 0.4 * visual
if combined >= 0.5: caption
```

This fails because:
- A **gunshot** at 0.7 audio confidence with no visible reaction should *still* be captioned — gunshots are always narratively significant
- A **doorbell** at 0.7 audio confidence with no visible reaction should *not* be captioned — if nobody reacts to it, it's probably background
- **Background rain** at 0.9 audio confidence should almost never be captioned — unless someone on screen looks up at the sky

Different sounds need **fundamentally different evidence** to justify a caption.

#### Category-Aware Fusion

The system classifies every detected sound into one of four behavioral categories, each with its own weights and threshold:

| Category | Description | α (audio) | β (visual) | Threshold | Logic |
|---|---|---|---|---|---|
| `high_impact` | Gunshot, Explosion, Siren, Scream | 0.85 | 0.15 | 0.30 | Caption even without visual reaction |
| `interactive` | Doorbell, Knock, Phone, Dog bark | 0.40 | 0.60 | 0.50 | Only caption if someone reacts |
| `social` | Laughter, Applause, Crying, Cough | 0.55 | 0.45 | 0.45 | Context dependent |
| `ambient` | Rain, Wind, Traffic, Music, Engine | 0.25 | 0.75 | 0.70 | Almost never — needs strong visual |

The categories and their weights are defined in `config/sound_categories.yaml` — zero hardcoded values. An editor can adjust any threshold without touching code.

#### Fusion Formula

For each event:

```
1. Look up category → get α, β, threshold
2. If on_scene_cut:
     combined = audio_confidence  (visual unreliable)
     threshold = max(category_threshold, 0.50)
3. Else:
     combined = α × audio_confidence + β × reaction_score
4. If speech_paused:
     combined += 0.15  (speech-pause bonus)
5. Accept if combined ≥ threshold
```

#### Duration Splitting

Subtitle standards recommend no single caption exceed 3 seconds. Events longer than `max_cc_duration` (configurable, default 3.0s) are automatically split into multiple consecutive CCs.

---

### Output Formats (`src/output/`)

The pipeline generates four output files for every processed video:

| Format | Filename | Purpose |
|---|---|---|
| **SRT** | `video_cc.srt` | Standard subtitle format — importable into any video editor |
| **Summary** | `video_cc_summary.txt` | Human-readable accept/reject list with scores and reasoning |
| **JSON** | `video_cc_report.json` | Machine-readable full event dump — for downstream pipelines |
| **HTML** | `video_cc_report.html` | Professional dark-themed report for editor review |

#### Label Mapper (`label_mapper.py`)

Maps 114 YAMNet class names to human-readable CC bracket labels, organized by category:

- **High impact:** `Gunshot, gunfire` → `[gunshot]`, `Ambulance (siren)` → `[ambulance siren]`
- **Interactive:** `Doorbell` → `[doorbell]`, `Bark` → `[dog barking]`
- **Social:** `Crying, sobbing` → `[crying]`, `Baby cry, infant cry` → `[baby crying]`
- **India-specific:** `Drum` → `[drums]`, `Fireworks` → `[firecrackers]`, `Bell` → `[bell]`, `Tabla` → `[tabla]`, `Flute` → `[flute]`, `Gong` → `[gong]`

Substring matching handles compound YAMNet labels. Unknown classes fall back to first-word extraction: `SomeUnknownClass` → `[someunknownclass]`.

---

### Web UI — Editorial Review Interface (`web/`)

Built as a bonus feature — not required by the issue, but demonstrates a complete editorial workflow.

**Tech stack:** FastAPI + vanilla HTML/CSS/JS (no React, no npm, no build step)

**Design:** Minimalist monochrome (black and white) — tasteful, not flashy. Inter + JetBrains Mono typography. 960px max-width. CSS transitions on everything.

**Features:**
- **Upload screen:** Drag-and-drop or click to upload. Feature cards explain the three goals.
- **Processing screen:** Real-time progress bar with stage labels (Extracting audio → Running YAMNet → Scoring reactions → Running fusion)
- **Results screen:**
  - Stats bar: Detected / Accepted / Filtered / Filter Rate
  - Video player with playback
  - Interactive timeline with event markers (click to seek)
  - Event cards showing CC label, timestamps, audio/visual scores, category badge, and accept/reject toggle
  - Filter tabs: All / Accepted / Rejected
  - Live SRT preview that updates when you toggle events
  - "Download SRT" button exports only accepted events

---

### Evaluation Framework (`eval/`)

The system includes a built-in evaluation mode that computes standard metrics against ground-truth annotations.

**Metrics:**
- **Precision:** What fraction of our suggestions were correct?
- **Recall:** What fraction of real events did we catch?
- **F1-score:** Harmonic mean of precision and recall
- **Overcaption Rate:** What fraction of our suggestions were false positives? (This is the metric the issue cares about most)

**Matching:** Uses **Temporal IoU** (Intersection over Union) with a configurable threshold (default: 0.3) to match predicted events to ground-truth events.

**Usage:**
```bash
python3 main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json
```

---

### Testing — 30 Tests

The test suite covers every module:

| Test Class | Tests | What It Covers |
|---|---|---|
| `TestConfig` | 2 | Config loading, sound category parsing |
| `TestSpeechFilter` | 2 | Speech-pause detection, overlap calculation |
| `TestEventMerging` | 2 | Same-label merging, cross-label separation |
| `TestDecisionEngine` | 5 | High impact accept, ambient reject, interactive needs reaction, scene-cut mode, speech-pause bonus |
| `TestOutput` | 4 | SRT timestamps, file structure, label mapping, fallback |
| `TestEvaluator` | 4 | Perfect predictions, overcaption, no predictions, temporal IoU |
| `TestReportGenerator` | 3 | JSON structure, HTML elements, filter rate |
| `TestExtendedLabels` | 5 | India-specific, high impact, social, transport, nature |
| `TestEnergyVAD` | 3 | Threshold behavior, silent detection, loud detection |

```bash
python3 -m pytest tests/test_all.py -v
# 30 passed in 0.16s
```

---

## Hindi/Regional Content Design

The tool is specifically designed for Indian regional content:

- **WebRTC VAD aggressiveness=3:** Tuned for dense Hindi dialogue where speakers talk rapidly with minimal pauses
- **India-specific label mappings:** Fireworks→`[firecrackers]` (Diwali scenes), Drum→`[drums]` (dhol, tabla), Bell→`[bell]` (temple bells), Tabla→`[tabla]`, Flute→`[flute]` (bansuri), Gong→`[gong]`
- **SRT encoding:** UTF-8 by default, supporting Devanagari and other Indic scripts in CC text
- **SLS compatibility:** SRT output format is compatible with PlanetRead's SLS karaoke subtitle pipeline

---

## Known Limitations

1. **YAMNet is AudioSet-trained (English/Western-centric)** — Indian sounds like shehnai, mridangam, or conch shell may classify generically. Mitigation: substring label mapper + PANNs can be swapped in.
2. **Reaction window (300–1500ms)** is configurable but may miss very fast (<300ms) or very slow (>1500ms) reactions. Tunable via `config/default.yaml`.
3. **ffmpeg preferred** — without it, OpenCV fallback generates silent WAV (visual-only mode). Audio detection requires ffmpeg.
4. **WebRTC VAD** may not compile on all platforms — automatically falls back to energy-based VAD.
5. **Confidence calibration** — YAMNet softmax scores are not true probabilities. High-confidence ambient sounds may still be ambient.
6. **Single-machine processing** — no distributed processing; jobs run in-memory on a single server.

---

## What I'd Improve During the Coding Period

1. **Benchmark on real PlanetRead Hindi content** with editor feedback to calibrate thresholds
2. **Swap in PANNs** (Pretrained Audio Neural Networks) for finer-grained Indian sound classification
3. **Per-class confidence calibration** on representative content samples
4. **Expose category weights in the Web UI** for real-time editor tuning without config changes
5. **Full 521-class label taxonomy mapping** covering every YAMNet output
6. **Persistent job storage (SQLite)** to move beyond in-memory processing
7. **Batch processing** — process multiple videos via CLI or Web UI
8. **Collaboration hooks** — multiple editors reviewing the same video

---

## How to Run

```bash
# One-command setup
chmod +x setup.sh && ./setup.sh

# CLI — process a video
python3 main.py video.mp4 --verbose

# Formatted demo with color output
python3 demo.py samples/demo_clip.avi

# Web UI — editorial review interface
python3 web/app.py   # → http://localhost:8000

# Tests — 30 passing
python3 -m pytest tests/test_all.py -v

# Evaluation against ground truth
python3 main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json

# Generate demo test data
python3 tests/generate_demo_data.py
```

---

## Contact Information

**Name:** Ashutosh Singh
**Email:** ashutoshx002@gmail.com
**GitHub:** [amitkumarashutosh](https://github.com/amitkumarashutosh)
**Phone:** +91 95559 05213

---

## Conclusion

This tool doesn't just detect sounds — it **decides which sounds matter**. The category-aware fusion engine, the temporal reaction window, the multi-person visual analysis, and the India-specific label mappings together form a system that reduces overcaptioning while catching the moments that genuinely need CC annotations.

Every design decision — from the 300ms reaction delay to the 0.70 ambient threshold — is grounded in how humans actually perceive and react to sound in video content.

The 30 automated tests, the professional HTML report, the monochrome Web UI, and the evaluation framework demonstrate production readiness, not just a proof of concept.
