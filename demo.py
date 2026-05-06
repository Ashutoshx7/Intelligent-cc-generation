#!/usr/bin/env python3
"""
Demonstration script — runs all 3 Goals and prints formatted output.
Use this for the PR demo video recording.
"""
import os
import sys
import time

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'

def header(text):
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'═' * 60}{RESET}\n")

def step(text):
    print(f"  {BLUE}▸{RESET} {text}")

def ok(text):
    print(f"  {GREEN}✓{RESET} {text}")

def warn(text):
    print(f"  {YELLOW}⚠{RESET} {text}")

def fail(text):
    print(f"  {RED}✗{RESET} {text}")

def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "samples/test_clip.avi"

    if not os.path.exists(video):
        fail(f"Video not found: {video}")
        print(f"\n  Generate test data first: python tests/generate_test_data.py\n")
        sys.exit(1)

    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════╗
║   Intelligent CC Suggestion Tool — DMP 2026 · PlanetRead ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")
    print(f"  Input: {BOLD}{video}{RESET}")
    t_start = time.time()

    # ── GOAL 1 ──────────────────────────────────────────────
    header("GOAL 1: Sound Event Detection")

    step("Loading config...")
    from src.config_loader import load_config
    config = load_config("config/default.yaml")
    ok("Config loaded (all thresholds from YAML)")

    step("Extracting audio...")
    from src.audio.extractor import extract_audio, load_wav_as_float
    base = os.path.splitext(video)[0]
    wav_preexisted = os.path.exists(f"{base}_audio.wav")
    wav_path = extract_audio(video, sample_rate=config['audio']['sample_rate'])
    waveform, sr = load_wav_as_float(wav_path)
    duration = len(waveform) / sr
    ok(f"Audio: {duration:.1f}s at {sr}Hz")

    step("Running speech filter (VAD)...")
    from src.audio.speech_filter import SpeechFilter
    sf = SpeechFilter(aggressiveness=config['audio']['vad_aggressiveness'], sample_rate=sr)
    speech_segs = sf.get_speech_segments(waveform)
    speech_time = sum(e - s for s, e in speech_segs)
    ok(f"VAD: {len(speech_segs)} speech segments ({speech_time:.1f}s total)")

    step("Running YAMNet (521 AudioSet classes)...")
    from src.audio.yamnet_detector import YAMNetDetector
    detector = YAMNetDetector(config)
    all_events = detector.detect(waveform)
    ok(f"YAMNet: {len(all_events)} raw events detected")

    step("Filtering speech overlap...")
    events = [e for e in all_events if not sf.is_during_speech(e["start_time"], e["end_time"], speech_segs)]
    ok(f"After filter: {len(events)} non-speech events")

    if events:
        print(f"\n  {DIM}{'─' * 56}{RESET}")
        print(f"  {DIM}{'ID':>4}  {'Label':<25} {'Conf':>5}  {'Time'}{RESET}")
        print(f"  {DIM}{'─' * 56}{RESET}")
        for e in events:
            print(f"  {DIM}#{e['id']:>3}{RESET}  {e['label']:<25} {e['confidence']:.2f}   {e['start_time']:.1f}s → {e['end_time']:.1f}s")
        print(f"  {DIM}{'─' * 56}{RESET}")

    # ── GOAL 2 ──────────────────────────────────────────────
    header("GOAL 2: Visual Reaction Detection")

    step("Detecting scene cuts (histogram)...")
    from src.visual.scene_cut import SceneCutDetector
    cut_det = SceneCutDetector(config['visual']['scene_cut_threshold'])
    cuts = cut_det.detect_cuts(video)
    ok(f"Scene cuts: {len(cuts)} detected")

    step("Initializing MediaPipe models...")
    from src.visual.frame_extractor import FrameExtractor
    from src.visual.pose_analyzer import PoseAnalyzer
    from src.visual.face_analyzer import FaceAnalyzer
    fe = FrameExtractor(config)
    pa = PoseAnalyzer(config)
    fa = FaceAnalyzer(config)
    ok("PoseLandmarker + FaceLandmarker ready (multi-person)")

    step("Scoring reactions (300–1500ms window after each event)...")
    for event in events:
        on_cut = cut_det.is_on_scene_cut(event["start_time"], cuts, config['visual']['scene_cut_tolerance'])
        event["on_scene_cut"] = on_cut
        if on_cut:
            event["reaction_score"] = 0.0
            event["reaction_persons"] = 0
        else:
            frames = fe.extract_reaction_frames(video, event["start_time"])
            if not frames:
                event["reaction_score"] = 0.0
                event["reaction_persons"] = 0
            else:
                scores, max_p = [], 0
                for ts, frame in frames:
                    pr = pa.analyze(frame)
                    fr = fa.analyze(frame)
                    scores.append(max(pr["pose_score"], fr["face_score"]))
                    max_p = max(max_p, pr["num_persons"], fr["num_faces"])
                event["reaction_score"] = max(scores) if scores else 0.0
                event["reaction_persons"] = max_p
        event["speech_paused"] = sf.was_speech_before(event["start_time"], speech_segs)
    pa.close()
    fa.close()
    ok("Reaction scores computed")

    print(f"\n  {DIM}{'─' * 56}{RESET}")
    print(f"  {DIM}{'ID':>4}  {'Label':<25} {'Audio':>5} {'Visual':>6} {'Flags'}{RESET}")
    print(f"  {DIM}{'─' * 56}{RESET}")
    for e in events:
        flags = []
        if e.get("on_scene_cut"): flags.append("⚡cut")
        if e.get("speech_paused"): flags.append("🗣pause")
        flag_str = " ".join(flags)
        print(f"  {DIM}#{e['id']:>3}{RESET}  {e['label']:<25} {e['confidence']:.2f}  {e['reaction_score']:.2f}   {flag_str}")
    print(f"  {DIM}{'─' * 56}{RESET}")

    # ── GOAL 3 ──────────────────────────────────────────────
    header("GOAL 3: CC Decision Engine + SRT Output")

    step("Running category-aware fusion...")
    from src.fusion.category_mapper import CategoryMapper
    from src.fusion.decision_engine import DecisionEngine
    from src.output.label_mapper import map_label
    mapper = CategoryMapper("config/sound_categories.yaml")
    engine = DecisionEngine(config, mapper)

    all_copy = [e.copy() for e in events]
    accepted = engine.decide(events)
    for e in accepted:
        e["cc_text"] = map_label(e["label"])

    print(f"\n  {DIM}{'─' * 56}{RESET}")
    for e in all_copy:
        cat = mapper.get_category(e["label"])
        cc = map_label(e["label"])
        is_acc = any(a["id"] == e["id"] for a in accepted)
        icon = f"{GREEN}ACCEPT{RESET}" if is_acc else f"{RED}REJECT{RESET}"
        print(f"  #{e['id']:>3}  {cc:<18} [{cat['category']:<12}]  → {icon}")
    print(f"  {DIM}{'─' * 56}{RESET}")
    print(f"\n  {BOLD}{len(accepted)}{RESET} accepted / {len(all_copy)} total events")

    step("Writing SRT file...")
    output = os.path.splitext(video)[0] + "_cc.srt"
    from src.output.srt_writer import write_srt, write_summary
    write_srt(accepted, output)
    write_summary(accepted, all_copy, output)
    ok(f"SRT → {output}")

    if accepted:
        print(f"\n  {DIM}{'─' * 56}{RESET}")
        with open(output) as f:
            for line in f.read().strip().split('\n'):
                print(f"  {line}")
        print(f"  {DIM}{'─' * 56}{RESET}")
    else:
        print(f"\n  {DIM}(no events accepted — all filtered as ambient/low-confidence){RESET}")

    # Cleanup
    if not wav_preexisted and os.path.exists(wav_path):
        os.remove(wav_path)

    # ── SUMMARY ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════╗
║                    DEMO COMPLETE                          ║
╠══════════════════════════════════════════════════════════╣
║  Input:     {video:<44} ║
║  Duration:  {duration:<44.1f} ║
║  Events:    {len(all_copy)} detected → {len(accepted)} accepted{' ' * (33 - len(str(len(all_copy))) - len(str(len(accepted))))}║
║  Output:    {output:<44} ║
║  Time:      {elapsed:.1f}s ({elapsed/duration:.1f}x realtime){' ' * (37 - len(f'{elapsed:.1f}s ({elapsed/duration:.1f}x realtime)'))}║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

    # Web UI note
    print(f"  {BOLD}Web UI:{RESET} python web/app.py → http://localhost:8000\n")


if __name__ == "__main__":
    main()
