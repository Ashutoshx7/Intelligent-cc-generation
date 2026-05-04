"""Main pipeline: chains Goal 1 (Audio) -> Goal 2 (Visual) -> Goal 3 (Decision + Output)."""
import os
import logging
import time

from src.config_loader import load_config, load_sound_categories
from src.audio.extractor import extract_audio, load_wav_as_float
from src.audio.speech_filter import SpeechFilter
from src.audio.yamnet_detector import YAMNetDetector
from src.visual.scene_cut import SceneCutDetector
from src.visual.frame_extractor import FrameExtractor
from src.visual.pose_analyzer import PoseAnalyzer
from src.visual.face_analyzer import FaceAnalyzer
from src.fusion.category_mapper import CategoryMapper
from src.fusion.decision_engine import DecisionEngine
from src.output.label_mapper import map_label
from src.output.srt_writer import write_srt, write_summary

logger = logging.getLogger(__name__)


def run_pipeline(video_path: str, output_path: str,
                 config_path: str = "config/default.yaml",
                 categories_path: str = "config/sound_categories.yaml",
                 threshold_override: float = None,
                 verbose: bool = False):
    """
    Full CC suggestion pipeline.

    Stages:
    1. Extract audio, run speech filter + YAMNet detection (Goal 1)
    2. Detect scene cuts, extract reaction frames, run pose + face analysis (Goal 2)
    3. Category-aware fusion, label mapping, SRT output (Goal 3)

    Args:
        video_path: Input video file (mp4, mkv, avi, mov, etc.).
        output_path: Output SRT file path.
        config_path: Path to default.yaml.
        categories_path: Path to sound_categories.yaml.
        threshold_override: Override all category thresholds (optional).
        verbose: Enable debug logging.
    """
    t_start = time.time()
    config = load_config(config_path)

    if threshold_override is not None:
        config['fusion']['threshold'] = threshold_override

    # ========================================================
    # GOAL 1: Sound Event Detection
    # ========================================================
    logger.info("=" * 60)
    logger.info("GOAL 1: Sound Event Detection")
    logger.info("=" * 60)

    # Step 1.1: Extract audio
    wav_path = extract_audio(video_path, sample_rate=config['audio']['sample_rate'])
    waveform, sr = load_wav_as_float(wav_path)
    audio_duration = len(waveform) / sr
    logger.info(f"Audio: {audio_duration:.1f}s at {sr}Hz")

    # Step 1.2: Speech filtering (VAD)
    speech_filter = SpeechFilter(
        aggressiveness=config['audio']['vad_aggressiveness'],
        sample_rate=sr
    )
    speech_segments = speech_filter.get_speech_segments(waveform)

    # Step 1.3: YAMNet sound event detection
    detector = YAMNetDetector(config)
    events = detector.detect(waveform)

    # Step 1.4: Remove events that overlap heavily with speech
    pre_filter_count = len(events)
    events = [e for e in events
              if not speech_filter.is_during_speech(
                  e["start_time"], e["end_time"], speech_segments)]
    if pre_filter_count != len(events):
        logger.info(f"Speech overlap filter removed {pre_filter_count - len(events)} events")

    logger.info(f"Goal 1 complete: {len(events)} non-speech events detected")

    if not events:
        logger.info("No non-speech events detected. Writing empty SRT.")
        write_srt([], output_path)
        elapsed = time.time() - t_start
        logger.info(f"Pipeline complete in {elapsed:.1f}s (no events)")
        return

    # ========================================================
    # GOAL 2: Visual Reaction Detection
    # ========================================================
    logger.info("=" * 60)
    logger.info("GOAL 2: Visual Reaction Detection")
    logger.info("=" * 60)

    # Step 2.1: Detect scene cuts (one pass over entire video)
    cut_detector = SceneCutDetector(config['visual']['scene_cut_threshold'])
    scene_cuts = cut_detector.detect_cuts(video_path)

    # Step 2.2: Score visual reactions for each audio event
    frame_extractor = FrameExtractor(config)
    pose_analyzer = PoseAnalyzer(config)
    face_analyzer = FaceAnalyzer(config)

    cut_tolerance = config['visual']['scene_cut_tolerance']

    for event in events:
        # Check if event is on a scene cut
        on_cut = cut_detector.is_on_scene_cut(event["start_time"], scene_cuts, cut_tolerance)
        event["on_scene_cut"] = on_cut

        if on_cut:
            # Scene cut makes visual analysis unreliable — skip it
            event["reaction_score"] = 0.0
            event["reaction_persons"] = 0
            logger.debug(f"Event #{event['id']} '{event['label']}' on scene cut — skipping visual")
        else:
            # Extract frames in reaction window (300ms - 1500ms after event onset)
            frames = frame_extractor.extract_reaction_frames(video_path, event["start_time"])

            if not frames:
                event["reaction_score"] = 0.0
                event["reaction_persons"] = 0
            else:
                # Score each frame, take PEAK (reactions are spiky, not sustained)
                frame_scores = []
                max_persons = 0

                for ts, frame in frames:
                    pose_result = pose_analyzer.analyze(frame)
                    face_result = face_analyzer.analyze(frame)

                    # Use max of pose and face (either signal is valid)
                    frame_score = max(pose_result["pose_score"], face_result["face_score"])
                    frame_scores.append(frame_score)

                    max_persons = max(max_persons,
                                     pose_result["num_persons"],
                                     face_result["num_faces"])

                # Peak reaction across temporal window
                event["reaction_score"] = max(frame_scores) if frame_scores else 0.0
                event["reaction_persons"] = max_persons

        # Check if speech paused just before this event
        event["speech_paused"] = speech_filter.was_speech_before(
            event["start_time"], speech_segments
        )

    # Cleanup visual models
    pose_analyzer.close()
    face_analyzer.close()

    logger.info(f"Goal 2 complete: reaction scores computed for {len(events)} events")

    # ========================================================
    # GOAL 3: CC Decision Engine + SRT Output
    # ========================================================
    logger.info("=" * 60)
    logger.info("GOAL 3: CC Decision Engine + Output")
    logger.info("=" * 60)

    # Step 3.1: Category-aware fusion
    category_mapper = CategoryMapper(categories_path)
    engine = DecisionEngine(config, category_mapper)

    all_events_copy = []
    for e in events:
        all_events_copy.append(e.copy())

    accepted = engine.decide(events)

    # Step 3.2: Map labels to CC text
    for event in accepted:
        event["cc_text"] = map_label(event["label"])

    # Step 3.3: Write SRT output
    write_srt(accepted, output_path, config['output']['encoding'])
    write_summary(accepted, all_events_copy, output_path)

    # Cleanup temp audio file
    if os.path.exists(wav_path) and wav_path.endswith("_audio.wav"):
        os.remove(wav_path)
        logger.debug(f"Cleaned up temp audio: {wav_path}")

    # Final summary
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Video:     {video_path}")
    logger.info(f"  Duration:  {audio_duration:.1f}s")
    logger.info(f"  Events:    {len(all_events_copy)} detected -> {len(accepted)} accepted")
    logger.info(f"  Output:    {output_path}")
    logger.info(f"  Time:      {elapsed:.1f}s ({elapsed/audio_duration:.1f}x realtime)")
    logger.info("=" * 60)
