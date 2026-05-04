#!/usr/bin/env python3
"""
Intelligent CC Suggestion Tool — CLI Entry Point

Generates intelligent closed caption suggestions for non-speech audio events
in video files, using audio-visual fusion to avoid over-captioning.

Usage:
    python main.py video.mp4
    python main.py video.mp4 -o captions.srt --verbose
    python main.py video.mp4 --threshold 0.35 --verbose
    python main.py video.mp4 --evaluate --ground-truth annotations.json
"""
import argparse
import logging
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate intelligent closed caption suggestions for video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py video.mp4
  python main.py video.mp4 -o captions.srt --verbose
  python main.py video.mp4 --threshold 0.35
  python main.py video.mp4 --evaluate --ground-truth eval/ground_truth/clip.json
        """
    )

    # Required
    parser.add_argument("video", help="Path to input video file (mp4, mkv, avi, mov, etc.)")

    # Output options
    parser.add_argument("-o", "--output", default=None,
                        help="Output SRT path (default: <video>_cc.srt)")

    # Config options
    parser.add_argument("-c", "--config", default="config/default.yaml",
                        help="Config YAML path (default: config/default.yaml)")
    parser.add_argument("--categories", default="config/sound_categories.yaml",
                        help="Sound categories YAML path")

    # Tuning options
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override fusion threshold for all categories")

    # Evaluation mode
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against ground truth annotations")
    parser.add_argument("--ground-truth", default=None,
                        help="Path to ground truth JSON for evaluation")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Default output path
    if args.output is None:
        base = os.path.splitext(args.video)[0]
        args.output = f"{base}_cc.srt"

    # Run pipeline
    from src.pipeline import run_pipeline
    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        config_path=args.config,
        categories_path=args.categories,
        threshold_override=args.threshold,
        verbose=args.verbose,
    )

    # Optional evaluation
    if args.evaluate:
        if not args.ground_truth:
            print("Error: --ground-truth path required for evaluation mode")
            sys.exit(1)

        from eval.evaluator import evaluate, load_ground_truth
        from src.output.srt_writer import format_timestamp

        gt = load_ground_truth(args.ground_truth)

        # Parse the generated SRT to get predicted events
        predicted = _parse_srt_events(args.output)

        results = evaluate(predicted, gt)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Precision:        {results['precision']:.3f}")
        print(f"  Recall:           {results['recall']:.3f}")
        print(f"  F1 Score:         {results['f1']:.3f}")
        print(f"  Overcaption Rate: {results['overcaption_rate']:.3f}")
        print(f"  TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}")
        print("=" * 50)


def _parse_srt_events(srt_path: str) -> list:
    """Parse an SRT file back into event dicts for evaluation."""
    events = []
    with open(srt_path, 'r') as f:
        lines = f.read().strip().split('\n')

    i = 0
    while i < len(lines):
        # Skip empty lines and sequence numbers
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Try to parse timestamp line
        if '-->' in line:
            parts = line.split('-->')
            start = _parse_srt_time(parts[0].strip())
            end = _parse_srt_time(parts[1].strip())

            # Next line is the text
            text = ""
            i += 1
            if i < len(lines) and lines[i].strip():
                text = lines[i].strip()

            events.append({
                "start_time": start,
                "end_time": end,
                "cc_text": text,
                "label": text.strip("[]"),
            })

        i += 1

    return events


def _parse_srt_time(time_str: str) -> float:
    """Parse SRT timestamp string to seconds."""
    # Format: HH:MM:SS,mmm
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    h = float(parts[0])
    m = float(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


if __name__ == "__main__":
    main()
