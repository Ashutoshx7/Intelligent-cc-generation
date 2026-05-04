"""Generate SRT subtitle files from accepted CC events."""
import logging

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Examples:
        0.0     -> "00:00:00,000"
        65.5    -> "00:01:05,500"
        3723.12 -> "01:02:03,120"
    """
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(events: list, output_path: str, encoding: str = "utf-8"):
    """
    Write accepted events to a standard SRT file.

    SRT format:
        1
        00:00:12,480 --> 00:00:13,440
        [gunshot]

        2
        00:00:28,320 --> 00:00:28,800
        [glass breaking]

    Args:
        events: List of accepted event dicts with start_time, end_time, cc_text.
        output_path: Where to write the .srt file.
        encoding: File encoding (default UTF-8).
    """
    # Sort by start time
    sorted_events = sorted(events, key=lambda e: e["start_time"])

    with open(output_path, 'w', encoding=encoding) as f:
        for i, event in enumerate(sorted_events, 1):
            start = format_timestamp(event["start_time"])
            end = format_timestamp(event["end_time"])
            text = event.get("cc_text", f"[{event.get('label', 'unknown')}]")

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n")
            f.write("\n")

    logger.info(f"Wrote {len(sorted_events)} CC entries to {output_path}")


def write_summary(events: list, all_events: list, output_path: str):
    """
    Write a human-readable summary alongside the SRT.
    Useful for editor review and debugging.
    """
    summary_path = output_path.replace('.srt', '_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("CC Suggestion Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total audio events detected: {len(all_events)}\n")
        f.write(f"Events accepted for CC:      {len(events)}\n")
        f.write(f"Events rejected:             {len(all_events) - len(events)}\n\n")

        if len(all_events) > 0:
            overcaption_avoided = len(all_events) - len(events)
            f.write(f"Overcaption prevention: {overcaption_avoided} ambient/insignificant "
                    f"sounds filtered out\n\n")

        # Accepted CCs
        f.write("ACCEPTED CCs:\n")
        f.write("-" * 60 + "\n")
        for e in sorted(events, key=lambda x: x["start_time"]):
            f.write(f"  {format_timestamp(e['start_time'])} -> "
                    f"{format_timestamp(e['end_time'])}  "
                    f"{e.get('cc_text', '?'):20s}  "
                    f"(audio={e['confidence']:.2f} "
                    f"visual={e.get('reaction_score', 0):.2f} "
                    f"combined={e.get('combined_score', 0):.2f} "
                    f"[{e.get('category', '?')}])\n")

        # Rejected events
        f.write(f"\nREJECTED EVENTS:\n")
        f.write("-" * 60 + "\n")
        rejected = [e for e in all_events if not e.get("accepted", False)]
        for e in sorted(rejected, key=lambda x: x["start_time"]):
            f.write(f"  {format_timestamp(e['start_time'])} -> "
                    f"{format_timestamp(e['end_time'])}  "
                    f"{e['label']:30s}  "
                    f"(audio={e['confidence']:.2f} "
                    f"visual={e.get('reaction_score', 0):.2f} "
                    f"combined={e.get('combined_score', 0):.2f} "
                    f"[{e.get('category', '?')}]) -> REJECTED\n")

    logger.info(f"Wrote summary to {summary_path}")
