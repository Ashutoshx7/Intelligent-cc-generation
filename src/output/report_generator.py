"""Generate professional HTML report and JSON output for CC suggestions."""
import json
import logging
import os
from datetime import datetime

from src.output.srt_writer import format_timestamp

logger = logging.getLogger(__name__)


def write_json_report(events: list, all_events: list, output_path: str,
                      video_path: str = "", duration: float = 0.0):
    """
    Write machine-readable JSON report with full event data.

    Useful for:
    - Integration with downstream subtitle editors
    - Automated evaluation pipelines
    - Editor review dashboards
    """
    json_path = output_path.replace('.srt', '_report.json')

    accepted_ids = {e["id"] for e in events}
    report = {
        "tool": "Intelligent CC Suggestion Tool",
        "version": "1.0.0",
        "generated": datetime.now().isoformat(),
        "video": video_path,
        "duration_seconds": round(duration, 2),
        "summary": {
            "total_detected": len(all_events),
            "accepted": len(events),
            "rejected": len(all_events) - len(events),
            "filter_rate": round(1 - len(events) / max(len(all_events), 1), 3),
        },
        "accepted_events": [
            {
                "id": e["id"],
                "label": e.get("label", ""),
                "cc_text": e.get("cc_text", ""),
                "start_time": round(e["start_time"], 3),
                "end_time": round(e["end_time"], 3),
                "start_srt": format_timestamp(e["start_time"]),
                "end_srt": format_timestamp(e["end_time"]),
                "audio_confidence": round(e.get("confidence", 0), 4),
                "visual_reaction": round(e.get("reaction_score", 0), 4),
                "combined_score": round(e.get("combined_score", 0), 4),
                "category": e.get("category", "default"),
                "on_scene_cut": e.get("on_scene_cut", False),
                "speech_paused": e.get("speech_paused", False),
            }
            for e in sorted(events, key=lambda x: x["start_time"])
        ],
        "rejected_events": [
            {
                "id": e["id"],
                "label": e.get("label", ""),
                "start_time": round(e["start_time"], 3),
                "end_time": round(e["end_time"], 3),
                "audio_confidence": round(e.get("confidence", 0), 4),
                "visual_reaction": round(e.get("reaction_score", 0), 4),
                "combined_score": round(e.get("combined_score", 0), 4),
                "category": e.get("category", "default"),
                "reason": _get_reject_reason(e),
            }
            for e in sorted(all_events, key=lambda x: x["start_time"])
            if e["id"] not in accepted_ids
        ],
    }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Wrote JSON report to {json_path}")
    return json_path


def _get_reject_reason(event: dict) -> str:
    """Generate human-readable rejection reason."""
    cat = event.get("category", "default")
    conf = event.get("confidence", 0)
    react = event.get("reaction_score", 0)
    combined = event.get("combined_score", 0)

    if cat == "ambient":
        return f"Ambient sound ({cat}) — below threshold even with visual"
    elif react < 0.1 and conf < 0.5:
        return f"Low confidence ({conf:.2f}) and no visual reaction"
    elif react < 0.1:
        return f"No visible reaction to support audio signal ({conf:.2f})"
    else:
        return f"Combined score {combined:.2f} below category threshold"


def write_html_report(events: list, all_events: list, output_path: str,
                      video_path: str = "", duration: float = 0.0):
    """
    Generate a professional HTML report for editor review.

    Features:
    - Stats overview with accept/reject/filter rate
    - Interactive event table with color-coded decisions
    - Category distribution chart (CSS-only, no JS dependencies)
    - SRT preview
    """
    html_path = output_path.replace('.srt', '_report.html')
    accepted_ids = {e["id"] for e in events}
    filename = os.path.basename(video_path)

    # Compute category stats
    cat_counts = {}
    for e in all_events:
        cat = e.get("category", "default")
        if cat not in cat_counts:
            cat_counts[cat] = {"total": 0, "accepted": 0}
        cat_counts[cat]["total"] += 1
        if e["id"] in accepted_ids:
            cat_counts[cat]["accepted"] += 1

    # Build event rows
    event_rows = ""
    for e in sorted(all_events, key=lambda x: x["start_time"]):
        is_acc = e["id"] in accepted_ids
        status_class = "accepted" if is_acc else "rejected"
        status_text = "✓ ACCEPT" if is_acc else "✗ REJECT"
        cc = e.get("cc_text", f"[{e.get('label', '?')}]")
        flags = []
        if e.get("on_scene_cut"): flags.append("⚡ cut")
        if e.get("speech_paused"): flags.append("🗣 pause")

        event_rows += f"""
        <tr class="{status_class}">
            <td>{format_timestamp(e['start_time'])}</td>
            <td>{format_timestamp(e['end_time'])}</td>
            <td><strong>{cc}</strong><br><small>{e.get('label', '')}</small></td>
            <td>{e.get('category', 'default')}</td>
            <td>{e.get('confidence', 0):.2f}</td>
            <td>{e.get('reaction_score', 0):.2f}</td>
            <td>{e.get('combined_score', 0):.2f}</td>
            <td>{' '.join(flags)}</td>
            <td class="status-{status_class}">{status_text}</td>
        </tr>"""

    # Build category bars
    cat_bars = ""
    max_total = max((c["total"] for c in cat_counts.values()), default=1)
    for cat, counts in sorted(cat_counts.items()):
        pct = int(counts["total"] / max_total * 100)
        acc_pct = int(counts["accepted"] / max(counts["total"], 1) * 100)
        cat_bars += f"""
        <div class="cat-row">
            <span class="cat-name">{cat}</span>
            <div class="cat-bar-bg">
                <div class="cat-bar" style="width: {pct}%">{counts['total']} detected</div>
            </div>
            <span class="cat-rate">{acc_pct}% accepted</span>
        </div>"""

    # SRT preview
    srt_lines = ""
    for i, e in enumerate(sorted(events, key=lambda x: x["start_time"]), 1):
        cc = e.get("cc_text", f"[{e.get('label', '?')}]")
        srt_lines += f"{i}\n{format_timestamp(e['start_time'])} --> {format_timestamp(e['end_time'])}\n{cc}\n\n"

    filter_rate = round((1 - len(events) / max(len(all_events), 1)) * 100)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CC Report — {filename}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 40px; line-height: 1.6; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ font-size: 24px; font-weight: 300; margin-bottom: 8px; letter-spacing: -0.5px; }}
        .subtitle {{ color: #666; font-size: 13px; margin-bottom: 40px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #222; border: 1px solid #222; margin-bottom: 40px; }}
        .stat {{ background: #0a0a0a; padding: 24px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: 200; font-family: 'JetBrains Mono', monospace; }}
        .stat-value.green {{ color: #4ade80; }}
        .stat-value.red {{ color: #f87171; }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #666; margin-top: 4px; }}
        h2 {{ font-size: 16px; font-weight: 400; margin: 32px 0 16px; text-transform: uppercase; letter-spacing: 1px; color: #888; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 12px 16px; border-bottom: 1px solid #333; color: #888; font-weight: 400; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }}
        td {{ padding: 10px 16px; border-bottom: 1px solid #1a1a1a; }}
        tr.accepted {{ background: rgba(74, 222, 128, 0.03); }}
        tr.rejected {{ opacity: 0.5; }}
        .status-accepted {{ color: #4ade80; font-weight: 600; }}
        .status-rejected {{ color: #666; }}
        .cat-row {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .cat-name {{ width: 100px; font-size: 12px; color: #888; }}
        .cat-bar-bg {{ flex: 1; height: 24px; background: #1a1a1a; border-radius: 4px; overflow: hidden; margin: 0 12px; }}
        .cat-bar {{ height: 100%; background: #333; display: flex; align-items: center; padding-left: 8px; font-size: 11px; color: #aaa; border-radius: 4px; }}
        .cat-rate {{ font-size: 12px; color: #666; width: 100px; text-align: right; }}
        .srt-preview {{ background: #111; border: 1px solid #222; padding: 20px; font-family: 'JetBrains Mono', monospace; font-size: 13px; white-space: pre; overflow-x: auto; line-height: 1.8; }}
        .footer {{ margin-top: 60px; padding-top: 20px; border-top: 1px solid #1a1a1a; font-size: 11px; color: #444; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CC Suggestion Report</h1>
        <p class="subtitle">{filename} · {duration:.1f}s · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class="stats">
            <div class="stat"><div class="stat-value">{len(all_events)}</div><div class="stat-label">Detected</div></div>
            <div class="stat"><div class="stat-value green">{len(events)}</div><div class="stat-label">Accepted</div></div>
            <div class="stat"><div class="stat-value red">{len(all_events) - len(events)}</div><div class="stat-label">Filtered</div></div>
            <div class="stat"><div class="stat-value">{filter_rate}%</div><div class="stat-label">Filter Rate</div></div>
        </div>

        <h2>Category Distribution</h2>
        {cat_bars}

        <h2>Event Details</h2>
        <table>
            <thead>
                <tr><th>Start</th><th>End</th><th>Label</th><th>Category</th><th>Audio</th><th>Visual</th><th>Combined</th><th>Flags</th><th>Decision</th></tr>
            </thead>
            <tbody>
                {event_rows}
            </tbody>
        </table>

        <h2>SRT Preview</h2>
        <div class="srt-preview">{srt_lines if srt_lines else 'No accepted events.'}</div>

        <div class="footer">
            Intelligent CC Suggestion Tool · DMP 2026 · PlanetRead · C4GT
        </div>
    </div>
</body>
</html>"""

    with open(html_path, 'w') as f:
        f.write(html)

    logger.info(f"Wrote HTML report to {html_path}")
    return html_path
