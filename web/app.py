"""FastAPI backend for the Intelligent CC Suggestion Tool."""
import os
import sys
import json
import uuid
import shutil
import logging
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent CC Suggestion Tool", version="1.0.0")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job storage
jobs = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI with auto-cache-busting."""
    import time
    html_path = STATIC_DIR / "index.html"
    html = html_path.read_text()
    # Auto cache-bust: replace ?v=X with current timestamp
    cache_buster = str(int(time.time()))
    html = html.replace('style.css?v=4', f'style.css?v={cache_buster}')
    html = html.replace('app.js?v=4', f'app.js?v={cache_buster}')
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    )


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file and return a job ID."""
    # Validate file type
    allowed_ext = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {ext}. Use: {allowed_ext}")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save file
    video_path = job_dir / f"input{ext}"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Also save a copy for serving to the video player
    serve_path = job_dir / f"video{ext}"
    shutil.copy(str(video_path), str(serve_path))

    jobs[job_id] = {
        "id": job_id,
        "status": "uploaded",
        "filename": file.filename,
        "video_path": str(video_path),
        "serve_path": str(serve_path),
        "ext": ext,
        "events": [],
        "accepted": [],
        "progress": 0,
        "stage": "",
    }

    logger.info(f"Job {job_id}: uploaded {file.filename} ({len(content)} bytes)")
    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/process/{job_id}")
async def process_video(job_id: str):
    """Start processing a video (runs pipeline in background)."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        return {"status": "already processing"}

    job["status"] = "processing"
    job["progress"] = 0
    job["stage"] = "Starting..."

    # Run pipeline in background thread
    asyncio.create_task(_run_pipeline_async(job_id))

    return {"status": "processing", "job_id": job_id}


async def _run_pipeline_async(job_id: str):
    """Run the CC pipeline asynchronously."""
    job = jobs[job_id]
    video_path = job["video_path"]

    try:
        # Import pipeline components
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_pipeline_sync, job_id, video_path)

        job["events"] = result["all_events"]
        job["accepted"] = result["accepted"]
        job["status"] = "complete"
        job["progress"] = 100
        job["stage"] = "Complete"
        logger.info(f"Job {job_id}: pipeline complete. {len(result['accepted'])} CCs accepted")

    except Exception as e:
        logger.error(f"Job {job_id}: pipeline failed: {e}", exc_info=True)
        job["status"] = "error"
        job["stage"] = str(e)


def _run_pipeline_sync(job_id: str, video_path: str) -> dict:
    """Synchronous pipeline execution (runs in thread pool)."""
    from src.config_loader import load_config
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

    config_path = os.path.join(PROJECT_ROOT, "config", "default.yaml")
    categories_path = os.path.join(PROJECT_ROOT, "config", "sound_categories.yaml")
    config = load_config(config_path)

    job = jobs[job_id]

    # Goal 1: Audio
    job["stage"] = "Extracting audio..."
    job["progress"] = 5
    wav_path = extract_audio(video_path, sample_rate=config['audio']['sample_rate'])
    waveform, sr = load_wav_as_float(wav_path)

    job["stage"] = "Detecting speech segments..."
    job["progress"] = 15
    sf = SpeechFilter(aggressiveness=config['audio']['vad_aggressiveness'], sample_rate=sr)
    speech_segments = sf.get_speech_segments(waveform)

    job["stage"] = "Running YAMNet sound detection..."
    job["progress"] = 25
    detector = YAMNetDetector(config)
    events = detector.detect(waveform)
    events = [e for e in events if not sf.is_during_speech(e["start_time"], e["end_time"], speech_segments)]

    job["stage"] = f"Detected {len(events)} non-speech events"
    job["progress"] = 40

    if not events:
        return {"all_events": [], "accepted": []}

    # Goal 2: Visual
    job["stage"] = "Detecting scene cuts..."
    job["progress"] = 45
    cut_detector = SceneCutDetector(config['visual']['scene_cut_threshold'])
    scene_cuts = cut_detector.detect_cuts(video_path)

    job["stage"] = "Analyzing visual reactions..."
    job["progress"] = 50
    fe = FrameExtractor(config)
    pa = PoseAnalyzer(config)
    fa = FaceAnalyzer(config)

    total = len(events)
    for i, event in enumerate(events):
        job["progress"] = 50 + int(30 * (i / total))
        on_cut = cut_detector.is_on_scene_cut(event["start_time"], scene_cuts, config['visual']['scene_cut_tolerance'])
        event["on_scene_cut"] = on_cut

        if on_cut:
            event["reaction_score"] = 0.0
            event["reaction_persons"] = 0
        else:
            frames = fe.extract_reaction_frames(video_path, event["start_time"])
            if not frames:
                event["reaction_score"] = 0.0
                event["reaction_persons"] = 0
            else:
                scores = []
                max_p = 0
                for ts, frame in frames:
                    pr = pa.analyze(frame)
                    fr = fa.analyze(frame)
                    scores.append(max(pr["pose_score"], fr["face_score"]))
                    max_p = max(max_p, pr["num_persons"], fr["num_faces"])
                event["reaction_score"] = max(scores) if scores else 0.0
                event["reaction_persons"] = max_p

        event["speech_paused"] = sf.was_speech_before(event["start_time"], speech_segments)

    pa.close()
    fa.close()

    # Goal 3: Decision
    job["stage"] = "Running decision engine..."
    job["progress"] = 85
    mapper = CategoryMapper(categories_path)
    engine = DecisionEngine(config, mapper)

    accepted = engine.decide(events)
    for e in events:
        cat = mapper.get_category(e["label"])
        e["category"] = cat["category"]
        e["cc_text"] = map_label(e["label"])
        e["combined_score"] = e.get("combined_score", 0.0)
        e["accepted"] = e.get("accepted", False)
        
    all_events = events

    job["progress"] = 95
    job["stage"] = "Generating output..."

    # Clean up temp WAV if it was created by ffmpeg
    if wav_path.endswith("_audio.wav") and os.path.exists(wav_path):
        base_check = os.path.splitext(video_path)[0] + "_audio.wav"
        if wav_path == base_check:
            pass  # keep pre-existing

    return {"all_events": all_events, "accepted": accepted}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "num_events": len(job.get("events", [])),
        "num_accepted": len(job.get("accepted", [])),
    }


@app.get("/api/events/{job_id}")
async def get_events(job_id: str):
    """Get all detected events with scores."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "complete":
        raise HTTPException(400, "Processing not complete")

    # Serialize events (strip numpy types)
    serializable = []
    for e in job["events"]:
        serializable.append({
            "id": e.get("id"),
            "label": e.get("label", ""),
            "cc_text": e.get("cc_text", ""),
            "confidence": round(float(e.get("confidence", 0)), 3),
            "start_time": round(float(e.get("start_time", 0)), 3),
            "end_time": round(float(e.get("end_time", 0)), 3),
            "reaction_score": round(float(e.get("reaction_score", 0)), 3),
            "combined_score": round(float(e.get("combined_score", 0)), 3),
            "category": e.get("category", "default"),
            "accepted": e.get("accepted", False),
            "on_scene_cut": e.get("on_scene_cut", False),
            "speech_paused": e.get("speech_paused", False),
        })

    return {"events": serializable}


@app.post("/api/toggle/{job_id}/{event_id}")
async def toggle_event(job_id: str, event_id: int):
    """Toggle accept/reject for a specific event."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    for e in job["events"]:
        if e.get("id") == event_id:
            e["accepted"] = not e.get("accepted", False)
            return {"id": event_id, "accepted": e["accepted"]}
    raise HTTPException(404, "Event not found")


@app.get("/api/export/{job_id}")
async def export_srt(job_id: str):
    """Export accepted events as SRT file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]

    accepted = [e for e in job["events"] if e.get("accepted", False)]
    accepted.sort(key=lambda e: e["start_time"])

    srt_lines = []
    for i, e in enumerate(accepted, 1):
        start = _fmt_ts(e["start_time"])
        end = _fmt_ts(e["end_time"])
        text = e.get("cc_text", f"[{e.get('label', 'unknown')}]")
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    srt_content = "\n".join(srt_lines)

    # Save to file
    srt_path = UPLOAD_DIR / job_id / "output.srt"
    srt_path.write_text(srt_content)

    return FileResponse(str(srt_path), filename=f"{job['filename']}_cc.srt",
                       media_type="text/plain")


@app.get("/api/video/{job_id}")
async def serve_video(job_id: str):
    """Serve the uploaded video for playback."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return FileResponse(job["serve_path"],
                       media_type=f"video/{job['ext'].strip('.')}")


def _fmt_ts(seconds: float) -> str:
    """Format seconds to SRT timestamp."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
