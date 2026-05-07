"""Extract audio track from video as 16kHz mono WAV."""
import subprocess
import shutil
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """
    Extract audio from any video container to 16kHz mono WAV.

    Tries ffmpeg first. If ffmpeg is not installed, looks for a pre-existing
    WAV file alongside the video (e.g., video_audio.wav).

    Args:
        video_path: Path to input video (mp4, mkv, avi, mov, etc.)
        output_path: Where to write WAV. If None, auto-generates from video name.
        sample_rate: Target sample rate (16000 for YAMNet).

    Returns:
        Path to the output WAV file.

    Raises:
        FileNotFoundError: If video_path doesn't exist.
        RuntimeError: If extraction fails by all methods.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    base = os.path.splitext(video_path)[0]
    if output_path is None:
        output_path = f"{base}_audio.wav"

    # Check if WAV already exists (pre-extracted or from test data generator)
    if os.path.exists(output_path):
        logger.info(f"Using existing audio file: {output_path}")
        return output_path

    # Try ffmpeg
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # 16-bit PCM
            "-ar", str(sample_rate),  # resample to target rate
            "-ac", "1",               # mono
            "-y",                     # overwrite if exists
            output_path
        ]

        logger.info(f"Extracting audio with ffmpeg: {video_path} -> {output_path}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

        file_size = os.path.getsize(output_path)
        logger.info(f"Audio extracted: {output_path} ({file_size} bytes)")
        return output_path

    # No system ffmpeg — try moviepy (ships its own ffmpeg binary via imageio-ffmpeg)
    logger.info("System ffmpeg not found. Trying moviepy (bundled ffmpeg)...")
    try:
        from moviepy import VideoFileClip
        import scipy.io.wavfile as wavfile

        clip = VideoFileClip(video_path)
        if clip.audio is None:
            logger.warning("Video has no audio track — creating silent WAV")
            duration = clip.duration or 10.0
            clip.close()
            num_samples = int(duration * sample_rate)
            silence = np.zeros(num_samples, dtype=np.int16)
            wavfile.write(output_path, sample_rate, silence)
            return output_path

        # Extract audio to WAV via moviepy's bundled ffmpeg
        clip.audio.write_audiofile(
            output_path,
            fps=sample_rate,
            nbytes=2,        # 16-bit
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],  # mono
            logger=None,     # suppress moviepy progress bar
        )
        clip.close()

        file_size = os.path.getsize(output_path)
        logger.info(f"Audio extracted via moviepy: {output_path} ({file_size} bytes)")
        return output_path

    except ImportError:
        logger.warning("moviepy not installed — falling back to OpenCV silent WAV")
    except Exception as moviepy_err:
        logger.warning(f"moviepy extraction failed: {moviepy_err} — falling back to OpenCV silent WAV")

    # Last resort: OpenCV-based silent WAV (visual-only mode)
    logger.info("Creating silent WAV for visual-only mode...")
    try:
        import cv2
        import scipy.io.wavfile as wavfile

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 10.0
        cap.release()

        logger.warning(
            f"Creating silent WAV ({duration:.1f}s) — install ffmpeg or moviepy for real audio"
        )
        num_samples = int(duration * sample_rate)
        silence = np.zeros(num_samples, dtype=np.int16)
        wavfile.write(output_path, sample_rate, silence)
        logger.info(f"Silent WAV created: {output_path}")
        return output_path

    except Exception as fallback_err:
        raise RuntimeError(
            f"Cannot extract audio: no ffmpeg, moviepy failed, OpenCV failed ({fallback_err}). "
            f"Install ffmpeg ('apt install ffmpeg') or moviepy ('pip install moviepy')"
        )


def load_wav_as_float(wav_path: str) -> tuple:
    """
    Load WAV file and return (waveform, sample_rate).

    Waveform is float32 in [-1.0, 1.0] range — required by YAMNet.
    Handles int16 and float32 WAV formats.

    Returns:
        (waveform: np.ndarray[float32], sample_rate: int)
    """
    import scipy.io.wavfile as wavfile

    sample_rate, audio_data = wavfile.read(wav_path)

    # Convert to float32 [-1.0, 1.0]
    if audio_data.dtype == np.int16:
        waveform = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.float32:
        waveform = audio_data
    else:
        waveform = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

    # If stereo, take first channel
    if waveform.ndim > 1:
        waveform = waveform[:, 0]

    logger.info(f"Loaded WAV: {len(waveform)/sample_rate:.1f}s, {sample_rate}Hz, "
                f"range=[{waveform.min():.3f}, {waveform.max():.3f}]")
    return waveform, sample_rate
