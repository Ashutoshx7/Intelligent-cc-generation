#!/usr/bin/env python3
"""
Generate a synthetic test video with embedded audio events for pipeline testing.
Creates a 10-second video with visual content and a separate WAV audio file.
No ffmpeg needed — uses OpenCV for video and scipy for audio.
"""
import numpy as np
import cv2
import scipy.io.wavfile as wavfile
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video parameters
FPS = 24
DURATION = 10  # seconds
WIDTH, HEIGHT = 640, 480
TOTAL_FRAMES = FPS * DURATION

# Audio parameters
SAMPLE_RATE = 16000
TOTAL_SAMPLES = SAMPLE_RATE * DURATION


def generate_test_video():
    """Generate a test .avi video with visual changes simulating reactions."""
    video_path = os.path.join(OUTPUT_DIR, "test_clip.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / FPS

        # Base frame: dark blue background
        frame = np.full((HEIGHT, WIDTH, 3), (40, 30, 20), dtype=np.uint8)

        # Draw a "person" (simple circle for head + rectangle for body)
        center_x = WIDTH // 2
        center_y = HEIGHT // 2 - 50

        # Simulate head turn at t=3s (reaction to "event 1")
        head_offset_x = 0
        if 3.3 < t < 4.0:
            head_offset_x = int(30 * np.sin((t - 3.3) * np.pi / 0.7))

        # Draw body
        cv2.rectangle(frame, (center_x - 40, center_y + 30),
                      (center_x + 40, center_y + 120), (100, 100, 200), -1)

        # Draw head
        cv2.circle(frame, (center_x + head_offset_x, center_y),
                   35, (150, 150, 220), -1)

        # Draw eyes
        cv2.circle(frame, (center_x + head_offset_x - 12, center_y - 5),
                   5, (30, 30, 30), -1)
        cv2.circle(frame, (center_x + head_offset_x + 12, center_y - 5),
                   5, (30, 30, 30), -1)

        # Simulate "surprise" mouth open at t=7s (reaction to "event 2")
        mouth_height = 3
        if 7.3 < t < 8.0:
            mouth_height = int(15 * np.sin((t - 7.3) * np.pi / 0.7))

        cv2.ellipse(frame, (center_x + head_offset_x, center_y + 15),
                    (8, mouth_height), 0, 0, 360, (50, 50, 50), -1)

        # Scene cut at t=5s (abrupt color change)
        if 5.0 < t < 5.5:
            frame = np.full((HEIGHT, WIDTH, 3), (200, 180, 50), dtype=np.uint8)
            cv2.putText(frame, "SCENE CUT", (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Timestamp overlay
        cv2.putText(frame, f"t={t:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"Video written: {video_path} ({TOTAL_FRAMES} frames, {DURATION}s)")
    return video_path


def generate_test_audio():
    """
    Generate a test WAV with synthetic audio events:
    - t=0-2s: silence
    - t=2-3s: speech-like signal (300Hz tone)
    - t=3s: sharp transient (simulates gunshot/impact)
    - t=3.5-4.5s: silence
    - t=5s: another transient (during scene cut)
    - t=6.5-7s: speech-like
    - t=7s: loud burst (simulates explosion)
    - t=8-10s: quiet ambient noise
    """
    audio = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    t = np.arange(TOTAL_SAMPLES) / SAMPLE_RATE

    # Background noise floor
    audio += np.random.randn(TOTAL_SAMPLES).astype(np.float32) * 0.005

    # Speech-like signal (t=2-3s) — 300Hz tone with harmonics
    mask = (t >= 2.0) & (t < 3.0)
    audio[mask] += 0.3 * np.sin(2 * np.pi * 300 * t[mask]).astype(np.float32)
    audio[mask] += 0.15 * np.sin(2 * np.pi * 600 * t[mask]).astype(np.float32)

    # Event 1: Sharp transient at t=3.0s (impact/gunshot)
    event1_start = int(3.0 * SAMPLE_RATE)
    event1_end = int(3.3 * SAMPLE_RATE)
    event1_t = np.arange(event1_end - event1_start) / SAMPLE_RATE
    audio[event1_start:event1_end] += (0.8 * np.exp(-event1_t * 10) *
                                        np.sin(2 * np.pi * 2000 * event1_t)).astype(np.float32)

    # Event 2: Transient at t=5.0s (during scene cut)
    event2_start = int(5.0 * SAMPLE_RATE)
    event2_end = int(5.2 * SAMPLE_RATE)
    event2_t = np.arange(event2_end - event2_start) / SAMPLE_RATE
    audio[event2_start:event2_end] += (0.6 * np.exp(-event2_t * 8) *
                                        np.sin(2 * np.pi * 1500 * event2_t)).astype(np.float32)

    # Speech-like signal (t=6.5-7s)
    mask = (t >= 6.5) & (t < 7.0)
    audio[mask] += 0.25 * np.sin(2 * np.pi * 280 * t[mask]).astype(np.float32)

    # Event 3: Loud burst at t=7.0s (explosion-like)
    event3_start = int(7.0 * SAMPLE_RATE)
    event3_end = int(7.5 * SAMPLE_RATE)
    event3_t = np.arange(event3_end - event3_start) / SAMPLE_RATE
    audio[event3_start:event3_end] += (0.9 * np.exp(-event3_t * 5) *
                                        np.random.randn(event3_end - event3_start).astype(np.float32) * 0.5)
    audio[event3_start:event3_end] += (0.7 * np.exp(-event3_t * 4) *
                                        np.sin(2 * np.pi * 100 * event3_t)).astype(np.float32)

    # Clip to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    # Save as 16-bit WAV
    wav_path = os.path.join(OUTPUT_DIR, "test_clip_audio.wav")
    int16_audio = (audio * 32767).astype(np.int16)
    wavfile.write(wav_path, SAMPLE_RATE, int16_audio)
    print(f"Audio written: {wav_path} ({DURATION}s, {SAMPLE_RATE}Hz)")
    return wav_path


if __name__ == "__main__":
    video_path = generate_test_video()
    audio_path = generate_test_audio()
    print(f"\nTest files ready:")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    print(f"\nRun pipeline: python3 main.py {video_path} --verbose")
