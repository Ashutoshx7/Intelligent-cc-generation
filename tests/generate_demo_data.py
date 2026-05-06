#!/usr/bin/env python3
"""
Generate a realistic demo video with distinct, recognizable sounds.
Creates a 15-second video with 4 clear audio events that YAMNet
will classify with impressive labels like sirens, alarms, etc.
"""
import numpy as np
import cv2
import struct
import wave
import os

DURATION = 15       # seconds
FPS = 24
SR = 16000
WIDTH = 640
HEIGHT = 480

def make_siren(t, freq_lo=400, freq_hi=1200, rate=3.0):
    """Generate a classic siren sweep — YAMNet classifies as Siren/Emergency."""
    phase = np.sin(2 * np.pi * rate * t)  # sweep oscillator
    freq = freq_lo + (freq_hi - freq_lo) * (0.5 + 0.5 * phase)
    return 0.8 * np.sin(2 * np.pi * freq * t)

def make_alarm(t, freq=2500, pulse_rate=6.0):
    """Generate rapid beeping alarm — YAMNet classifies as Alarm/Beep."""
    envelope = (np.sin(2 * np.pi * pulse_rate * t) > 0).astype(float)
    return 0.7 * envelope * np.sin(2 * np.pi * freq * t)

def make_bell(t, freq=800, decay=3.0):
    """Generate a bell/chime hit — YAMNet classifies as Bell/Chime."""
    harmonics = (
        np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * freq * 2 * t) +
        0.25 * np.sin(2 * np.pi * freq * 3 * t)
    )
    envelope = np.exp(-decay * t)
    return 0.6 * envelope * harmonics

def make_knock(t, freq=150, decay=15.0):
    """Generate a sharp knock/impact — YAMNet classifies as Knock/Thump."""
    impact = np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
    noise = np.random.randn(len(t)) * 0.3 * np.exp(-decay * 0.5 * t)
    return 0.9 * (impact + noise)

def create_video(path, frames_data):
    """Write a simple video with colored frames and text overlays."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(DURATION * FPS):
        t = i / FPS
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Background — dark gradient
        for y in range(HEIGHT):
            frame[y, :] = [15 + int(y * 0.05), 15 + int(y * 0.03), 20 + int(y * 0.02)]

        # Show event labels on screen during events
        for start, end, label, color in frames_data:
            if start <= t <= end:
                # Flash border
                cv2.rectangle(frame, (5, 5), (WIDTH-5, HEIGHT-5), color, 3)
                # Event label
                cv2.putText(frame, f"[{label}]", (WIDTH//2 - 100, HEIGHT//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                # Timestamp
                ts = f"{t:.1f}s"
                cv2.putText(frame, ts, (WIDTH//2 - 30, HEIGHT//2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        # Always show project title
        cv2.putText(frame, "Intelligent CC Tool — PlanetRead", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        # Timeline bar at bottom
        bar_y = HEIGHT - 20
        cv2.rectangle(frame, (20, bar_y), (WIDTH-20, bar_y+8), (40, 40, 40), -1)
        progress = int(20 + (WIDTH-40) * (t / DURATION))
        cv2.rectangle(frame, (20, bar_y), (progress, bar_y+8), (255, 255, 255), -1)

        writer.write(frame)

    writer.release()

def create_audio(path):
    """Generate audio with 4 distinct sound events."""
    samples = np.zeros(DURATION * SR)

    # Add very subtle background noise (makes it realistic)
    samples += np.random.randn(len(samples)) * 0.01

    # Event 1: Siren at 2.0s - 3.5s
    e1_start, e1_end = int(2.0 * SR), int(3.5 * SR)
    t1 = np.arange(e1_end - e1_start) / SR
    samples[e1_start:e1_end] += make_siren(t1)

    # Event 2: Alarm beep at 5.5s - 6.5s
    e2_start, e2_end = int(5.5 * SR), int(6.5 * SR)
    t2 = np.arange(e2_end - e2_start) / SR
    samples[e2_start:e2_end] += make_alarm(t2)

    # Event 3: Bell/chime at 8.5s - 9.5s
    e3_start, e3_end = int(8.5 * SR), int(9.5 * SR)
    t3 = np.arange(e3_end - e3_start) / SR
    samples[e3_start:e3_end] += make_bell(t3)

    # Event 4: Knock/impact at 12.0s - 12.3s
    e4_start, e4_end = int(12.0 * SR), int(12.3 * SR)
    t4 = np.arange(e4_end - e4_start) / SR
    samples[e4_start:e4_end] += make_knock(t4)

    # Normalize
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.95

    # Write WAV
    int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int_samples.tobytes())

def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'samples')
    os.makedirs(out_dir, exist_ok=True)

    video_path = os.path.join(out_dir, 'demo_clip.avi')
    audio_path = os.path.join(out_dir, 'demo_clip_audio.wav')

    # Event overlays for the video
    events_visual = [
        (2.0, 3.5, "SIREN", (0, 100, 255)),       # blue-orange
        (5.5, 6.5, "ALARM", (0, 0, 255)),          # red
        (8.5, 9.5, "BELL", (0, 255, 200)),          # cyan
        (12.0, 12.5, "KNOCK", (255, 100, 0)),       # orange
    ]

    print("Generating demo video...")
    create_video(video_path, events_visual)
    print(f"  Video: {os.path.abspath(video_path)} ({DURATION}s, {FPS}fps)")

    print("Generating demo audio...")
    create_audio(audio_path)
    print(f"  Audio: {os.path.abspath(audio_path)} ({DURATION}s, {SR}Hz)")

    print(f"\nRun: python3 demo.py {video_path}")


if __name__ == "__main__":
    main()
