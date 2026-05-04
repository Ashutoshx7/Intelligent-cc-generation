#!/bin/bash
# Setup script for Intelligent CC Suggestion Tool
# Downloads required model files and installs dependencies

set -e

echo "=== Intelligent CC Suggestion Tool - Setup ==="

# Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

# Download MediaPipe model files
echo "[2/3] Downloading MediaPipe models..."
mkdir -p models

if [ ! -f "models/pose_landmarker_lite.task" ]; then
    curl -sL -o models/pose_landmarker_lite.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    echo "  Downloaded pose_landmarker_lite.task"
else
    echo "  pose_landmarker_lite.task already exists"
fi

if [ ! -f "models/face_landmarker.task" ]; then
    curl -sL -o models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    echo "  Downloaded face_landmarker.task"
else
    echo "  face_landmarker.task already exists"
fi

# Verify ffmpeg
echo "[3/3] Checking system dependencies..."
if command -v ffmpeg &> /dev/null; then
    echo "  ffmpeg: OK ($(ffmpeg -version | head -1))"
else
    echo "  ffmpeg: NOT FOUND — install with 'apt install ffmpeg'"
    echo "  (Pipeline can still run with pre-extracted WAV files)"
fi

echo ""
echo "=== Setup complete! ==="
echo "Usage: python main.py <video_file> [--verbose]"
