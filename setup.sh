#!/bin/bash
# Setup script for Intelligent CC Suggestion Tool
# Installs all dependencies and downloads required model files

set -e

echo "=== Intelligent CC Suggestion Tool — Setup ==="
echo ""

# Install Python dependencies
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Download MediaPipe model files
echo ""
echo "[2/4] Downloading MediaPipe models..."
mkdir -p models

if [ ! -f "models/pose_landmarker_lite.task" ]; then
    curl -sL -o models/pose_landmarker_lite.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    echo "  ✓ Downloaded pose_landmarker_lite.task"
else
    echo "  ✓ pose_landmarker_lite.task already exists"
fi

if [ ! -f "models/face_landmarker.task" ]; then
    curl -sL -o models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    echo "  ✓ Downloaded face_landmarker.task"
else
    echo "  ✓ face_landmarker.task already exists"
fi

# Generate test data if not present
echo ""
echo "[3/4] Setting up test data..."
if [ ! -f "samples/test_clip.avi" ]; then
    python tests/generate_test_data.py
    echo "  ✓ Generated synthetic test clip"
else
    echo "  ✓ Test clip already exists"
fi

# Check system dependencies
echo ""
echo "[4/4] Checking system dependencies..."
if command -v ffmpeg &> /dev/null; then
    echo "  ✓ ffmpeg: installed"
else
    echo "  ⚠ ffmpeg: NOT FOUND — install with 'sudo apt install ffmpeg'"
    echo "    (Pipeline can still run with pre-extracted WAV files or OpenCV fallback)"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "  CLI:      python main.py <video_file> [--verbose]"
echo "  Web UI:   python web/app.py  →  http://localhost:8000"
echo "  Tests:    python -m pytest tests/test_all.py -v"
