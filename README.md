# YOLOv8 Real-time Object Detection

A comprehensive YOLOv8-based object detection system with GUI interface for real-time webcam and video file processing.

## Features

- **Multiple YOLO Models**: Support for nano, small, medium, and large models
- **Real-time Detection**: Live webcam and video file processing
- **Complete Detection Coverage**: Dual-pass detection for maximum accuracy
- **GUI Interface**: User-friendly tkinter interface
- **Video Management**: Built-in video folder browser
- **Detection Logging**: Comprehensive detection logs and statistics
- **Multiple Output Formats**: Save screenshots and processed videos

## Installation

```bash
pip install ultralytics opencv-python numpy tkinter
```

## Usage

### Quick Start
```bash
python YOLO8.py
```

### Features
- **Webcam Detection**: Real-time object detection from camera
- **Video Processing**: Process video files with complete object detection
- **Model Selection**: Choose between different YOLO model sizes
- **Detection Logging**: Automatic logging of all detections
- **Progress Tracking**: Real-time progress monitoring

### Controls
- **Q**: Quit detection
- **S**: Save screenshot
- **P**: Pause/Resume video
- **F**: Fast forward (skip 10 frames)
- **R**: Reset detection smoothing

## Model Performance

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| Nano | Fastest | Good | Real-time webcam |
| Small | Fast | Better | Balanced performance |
| Medium | Moderate | High | Detailed analysis |
| Large | Slow | Highest | Maximum accuracy |

## Output Files

- `detection_log.txt`: Complete detection log
- `complete_detection.mp4`: Processed video with detections
- `detection_frame_*.jpg`: Saved screenshots

## Detection Statistics

The system provides comprehensive statistics including:
- Total objects detected
- Objects per frame
- Unique object types
- Detection confidence levels
- Processing progress

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Tkinter

## License

MIT License