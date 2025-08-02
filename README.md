# 🧠 YOLOv8 Real-Time Object Detection 🚀  
**Author**: Krishna Tripathi  

## 📌 Overview  
A comprehensive YOLOv8-based object detection system with GUI interface for real-time webcam and video file processing. This project demonstrates advanced object detection capabilities with dual-pass processing for maximum accuracy.

## ⚙️ Features

- **Multiple YOLO Models**: Support for nano, small, medium, and large models
- **Real-time Detection**: Live webcam and video file processing
- **Complete Detection Coverage**: Dual-pass detection for maximum accuracy
- **GUI Interface**: User-friendly tkinter interface
- **Video Management**: Built-in video folder browser
- **Detection Logging**: Comprehensive detection logs and statistics
- **Multiple Output Formats**: Save screenshots and processed videos

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start
```bash
python YOLO8.py
```

### Controls
- **Q**: Quit detection
- **S**: Save screenshot
- **P**: Pause/Resume video
- **F**: Fast forward (skip 10 frames)

## 📊 Model Performance

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| Nano | Fastest | Good | Real-time webcam |
| Small | Fast | Better | Balanced performance |
| Medium | Moderate | High | Detailed analysis |
| Large | Slow | Highest | Maximum accuracy |

## 📂 Output Files

- `detection_log.txt`: Complete detection log
- `complete_detection.mp4`: Processed video with detections
- `frame_*.jpg`: Saved screenshots

## 👨‍💻 Author
Krishna Tripathi  
B.Tech Cyber Security | Open Source Enthusiast  
🌍 Kanpur, India

## 📜 License
MIT License