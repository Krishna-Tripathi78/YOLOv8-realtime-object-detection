# 🧠 YOLOv8 Real-Time Object Detection System 🚀  
**Author**: Krishna Tripathi  
**B.Tech Cyber Security | Kanpur, India**

## 📌 Overview  
Advanced YOLOv8 object detection system with modern GUI interface supporting real-time webcam detection, video processing, and image analysis. Features dual-pass detection algorithm, object tracking, and comprehensive logging capabilities.

## ⚙️ Key Features

### 🎯 **Detection Capabilities**
- **Multi-Pass Detection**: Dual-pass algorithm for enhanced accuracy
- **Real-Time Processing**: Live webcam detection with optimized performance
- **Video Analysis**: Process MP4, AVI, MOV, MKV video files
- **Image Processing**: Single image detection with detailed analysis
- **Object Tracking**: Optional tracking for video sources

### 🖥️ **Modern GUI Interface**
- **Dark Theme**: Professional dark blue interface design
- **Organized Layout**: Sectioned controls for better usability
- **Live Status**: Real-time feedback and source selection status
- **Multiple Input Methods**: Webcam, video files, image files, video folder browser
- **Visual Feedback**: Emoji-enhanced buttons and status messages

### 🤖 **YOLO Model Support**
- **YOLOv8n (Nano)**: Ultra-fast for real-time applications
- **YOLOv8s (Small)**: Balanced speed and accuracy
- **YOLOv8m (Medium)**: Higher accuracy for detailed analysis
- **YOLOv8l (Large)**: Maximum accuracy for critical applications

### 📊 **Advanced Analytics**
- **Confidence Scoring**: Detailed confidence levels for each detection
- **Object Counting**: Real-time count of detected objects
- **Detection Statistics**: Comprehensive analysis and reporting
- **Bounding Box Visualization**: Color-coded detection boxes
- **Performance Metrics**: FPS monitoring and processing statistics

## 🛠️ Installation

### Prerequisites
```bash
python >= 3.8
opencv-python
ultralytics
numpy
torch
torchvision
```

### Install Dependencies
```bash
pip install ultralytics opencv-python numpy torch torchvision
# OR
pip install -r requirements.txt
```

## 🚀 Usage Guide

### Launch Application
```bash
python YOLO8.py
```

### 🎮 **Interface Controls**

#### **Model Configuration**
- Select YOLO model size (nano/small/medium/large)
- Choose based on speed vs accuracy requirements

#### **Input Source Selection**
- 📹 **Webcam**: Real-time camera detection
- 📂 **Video Folder**: Browse videos from local folder
- 🎬 **Video File**: Select individual video files
- 🖼️ **Photo/Image**: Process single images

#### **Detection Options**
- 🎯 **Object Tracking**: Enable for video sources
- 💾 **Save Output**: Save processed videos with detections

#### **Runtime Controls**
- **Q**: Quit detection
- **S**: Save current frame screenshot
- **P**: Pause/Resume video playback
- **F**: Fast forward (skip 10 frames)
- **ESC**: Exit application

## 📊 Model Performance Comparison

| Model | Size | Speed (FPS) | mAP | Best Use Case |
|-------|------|-------------|-----|---------------|
| YOLOv8n | 6.2MB | 45-60 | 37.3 | Real-time webcam, mobile |
| YOLOv8s | 21.5MB | 35-45 | 44.9 | Balanced performance |
| YOLOv8m | 49.7MB | 25-35 | 50.2 | High accuracy applications |
| YOLOv8l | 83.7MB | 15-25 | 52.9 | Maximum precision required |

## 📂 Output Files & Logging

### **Video Processing**
- `complete_detection.mp4`: Processed video with bounding boxes
- `detection_log.txt`: Frame-by-frame detection log
- `frame_YYYYMMDD_HHMMSS.jpg`: Saved screenshots

### **Image Processing**
- `[filename]_detected.jpg`: Processed image with detections
- `[filename]_detection_log.txt`: Detailed detection report

### **Log Contents**
- Object detection coordinates
- Confidence scores
- Processing statistics
- Timestamp information
- Performance metrics

## 🔧 Technical Implementation

### **Detection Algorithm**
1. **Primary Pass**: Standard YOLOv8 detection
2. **Secondary Pass**: Enhanced detection on cropped regions
3. **NMS Processing**: Non-maximum suppression for accuracy
4. **Confidence Filtering**: Configurable threshold filtering

### **Performance Optimizations**
- Multi-threading for GUI responsiveness
- Efficient memory management
- Optimized frame processing pipeline
- Dynamic model loading

## 🎯 Supported Object Classes
Detects 80+ COCO dataset classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, bird, horse, etc.
- **Objects**: bottle, chair, laptop, phone, etc.

## 🚨 System Requirements
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: CUDA-compatible GPU recommended for faster processing
- **Storage**: 2GB free space for models and outputs

## 🤝 Contributing
Contributions welcome! Please feel free to submit pull requests or open issues for improvements.

## 📞 Support
For questions or support:
- 📧 Email: [Contact via GitHub]
- 🐛 Issues: [GitHub Issues Page]
- 💡 Feature Requests: [GitHub Discussions]

## 👨💻 Author
**Krishna Tripathi**  
🎓 B.Tech Cyber Security  
🌟 Open Source Enthusiast  
🌍 Kanpur, India  

*"Advancing computer vision through accessible AI solutions"*

## 📜 License
MIT License - Feel free to use, modify, and distribute

## 🙏 Acknowledgments
- Ultralytics team for YOLOv8
- OpenCV community
- PyTorch developers

---
*⭐ Star this repository if you found it helpful!*