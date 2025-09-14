import cv2
import os
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time

class YOLODetector:
    def __init__(self):
        self.models = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt', 
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt'
        }
        self.current_model = None
        self.cap = None
        self.running = False
        self.conf_threshold = 0.5
        
    def load_model(self, model_size='medium'):
        """Load YOLO model"""
        try:
            model_path = self.models.get(model_size, 'yolov8m.pt')
            self.current_model = YOLO(model_path)
            print(f"✅ Loaded {model_size} model successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            try:
                self.current_model = YOLO('yolov8n.pt')
                print("✅ Fallback to nano model")
                return True
            except:
                return False
    
    def detect_objects(self, frame):
        """Run object detection on frame"""
        if self.current_model is None:
            return frame, []
        
        try:
            results = self.current_model(frame, conf=self.conf_threshold, verbose=False)
            detections = []
            
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        label = self.current_model.names[int(cls)]
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'label': label,
                            'confidence': float(conf)
                        })
            
            return self.draw_detections(frame, detections), detections
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def process_webcam(self, model_size='medium', save_output=False):
        """Process webcam feed"""
        if not self.load_model(model_size):
            return False
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('webcam_output.mp4', fourcc, 20.0, (1280, 720))
        
        cv2.namedWindow("YOLO Webcam Detection", cv2.WINDOW_NORMAL)
        self.running = True
        frame_count = 0
        
        print("🎥 Webcam detection started. Press 'q' to quit, 's' to save frame")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame, detections = self.detect_objects(frame)
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Objects: {len(detections)} | Press 'q' to quit"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("YOLO Webcam Detection", processed_frame)
            
            if save_output and out:
                out.write(processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"webcam_frame_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 Saved: {filename}")
        
        self.cleanup(out)
        print(f"✅ Webcam detection completed. Processed {frame_count} frames")
        return True
    
    def process_video(self, video_path, model_size='medium', save_output=False):
        """Process video file"""
        if not self.load_model(model_size):
            return False
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_name = f"processed_{os.path.basename(video_path)}"
            out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
        
        cv2.namedWindow("YOLO Video Detection", cv2.WINDOW_NORMAL)
        self.running = True
        frame_count = 0
        
        print(f"🎬 Processing video: {os.path.basename(video_path)}")
        print("Press 'q' to quit, 's' to save frame, 'p' to pause")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame, detections = self.detect_objects(frame)
            
            # Add progress info
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | Objects: {len(detections)}"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("YOLO Video Detection", processed_frame)
            
            if save_output and out:
                out.write(processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"video_frame_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('p'):
                print("⏸️ Paused - press any key to continue")
                cv2.waitKey(0)
        
        self.cleanup(out)
        print(f"✅ Video processing completed. Processed {frame_count}/{total_frames} frames")
        return True
    
    def process_image(self, image_path, model_size='medium', save_output=True):
        """Process single image"""
        if not self.load_model(model_size):
            return False
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Cannot load image: {image_path}")
            return False
        
        print(f"📸 Processing image: {os.path.basename(image_path)}")
        
        processed_image, detections = self.detect_objects(image)
        
        # Add detection info
        height, width = image.shape[:2]
        info_text = f"Objects detected: {len(detections)} | Image: {width}x{height}"
        cv2.putText(processed_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display results
        cv2.namedWindow("YOLO Image Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Image Detection", processed_image)
        
        if save_output:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_name = f"{base_name}_detected.jpg"
            cv2.imwrite(output_name, processed_image)
            print(f"💾 Saved: {output_name}")
        
        # Print detection details
        if detections:
            print(f"\n📋 Detection Results:")
            for i, det in enumerate(detections, 1):
                print(f"{i}. {det['label']}: {det['confidence']:.3f}")
        else:
            print("No objects detected")
        
        print("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    
    def cleanup(self, out=None):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.running = False

class YOLOGui:
    def __init__(self):
        self.detector = YOLODetector()
        self.root = tk.Tk()
        self.root.title("YOLOv8 Object Detection System")
        self.root.geometry("600x700")
        self.root.configure(bg='#2c3e50')
        self.setup_gui()
        
    def setup_gui(self):
        # Title
        title_label = tk.Label(self.root, text="🧠 YOLOv8 Detection System 🚀", 
                              font=('Arial', 20, 'bold'), fg='#ecf0f1', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Model Selection Frame
        model_frame = tk.LabelFrame(self.root, text="Model Configuration", 
                                   bg='#34495e', fg='#ecf0f1', font=('Arial', 12, 'bold'))
        model_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(model_frame, text="Model Size:", bg='#34495e', fg='#ecf0f1', 
                font=('Arial', 10)).pack(side='left', padx=10)
        
        self.model_var = tk.StringVar(value="medium")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=["nano", "small", "medium", "large"], 
                                  state="readonly", width=15)
        model_combo.pack(side='left', padx=10, pady=10)
        
        tk.Label(model_frame, text="Confidence:", bg='#34495e', fg='#ecf0f1', 
                font=('Arial', 10)).pack(side='left', padx=(20,5))
        
        self.conf_var = tk.DoubleVar(value=0.5)
        conf_scale = tk.Scale(model_frame, from_=0.1, to=0.9, resolution=0.1,
                             orient='horizontal', variable=self.conf_var, 
                             bg='#34495e', fg='#ecf0f1', length=150)
        conf_scale.pack(side='left', padx=10, pady=10)
        
        # Source Selection Frame
        source_frame = tk.LabelFrame(self.root, text="Input Source Selection", 
                                    bg='#34495e', fg='#ecf0f1', font=('Arial', 12, 'bold'))
        source_frame.pack(fill='x', padx=20, pady=10)
        
        # Source buttons
        btn_frame = tk.Frame(source_frame, bg='#34495e')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="📹 Select Webcam", command=self.select_webcam,
                 bg='#3498db', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, pady=5).pack(pady=5)
        
        tk.Button(btn_frame, text="🎬 Select Video File", command=self.select_video,
                 bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, pady=5).pack(pady=5)
        
        tk.Button(btn_frame, text="🖼️ Select Image File", command=self.select_image,
                 bg='#e67e22', fg='white', font=('Arial', 11, 'bold'), 
                 width=20, pady=5).pack(pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="No source selected")
        status_label = tk.Label(source_frame, textvariable=self.status_var, 
                               bg='#34495e', fg='#2ecc71', font=('Arial', 11, 'bold'))
        status_label.pack(pady=10)
        
        # Options Frame
        options_frame = tk.LabelFrame(self.root, text="Options", 
                                     bg='#34495e', fg='#ecf0f1', font=('Arial', 12, 'bold'))
        options_frame.pack(fill='x', padx=20, pady=10)
        
        self.save_var = tk.BooleanVar()
        tk.Checkbutton(options_frame, text="💾 Save Output", variable=self.save_var,
                      bg='#34495e', fg='#ecf0f1', selectcolor='#34495e', 
                      font=('Arial', 10)).pack(pady=10)
        
        # Control Buttons Frame
        control_frame = tk.LabelFrame(self.root, text="🎯 Detection Controls", 
                                     bg='#34495e', fg='#ecf0f1', font=('Arial', 14, 'bold'))
        control_frame.pack(fill='x', padx=20, pady=20)
        
        # Main control buttons
        tk.Button(control_frame, text="🚀 START DETECTION", command=self.start_detection,
                 bg='#27ae60', fg='white', font=('Arial', 14, 'bold'), 
                 width=25, pady=10).pack(pady=10)
        
        tk.Button(control_frame, text="⏹️ STOP DETECTION", command=self.stop_detection,
                 bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'), 
                 width=25, pady=10).pack(pady=10)
        
        # Help Frame
        help_frame = tk.LabelFrame(self.root, text="Keyboard Controls", 
                                  bg='#34495e', fg='#ecf0f1', font=('Arial', 11, 'bold'))
        help_frame.pack(fill='x', padx=20, pady=10)
        
        help_text = "Q: Quit | S: Save Frame | P: Pause/Resume (video only)"
        tk.Label(help_frame, text=help_text, bg='#34495e', fg='#bdc3c7', 
                font=('Arial', 9)).pack(pady=5)
        
        # Footer
        footer = tk.Label(self.root, text="Made by Krishna Tripathi | B.Tech Cyber Security", 
                         font=('Arial', 10, 'italic'), fg='#95a5a6', bg='#2c3e50')
        footer.pack(pady=20)
        
        # Initialize variables
        self.source = None
        self.source_type = None
        
    def select_webcam(self):
        self.source = 0
        self.source_type = 'webcam'
        self.status_var.set("✅ Webcam selected")
        
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.source = file_path
            self.source_type = 'video'
            filename = os.path.basename(file_path)
            self.status_var.set(f"✅ Video: {filename}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.source = file_path
            self.source_type = 'image'
            filename = os.path.basename(file_path)
            self.status_var.set(f"✅ Image: {filename}")
    
    def start_detection(self):
        if self.source is None:
            messagebox.showerror("Error", "Please select a source first!")
            return
        
        # Update detector settings
        self.detector.conf_threshold = self.conf_var.get()
        model_size = self.model_var.get()
        save_output = self.save_var.get()
        
        def run_detection():
            try:
                if self.source_type == 'webcam':
                    self.detector.process_webcam(model_size, save_output)
                elif self.source_type == 'video':
                    self.detector.process_video(self.source, model_size, save_output)
                elif self.source_type == 'image':
                    self.detector.process_image(self.source, model_size, save_output)
            except Exception as e:
                messagebox.showerror("Error", f"Detection failed: {str(e)}")
        
        # Run in separate thread
        threading.Thread(target=run_detection, daemon=True).start()
    
    def stop_detection(self):
        self.detector.running = False
        messagebox.showinfo("Info", "Detection stopped")
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.detector.running = False
        self.detector.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    try:
        print("🚀 Starting YOLOv8 Detection System...")
        app = YOLOGui()
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        cv2.destroyAllWindows()