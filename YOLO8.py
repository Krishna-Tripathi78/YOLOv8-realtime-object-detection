import cv2
import os
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from queue import Queue

class AdvancedYOLO:
    def __init__(self):
        self.models = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt', 
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt'
        }
        self.current_model = None
        self.cap = None
        self.tracking_history = defaultdict(deque)
        self.object_counts = defaultdict(int)
        self.fps_counter = deque(maxlen=30)
        self.detection_zones = []
        self.alerts = []
        self.running = False
        self.prev_detections = []
        self.smooth_factor = 0.7
        
    def load_model(self, model_size='nano'):
        self.current_model = YOLO(self.models[model_size])
        

    

    
    def process_video(self, source, model_size='nano', save_output=False, enable_tracking=True):
        self.load_model(model_size)
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            return False
        
        # Get original video properties
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} at {fps} FPS, Total frames: {total_frames}")
        
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('complete_detection.mp4', fourcc, fps, (width, height))
        
        cv2.namedWindow("Complete YOLO Detection", cv2.WINDOW_NORMAL)
        self.running = True
        
        frame_count = 0
        total_objects = 0
        unique_objects = set()
        detection_log = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("\nVideo processing complete!")
                break
                
            frame_count += 1
            original_frame = frame.copy()
            
            # Multiple detection passes for complete coverage
            all_detections = []
            
            # Pass 1: Standard detection
            results1 = self.current_model(frame, conf=0.15, iou=0.4, verbose=False)
            
            # Pass 2: Enhanced detection with different settings
            results2 = self.current_model(frame, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)
            
            # Combine results from both passes
            for results in [results1, results2]:
                for r in results:
                    if r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        classes = r.boxes.cls.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confs):
                            x1, y1, x2, y2 = map(int, box)
                            label = self.current_model.names[int(cls)]
                            
                            # Avoid duplicate detections
                            detection_key = f"{label}_{x1}_{y1}_{x2}_{y2}"
                            if detection_key not in [d[4] for d in all_detections]:
                                all_detections.append((x1, y1, x2, y2, detection_key, label, conf))
            
            # Process and display all detections
            objects_in_frame = len(all_detections)
            
            for x1, y1, x2, y2, detection_key, label, conf in all_detections:
                unique_objects.add(label)
                
                # Enhanced color coding
                if conf > 0.8:
                    color = (0, 255, 0)  # Bright green - very confident
                    thickness = 4
                elif conf > 0.6:
                    color = (0, 255, 255)  # Yellow - confident
                    thickness = 3
                elif conf > 0.4:
                    color = (0, 165, 255)  # Orange - moderate
                    thickness = 2
                else:
                    color = (255, 0, 255)  # Magenta - low confidence
                    thickness = 2
                
                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Enhanced label with better visibility
                label_text = f"{label} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Label background
                cv2.rectangle(frame, (x1, y1-text_height-15), (x1+text_width+10, y1), color, -1)
                cv2.putText(frame, label_text, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Log detection
                detection_log.append(f"Frame {frame_count}: {label} (conf: {conf:.2f})")
            
            total_objects += objects_in_frame
            progress = (frame_count / total_frames) * 100
            
            # Comprehensive info display
            info_bg_height = 80
            cv2.rectangle(frame, (10, 10), (700, 10+info_bg_height), (0, 0, 0), -1)
            
            info_lines = [
                f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
                f"Objects in frame: {objects_in_frame} | Total detected: {total_objects}",
                f"Unique object types: {len(unique_objects)} | Types: {', '.join(list(unique_objects)[:5])}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (15, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Complete YOLO Detection", frame)
            
            if save_output and out:
                out.write(frame)
            
            # Progress feedback
            if frame_count % 30 == 0:
                print(f"Progress: {progress:.1f}% - Frame {frame_count}/{total_frames} - Objects: {objects_in_frame}")
            
            # Control playback
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                print(f"Saved frame {frame_count} with {objects_in_frame} objects")
            elif key == ord('p'):
                print("Paused - Press any key to continue")
                cv2.waitKey(0)
            elif key == ord('f'):  # Fast forward
                for _ in range(10):
                    self.cap.read()
                    frame_count += 10
        
        # Final summary
        print(f"\n=== DETECTION COMPLETE ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total objects detected: {total_objects}")
        print(f"Unique object types found: {len(unique_objects)}")
        print(f"Object types: {', '.join(unique_objects)}")
        print(f"Average objects per frame: {total_objects/frame_count:.2f}")
        
        # Save detection log
        with open('detection_log.txt', 'w') as f:
            f.write("\n".join(detection_log))
        print("Detection log saved to 'detection_log.txt'")
        
        self.cleanup(out)
        return True
    

    
    def cleanup(self, out=None):
        if self.cap:
            self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.running = False

class YOLOGui:
    def __init__(self):
        self.yolo = AdvancedYOLO()
        self.root = tk.Tk()
        self.root.title("Advanced YOLO Detection System")
        self.root.geometry("500x400")
        self.setup_gui()
        
    def setup_gui(self):
        # Model selection
        ttk.Label(self.root, text="Model Size:").pack(pady=5)
        self.model_var = tk.StringVar(value="small")
        ttk.Combobox(self.root, textvariable=self.model_var, 
                    values=["nano", "small", "medium", "large"]).pack(pady=5)
        
        # Source selection
        ttk.Label(self.root, text="Video Source:").pack(pady=5)
        ttk.Button(self.root, text="Use Webcam", command=self.use_webcam).pack(pady=5)
        ttk.Button(self.root, text="Browse Videos Folder", command=self.browse_videos_folder).pack(pady=5)
        ttk.Button(self.root, text="Select Video File", command=self.select_file).pack(pady=5)
        
        # Options
        self.tracking_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.root, text="Enable Object Tracking", variable=self.tracking_var).pack(pady=5)
        
        self.save_var = tk.BooleanVar()
        ttk.Checkbutton(self.root, text="Save Output Video", variable=self.save_var).pack(pady=5)
        
        # Control buttons
        ttk.Button(self.root, text="Start Detection", command=self.start_detection).pack(pady=10)
        ttk.Button(self.root, text="Stop Detection", command=self.stop_detection).pack(pady=5)
        
        self.source = None
        
    def use_webcam(self):
        self.source = 0
        messagebox.showinfo("Source Selected", "Webcam selected as video source")
        
    def browse_videos_folder(self):
        videos_dir = os.path.join(os.path.dirname(__file__), "videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
            messagebox.showinfo("Info", f"Created videos folder at: {videos_dir}\nAdd your video files there!")
            return
            
        video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            messagebox.showinfo("No Videos", f"No video files found in: {videos_dir}\nAdd some video files to this folder!")
            return
            
        # Create selection window
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Video from Folder")
        selection_window.geometry("400x300")
        
        ttk.Label(selection_window, text="Available Videos:").pack(pady=10)
        
        listbox = tk.Listbox(selection_window, height=10)
        listbox.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        for video in video_files:
            listbox.insert(tk.END, video)
            
        def select_video():
            selection = listbox.curselection()
            if selection:
                selected_video = video_files[selection[0]]
                self.source = os.path.join(videos_dir, selected_video)
                messagebox.showinfo("Source Selected", f"Video selected: {selected_video}")
                selection_window.destroy()
            else:
                messagebox.showwarning("No Selection", "Please select a video file")
                
        ttk.Button(selection_window, text="Select", command=select_video).pack(pady=10)
        ttk.Button(selection_window, text="Cancel", command=selection_window.destroy).pack(pady=5)
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.source = file_path
            messagebox.showinfo("Source Selected", f"Video file selected: {os.path.basename(file_path)}")
    
    def start_detection(self):
        if self.source is None:
            messagebox.showerror("Error", "Please select a video source first")
            return
            
        def run_detection():
            self.yolo.process_video(
                self.source,
                self.model_var.get(),
                self.save_var.get(),
                self.tracking_var.get()
            )
        
        threading.Thread(target=run_detection, daemon=True).start()
    
    def stop_detection(self):
        self.yolo.running = False
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = YOLOGui()
    app.run()
