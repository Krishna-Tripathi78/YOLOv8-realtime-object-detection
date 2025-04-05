import cv2
import threading
from ultralytics import YOLO
model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
cv2.namedWindow("Real-Time Object Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Real-Time Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame = None
def capture_frames():
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        frame = new_frame
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    if frame is None:
        continue
    results = model(frame, conf=0.25)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            confidence = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
