import cv2
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO

# =========================
# Configuration
# =========================
STREAM_URL = "http://192.168.0.50:8082/stream"  # ESP32-CAM IP
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
DURATION = 60             # Seconds to record per trigger
DETECTION_COOLDOWN = 65   # Minimum seconds between recordings
PERSON_CLASS_ID = 0       # YOLO class ID for humans
TIMEZONE = "America/Chicago"

# =========================
# Setup
# =========================
central_zone = ZoneInfo(TIMEZONE)
model = YOLO("yolov8n.pt")

print(f"[INFO] Connecting to ESP32-CAM stream at {STREAM_URL}...")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("[ERROR] Cannot open video stream. Check ESP32 IP and Wi-Fi connection.")
    exit(1)
print("[INFO] Stream opened successfully.")

last_trigger = 0

# =========================
# Main loop
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Run human detection
        results = model(frame, conf=0.4, verbose=False)
        human_detected = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == PERSON_CLASS_ID:
                    human_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Human {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        timestamp_now = time.time()
        if human_detected and (timestamp_now - last_trigger > DETECTION_COOLDOWN):
            # Generate timestamped output file
            now_central = datetime.now(central_zone)
            timestamp_str = now_central.strftime("%Y%m%d_%H%M%S")
            output_file = f"esp32_output_{timestamp_str}.mp4"
            print(f"[ALERT] Human detected! Recording to {output_file} for {DURATION} seconds...")

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

            # Capture loop for exactly DURATION seconds
            start_time = time.time()
            frame_count = 0
            while time.time() - start_time < DURATION:
                ret, record_frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_resized = cv2.resize(record_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                out.write(frame_resized)
                frame_count += 1

            out.release()
            last_trigger = timestamp_now
            print(f"[INFO] Finished recording {frame_count} frames. Video saved as {output_file}.")

        # Display detection frame
        cv2.imshow("ESP32-CAM Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed. Exiting.")
            break

except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")
