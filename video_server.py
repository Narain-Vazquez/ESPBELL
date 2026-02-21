import cv2
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# =========================
# Configuration
# =========================
STREAM_URL = "http://192.168.0.50:8082/stream"  # ESP32-CAM IP
FRAME_WIDTH = 640   # Match your ESP32-CAM resolution
FRAME_HEIGHT = 480
FPS = 20            # Frames per second for recording
DURATION = 60       # Record for 60 seconds

central_zone = ZoneInfo("America/Chicago")
now_central = datetime.now(central_zone)

timestamp = now_central.strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"esp32_output_{timestamp}.mp4"

# =========================
# Open the video stream
# =========================
print(f"[INFO] Connecting to ESP32-CAM stream at {STREAM_URL}...")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("[ERROR] Cannot open video stream. Check ESP32 IP and Wi-Fi connection.")
    exit(1)

print("[INFO] Stream opened successfully.")

# =========================
# Setup video writer
# =========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
print(f"[INFO] Recording video to {OUTPUT_FILE} for {DURATION} seconds...")

# =========================
# Capture loop
# =========================
frame_count = 0
start_time = time.time()

try:
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > DURATION:
            print(f"[INFO] Reached {DURATION} seconds. Stopping recording.")
            break

        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from stream. Retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        print(f"[DEBUG] Captured frame {frame_count}")

        # Resize frame if needed
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Write frame to video
        out.write(frame_resized)

        # Optional: display the frame
        cv2.imshow('ESP32-CAM', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed. Exiting capture loop.")
            break

except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received. Exiting...")

finally:
    # =========================
    # Release resources
    # =========================
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    elapsed_time = time.time() - start_time
    print(f"[INFO] Finished recording {frame_count} frames in {elapsed_time:.2f} seconds.")
    print(f"[INFO] Video saved as {OUTPUT_FILE}.")
