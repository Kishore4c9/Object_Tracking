import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# CONFIG
INPUT_VIDEO = "/mnt/c/Users/uidv601s/WSL_Codes/playground/object_tracking/videos/Road_traffic_video_for_object_recognition_720p.mp4"
OUTPUT_VIDEO = "/mnt/c/Users/uidv601s/WSL_Codes/playground/object_tracking/videos/Road_traffic_video_for_object_recognition_720p_output_EMA25.mp4"

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
CLASSES_TO_TRACK = [0, 2, 3, 5, 7]  # vehicle classes

DISPLAY = False  # set True if you want live preview

EMA_ALPHA = 0.25   # smoothing factor 
MAX_SPEED_KMH = 80.0


# INIT Model
model = YOLO(MODEL_PATH)
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)

track_history = {}          # Store tracking history
track_speed_ema = {}        # Store smoothed speed per track

cap = cv2.VideoCapture(INPUT_VIDEO)
assert cap.isOpened(), "Failed to open input video"

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_idx = 0

# Homography Setup (One-Time Calibration)
# IMAGE coordinates (pixels) – manually measured
img_pts = np.array([
    [0, 620],   # bottom-left lane marker
    [1275, 620],  # bottom-right lane marker
    [840, 350],   # top-right lane marker
    [430, 350],   # top-left lane marker
], dtype=np.float32)

# WORLD coordinates (meters) – real distances
world_pts = np.array([
    [0.0, 0.0],
    [25.5, 0.0],    # lane width ≈ 3.5 m
    [25.5, 30.0],   # 30 m forward
    [0.0, 30.0],
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)


# HELPER Functions
#def centroid(b):
#    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

def centroid(b): # For smoother tracking, shift centroid reference to bottom of the image
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, y2])

# Convert Image Centroid → World Point
def image_to_world(pt, H):
    p = np.array([[pt[0], pt[1], 1.0]], dtype=np.float32).T
    pw = H @ p
    pw /= pw[2]
    return pw[0][0], pw[1][0]  # (X, Y) in meters

# Velocity Estimation in px/s
def velocity(track_id, c, frame_idx):
    if track_id not in track_history:
        track_history[track_id] = (c, frame_idx)
        return 0.0

    c_prev, f_prev = track_history[track_id]
    dt = (frame_idx - f_prev) / fps
    if dt <= 0:
        return 0.0

    v = np.linalg.norm(c - c_prev) / dt
    track_history[track_id] = (c, frame_idx)
    return v


# Velocity Estimation in km/h
def estimate_speed_kmh(track_id, img_centroid, frame_idx, fps, H):
    X, Y = image_to_world(img_centroid, H)

    if track_id not in track_history:
        track_history[track_id] = ((X, Y), frame_idx)
        return 0.0

    (Xp, Yp), fp = track_history[track_id]
    dt = (frame_idx - fp) / fps
    if dt <= 0:
        return 0.0

    dist = np.sqrt((X - Xp)**2 + (Y - Yp)**2)
    speed_kmh = (dist / dt) * 3.6

    track_history[track_id] = ((X, Y), frame_idx)
    return speed_kmh


def ema_filter(track_id, speed_raw, alpha=0.2):
    if track_id not in track_speed_ema:
        track_speed_ema[track_id] = speed_raw
        return speed_raw

    speed_prev = track_speed_ema[track_id]
    speed_smooth = alpha * speed_raw + (1 - alpha) * speed_prev
    track_speed_ema[track_id] = speed_smooth
    return speed_smooth


# Object Detection & Tracking
#while True:
while frame_idx < 300:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # YOLO DETECTION
    result = model(frame, verbose=False)[0]
    
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls in zip(boxes, scores, classes):
            if score < CONF_THRESH or cls not in CLASSES_TO_TRACK:
                continue
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])

    detections = np.asarray(detections, dtype=np.float32)
    if detections.size == 0:
        detections = np.empty((0, 5), dtype=np.float32)

    # print(frame_idx, ". Detections:", detections.size)

    # SORT Tracking
    tracks = tracker.update(detections)

    # Plot
    for t in tracks:
        x1, y1, x2, y2, tid = t
        tid = int(tid)

        c = centroid([x1, y1, x2, y2])
        # v = velocity(tid, c, frame_idx) # Velocity in pixels/sec
        speed_raw = estimate_speed_kmh(
                                track_id=tid,
                                img_centroid=c,
                                frame_idx=frame_idx,
                                fps=fps,
                                H=H )
        
        # Clip velocity 
        speed_raw = np.clip(speed_raw, 0.0, MAX_SPEED_KMH)

        # EMA smoothing
        speed_kmh = int(ema_filter(tid, speed_raw, EMA_ALPHA))
        

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), 2)
        cv2.putText(frame,
                    f"ID {tid} | {int(speed_kmh)} km/h",
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2)

    writer.write(frame)

    if DISPLAY:
        cv2.imshow("Tracked Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# CLEANUP
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Output saved to {OUTPUT_VIDEO}")
