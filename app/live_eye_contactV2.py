import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from collections import deque

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("NOTE: pywin32 not installed — overlay will NOT be forced always-on-top. "
          "Install with `pip install pywin32` later to enable that.")


class GazeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 5, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GazeCNN().to(device)
model.load_state_dict(torch.load('models/best_gaze_model.pt', map_location=device))
model.eval()
print(f"Model loaded on {device}")

TOLERANCE_DEG = 12.0
SMOOTHING_WINDOW = 10
BREAK_CONFIRM_FRAMES = 8
LOG_PATH = "app/session_log.json"

CALIB_ROUNDS = 3
CALIB_ROUND_DURATION = 2.0

OVERLAY_W, OVERLAY_H = 260, 90
OVERLAY_X, OVERLAY_Y = 40, 40
WINDOW_NAME = "Eye Contact Monitor"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]


def beep(freq=1000, dur=150):
    if HAS_WINSOUND:
        try:
            winsound.Beep(freq, dur)
            return
        except Exception:
            pass
    print('\a')  # fallback


def get_eye_crop(frame, landmarks, idx_list, w, h, pad=8):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx_list])
    x_min, y_min = pts.min(axis=0).astype(int)
    x_max, y_max = pts.max(axis=0).astype(int)
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
    return frame[y_min:y_max, x_min:x_max]


def predict_gaze(eye_gray_60x36):
    img = eye_gray_60x36.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0]
    return pred


def get_both_eye_preds(frame, landmarks, w, h):
    left_pred, right_pred = None, None
    left_crop = get_eye_crop(frame, landmarks, LEFT_EYE_IDX, w, h)
    if left_crop.size > 0:
        left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
        left_resized = cv2.resize(left_gray, (60, 36))
        left_pred = predict_gaze(left_resized)
    right_crop = get_eye_crop(frame, landmarks, RIGHT_EYE_IDX, w, h)
    if right_crop.size > 0:
        right_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
        right_resized = cv2.resize(right_gray, (60, 36))
        right_pred = predict_gaze(right_resized)
    return left_pred, right_pred


def draw_overlay(status_text, status_color, deviation_deg):
    canvas = np.zeros((OVERLAY_H, OVERLAY_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.circle(canvas, (25, 45), 14, status_color, -1)
    cv2.putText(canvas, status_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"{deviation_deg:.1f} deg", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return canvas


def make_overlay_topmost(window_name, x, y, w, h):
    if not HAS_WIN32:
        return
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, w, h, win32con.SWP_SHOWWINDOW)


def load_session_log(left_baseline, right_baseline):
    return {
        "session_start": datetime.now().isoformat(),
        "events": [],
        "calibration": {
            "left_baseline_theta": float(left_baseline[0]),
            "left_baseline_phi": float(left_baseline[1]),
            "right_baseline_theta": float(right_baseline[0]),
            "right_baseline_phi": float(right_baseline[1]),
        }
    }


def save_session_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


# ============ OPEN CAMERA FIRST ============

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: could not open camera.")
    exit()

print("Camera opened. Showing live preview — press SPACE when ready to calibrate, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    display = frame.copy()
    cv2.putText(display, "Press SPACE to begin calibration", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ============ CALIBRATION PHASE ============

left_samples_all = []
right_samples_all = []

print(f"\n=== Calibration: {CALIB_ROUNDS} rounds ===\n")

for round_num in range(1, CALIB_ROUNDS + 1):
    print(f"Round {round_num}/{CALIB_ROUNDS} — get ready...")

    for count in [5, 4, 3, 2, 1]:
        countdown_start = time.time()
        beep(800, 120)
        while time.time() - countdown_start < 1.0:
            ret, frame = cap.read()
            if not ret:
                continue
            display = frame.copy()
            cv2.putText(display, f"Round {round_num}/{CALIB_ROUNDS}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, str(count), (display.shape[1] // 2 - 20, display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    beep(1500, 250)

    round_left = []
    round_right = []
    round_start = time.time()

    while time.time() - round_start < CALIB_ROUND_DURATION:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        display = frame.copy()
        remaining = CALIB_ROUND_DURATION - (time.time() - round_start)
        cv2.putText(display, f"LOOK AT CAMERA: {remaining:.1f}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_pred, right_pred = get_both_eye_preds(frame, landmarks, w, h)
            if left_pred is not None:
                round_left.append(left_pred)
            if right_pred is not None:
                round_right.append(right_pred)

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    beep(600, 150)

    left_samples_all.extend(round_left)
    right_samples_all.extend(round_right)
    print(f"  Round {round_num} done: {len(round_left)} left-eye, {len(round_right)} right-eye samples")

if len(left_samples_all) < 10 or len(right_samples_all) < 10:
    print("Not enough calibration samples captured. Exiting.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

left_arr = np.array(left_samples_all)
right_arr = np.array(right_samples_all)
left_baseline = np.median(left_arr, axis=0)
right_baseline = np.median(right_arr, axis=0)

print(f"\nCalibration complete.")
print(f"  Left eye baseline (theta, phi):  {np.degrees(left_baseline)}")
print(f"  Right eye baseline (theta, phi): {np.degrees(right_baseline)}")
print(f"  Total samples: {len(left_samples_all)} left, {len(right_samples_all)} right\n")

beep(2000, 400)

# ============ LIVE DETECTION PHASE (minimal overlay) ============

# Force a clean window recreation at the correct small size instead of
# reusing/resizing the old full-camera-frame window (avoids the overlay
# silently failing to display on some Windows/OpenCV builds).
cv2.destroyWindow(WINDOW_NAME)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, OVERLAY_W, OVERLAY_H)

session_log = load_session_log(left_baseline, right_baseline)

history = deque(maxlen=SMOOTHING_WINDOW)
break_streak = 0
currently_away = False
away_start_time = None
away_start_deg = None

print("Starting live monitoring (minimal overlay). Press 'q' (with overlay focused) to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status_label = "NO FACE"
        status_color = (110, 110, 110)
        deviation_deg = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_pred, right_pred = get_both_eye_preds(frame, landmarks, w, h)

            combined_pred = None
            if left_pred is not None and right_pred is not None:
                left_dev = left_pred - left_baseline
                right_dev = right_pred - right_baseline
                combined_pred = (left_dev + right_dev) / 2.0
            elif left_pred is not None:
                combined_pred = left_pred - left_baseline
            elif right_pred is not None:
                combined_pred = right_pred - right_baseline

            if combined_pred is not None:
                history.append(combined_pred)
                smoothed_dev = np.mean(history, axis=0)
                deviation_deg = float(np.degrees(np.linalg.norm(smoothed_dev)))

                if deviation_deg > TOLERANCE_DEG:
                    break_streak += 1
                else:
                    break_streak = 0

                now_away = break_streak >= BREAK_CONFIRM_FRAMES

                if now_away and not currently_away:
                    currently_away = True
                    away_start_time = time.time()
                    away_start_deg = deviation_deg
                elif not now_away and currently_away:
                    currently_away = False
                    event = {
                        "start_time": datetime.fromtimestamp(away_start_time).isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "duration_sec": round(time.time() - away_start_time, 2),
                        "peak_deviation_deg": round(away_start_deg, 2)
                    }
                    session_log["events"].append(event)
                    save_session_log(session_log)
                    print(f"Logged: looked away for {event['duration_sec']}s (peak {event['peak_deviation_deg']}°)")

                if now_away:
                    status_label = "LOOKING AWAY"
                    status_color = (0, 0, 255)
                else:
                    status_label = "EYE CONTACT"
                    status_color = (0, 200, 0)

        overlay_frame = draw_overlay(status_label, status_color, deviation_deg)
        cv2.imshow(WINDOW_NAME, overlay_frame)
        make_overlay_topmost(WINDOW_NAME, x=OVERLAY_X, y=OVERLAY_Y, w=OVERLAY_W, h=OVERLAY_H)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    import traceback
    print("LIVE LOOP CRASHED:")
    traceback.print_exc()

if currently_away:
    event = {
        "start_time": datetime.fromtimestamp(away_start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "duration_sec": round(time.time() - away_start_time, 2),
        "peak_deviation_deg": round(away_start_deg, 2),
        "note": "session ended while looking away"
    }
    session_log["events"].append(event)

session_log["session_end"] = datetime.now().isoformat()
save_session_log(session_log)
print(f"\nSession log saved to {LOG_PATH} — {len(session_log['events'])} look-away events recorded.")

cap.release()
cv2.destroyAllWindows()