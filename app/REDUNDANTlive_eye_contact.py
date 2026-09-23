import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import json
import os
import time
from datetime import datetime
from collections import deque

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

with open("app/calibration.json") as f:
    calib = json.load(f)

baseline = np.array([calib["baseline_theta"], calib["baseline_phi"]])
TOLERANCE_DEG = 12.0
SMOOTHING_WINDOW = 10
BREAK_CONFIRM_FRAMES = 8

LOG_PATH = "app/session_log.json"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]

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

def load_session_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return {"session_start": datetime.now().isoformat(), "events": []}

def save_session_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

session_log = load_session_log()
session_log["session_start"] = datetime.now().isoformat()
session_log["events"] = []

history = deque(maxlen=SMOOTHING_WINDOW)
break_streak = 0
currently_away = False
away_start_time = None
away_start_deg = None
session_start_ts = time.time()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = "NO FACE"
    status_color = (128, 128, 128)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_crop = get_eye_crop(frame, landmarks, LEFT_EYE_IDX, w, h)

        if left_crop.size > 0:
            left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            left_resized = cv2.resize(left_gray, (60, 36))
            pred = predict_gaze(left_resized)
            history.append(pred)

            smoothed = np.mean(history, axis=0)
            deviation_rad = np.abs(smoothed - baseline)
            deviation_deg = float(np.degrees(np.linalg.norm(deviation_rad)))

            if deviation_deg > TOLERANCE_DEG:
                break_streak += 1
            else:
                break_streak = 0

            now_away = break_streak >= BREAK_CONFIRM_FRAMES

            # --- event logging: log on transition, not every frame ---
            if now_away and not currently_away:
                # eye contact just broken -> record start of event
                currently_away = True
                away_start_time = time.time()
                away_start_deg = deviation_deg

            elif not now_away and currently_away:
                # eye contact just resumed -> record completed event
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
                status_text = f"LOOKING AWAY ({deviation_deg:.1f} deg)"
                status_color = (0, 0, 255)
            else:
                status_text = f"EYE CONTACT ({deviation_deg:.1f} deg)"
                status_color = (0, 255, 0)

    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.imshow("Eye Contact Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# if still "away" when quitting, log the final incomplete event
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