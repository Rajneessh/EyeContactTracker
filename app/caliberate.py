import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
import json

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

cap = cv2.VideoCapture(0)

print("Calibration starting. Look DIRECTLY at your camera.")
collect_duration = 3.0
predictions = []
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    display = frame.copy()

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_crop = get_eye_crop(frame, landmarks, LEFT_EYE_IDX, w, h)

        if left_crop.size > 0:
            left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            left_resized = cv2.resize(left_gray, (60, 36))
            theta, phi = predict_gaze(left_resized)

            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time
            remaining = collect_duration - elapsed

            if remaining > 0:
                predictions.append((theta, phi))
                cv2.putText(display, f"Look at camera... {remaining:.1f}s",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                break

    cv2.imshow("Calibration", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(predictions) < 10:
    print("Not enough samples captured, try again.")
else:
    predictions = np.array(predictions)
    baseline_theta = float(np.median(predictions[:, 0]))
    baseline_phi = float(np.median(predictions[:, 1]))
    std_theta = float(np.std(predictions[:, 0]))
    std_phi = float(np.std(predictions[:, 1]))

    calib = {
        "baseline_theta": baseline_theta,
        "baseline_phi": baseline_phi,
        "std_theta": std_theta,
        "std_phi": std_phi,
        "n_samples": len(predictions)
    }

    with open("app/calibration.json", "w") as f:
        json.dump(calib, f, indent=2)

    print(f"Calibration saved: {calib}")