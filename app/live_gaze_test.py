import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# --- Model definition (must match training exactly) ---
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

# --- MediaPipe setup (same as before) ---
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
    return pred  # theta, phi in radians

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_crop = get_eye_crop(frame, landmarks, LEFT_EYE_IDX, w, h)

        if left_crop.size > 0:
            left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            left_resized = cv2.resize(left_gray, (60, 36))

            theta, phi = predict_gaze(left_resized)

            # draw arrow on the upscaled display version
            disp = cv2.resize(left_resized, (240, 144))
            disp_color = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            cx, cy = 120, 72
            length = 60
            dx = int(length * np.sin(phi))
            dy = int(-length * np.sin(theta))
            cv2.arrowedLine(disp_color, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2)
            cv2.putText(disp_color, f"theta={np.degrees(theta):.1f} phi={np.degrees(phi):.1f}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow("Live Gaze", disp_color)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()