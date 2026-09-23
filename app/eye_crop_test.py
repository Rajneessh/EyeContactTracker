import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # needed for iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe's eye region landmark indices (left eye, from the subject's perspective)
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]

def get_eye_crop(frame, landmarks, idx_list, w, h, pad=8):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx_list])
    x_min, y_min = pts.min(axis=0).astype(int)
    x_max, y_max = pts.max(axis=0).astype(int)
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
    crop = frame[y_min:y_max, x_min:x_max]
    return crop

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
        right_crop = get_eye_crop(frame, landmarks, RIGHT_EYE_IDX, w, h)

        if left_crop.size > 0:
            left_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            left_resized = cv2.resize(left_gray, (60, 36))
            cv2.imshow("Left Eye", cv2.resize(left_resized, (240, 144)))  # upscaled for visibility

        if right_crop.size > 0:
            right_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)
            right_resized = cv2.resize(right_gray, (60, 36))
            cv2.imshow("Right Eye", cv2.resize(right_resized, (240, 144)))

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()