# EyeContactTracker

Real-time eye contact detection and gaze tracking for video interviews. Built to help HR firms and proctoring systems flag when a candidate's gaze or head position drifts away from the camera during online interviews — whether checking notes, reading answers off a second screen, or turning away.

The tool actively monitors live camera video, calibrates per candidate, captures candidate identification, and records detailed look-away events (start time, end time, duration, peak deviation angle, and event type).

---

## Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Webcam Frame   │────▶│    MediaPipe     │────▶│     Eye Crop      │
│  (live video)   │     │    Face Mesh     │     │    Extraction     │
│                 │     │   (landmarks)    │     │  (L + R, 36x60)   │
└─────────────────┘     └─────────┬────────┘     └─────────┬─────────┘
                                  │                        │
                                  ▼                        ▼
                       ┌────────────────────┐    ┌───────────────────┐
                       │  Geometric Head    │    │      GazeCNN      │
                       │  Pose Estimation   │    │  (5 conv layers,  │
                       │   (Yaw & Pitch)    │    │   ~394K params)   │
                       └──────────┬─────────┘    │   → theta, phi    │
                                  │              └─────────┬─────────┘
                                  │                        │
        ┌─────────────────────────┴────────────────────────┼──────────────────────┐
        ▼                                                  ▼                      ▼
┌────────────────────────┐                        ┌──────────────────┐  ┌──────────────────┐
│   Head-Pose Gating     │                        │  Per-session     │  │ Rolling smoothing│
│ (yaw/pitch > threshold)│                        │  baseline median │  │   + debounce     │
│  → FACE TURNED AWAY    │                        └────────┬─────────┘  └────────┬─────────┘
└────────────────────────┘                                 │                     │
                                                           ▼                     ▼
                                                  ┌──────────────────────────────────┐
                                                  │  Gaze Deviation vs Baseline      │
                                                  │  (dev > 5° → LOOKING AWAY)       │
                                                  └────────────────┬─────────────────┘
                                                                   ▼
                                                  ┌──────────────────────────────────┐
                                                  │     Candidate Session Folder     │
                                                  │  sessions/<Candidate_Timestamp>/ │
                                                  │  ├── candidate_photo.jpg         │
                                                  │  └── session_log.json            │
                                                  └──────────────────────────────────┘
```

---

## CNN Architecture (GazeCNN)

```text
Input: 1 x 36 x 60 (grayscale eye patch)
  │
  ▼
Conv(1→32, 3x3) + BN + ReLU
Conv(32→32, 3x3) + BN + ReLU
MaxPool(2) → 32 x 18 x 30
  │
  ▼
Conv(32→64, 3x3) + BN + ReLU
Conv(64→64, 3x3) + BN + ReLU
MaxPool(2) → 64 x 9 x 15
  │
  ▼
Conv(64→128, 3x3) + BN + ReLU
AdaptiveAvgPool(3x5) → 128 x 3 x 5
  │
  ▼
Flatten → 1920
  │
  ▼
FC(1920→128) + ReLU + Dropout(0.3)
FC(128→64) + ReLU
FC(64→2) → theta, phi (radians)
```

**~394K parameters total.** Model focuses on fine-grained eye direction while geometric landmark ratio gating handles head pose turns.

---

## Data & Model Pipeline

```text
MPIIGaze (.mat files, 15 participants)
  │
  ▼
┌──────────────────────┐
│    Load + unpack     │  427,316 samples (213,658 images x L/R eye)
│ 3D gaze → theta/phi  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Label sanity check  │  pupil-offset vs gaze-angle correlation
│  → found phi flipped │  → fixed with sign correction, re-verified
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Quality filtering   │  blur score (Laplacian variance)
│   → 97.9% retained   │  removed soft images + corruption pockets
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Participant-level   │  train: 12 participants (397,739)
│ train/val/test split │  val:   2 participants (18,496)
│  (no frame leakage)  │  test:  1 participant (2,058)
└──────────┬───────────┘
           ▼
GazeCNN training → 3.58° mean angular error on held-out test participant
```

---

## System Features & Evolutionary Improvements

### 1. HR Candidate Registration & Session Management (v5)
* **Form Popup**: Launches an HR registration popup at session start to enter the candidate's name.
* **Isolated Folders**: Creates a unique candidate directory under `sessions/{Candidate_Name}_{timestamp}/` for every session.
* **Automatic Photo Snapshot**: Automatically captures a high-resolution photograph of the candidate during the **2nd calibration round** and saves it as `candidate_photo.jpg`.
* **Structured Logs**: Generates per-candidate `session_log.json` files storing calibration baselines and look-away events.

### 2. Head-Pose Gating (Yaw & Pitch)
* **The Problem**: MPIIGaze participants kept their heads relatively still, so CNN gaze models can be mis-calibrated on large head turns (an out-of-distribution regime).
* **The Solution**: Computes real-time geometric head yaw & pitch ratios using MediaPipe Face Mesh landmarks (nose tip `1` relative to eye outer corners `33` & `263`).
* **High Sensitivity**:
  * `HEAD_YAW_GATE_RATIO = 0.07` — flags minor horizontal head turns immediately as `"FACE TURNED AWAY"`.
  * `HEAD_PITCH_GATE_RATIO = 0.07` — flags minor vertical head tilts (e.g. looking down at notes or a second screen).
  * `TOLERANCE_DEG = 5.0°` — flags minor eye gaze shifts when head is forward.

### 3. Always-on-Top Live UI & Windows Integrations
* **Live Status Overlay**: Draws a status banner and colored border over the camera feed (`EYE CONTACT`, `LOOKING AWAY`, `FACE TURNED AWAY`).
* **Always-on-Top Window**: Uses `win32gui` (`HWND_TOPMOST`) to keep the monitor visible during interviews.
* **Audio Countdown**: Uses `winsound` beeps for 5-4-3-2-1 countdowns during calibration.

---

## Session Log Format (`session_log.json`)

```json
{
  "candidate_name": "John Doe",
  "session_start": "2026-09-24T13:50:22.123456",
  "calibration": {
    "left_baseline_theta": -0.103,
    "left_baseline_phi": 0.004,
    "right_baseline_theta": -0.077,
    "right_baseline_phi": 0.017,
    "yaw_baseline_ratio": 0.014,
    "pitch_baseline_ratio": 0.418
  },
  "events": [
    {
      "start_time": "2026-09-24T13:51:05.100000",
      "end_time": "2026-09-24T13:51:06.010000",
      "duration_sec": 0.91,
      "peak_deviation_deg": 23.88,
      "type": "head_turned"
    },
    {
      "start_time": "2026-09-24T13:51:20.500000",
      "end_time": "2026-09-24T13:51:21.000000",
      "duration_sec": 0.50,
      "peak_deviation_deg": 6.25,
      "type": "gaze_drift"
    }
  ],
  "session_end": "2026-09-24T13:52:10.654321"
}
```

---

## Setup & Running

```bash
# Clone the repository
git clone https://github.com/Rajneessh/EyeContactTracker.git
cd EyeContactTracker

# Create and activate virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run live application (v5)
python app/live_eye_contactv5.py
```

### Flow:
1. Enter Candidate Name in the HR popup dialog.
2. OpenCV window opens $\rightarrow$ Press **SPACE** to start 3 calibration rounds.
3. Candidate photo automatically captured during Round 2 $\rightarrow$ saved to `sessions/{Candidate_Name}_{timestamp}/candidate_photo.jpg`.
4. Live monitoring starts. Press `q` to quit.

---

## Packaging as Standalone Executable (PyInstaller)

To distribute to HR personnel without requiring Python or dependencies installed:

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable package using the spec file
pyinstaller EyeContactMonitor.spec
```

The standalone application bundle will be created in `dist/EyeContactMonitor/`. Users can launch `EyeContactMonitor.exe` directly on any Windows machine.

---

## Repo Structure

| Directory / File | Description |
| :--- | :--- |
| `app/live_eye_contactv5.py` | Main production application (Candidate registration, photo capture, pose gating, live tracking) |
| `app/live_eye_contactv4.py` | Ultra-sensitive pose & gaze tracking core |
| `app/live_eye_contactv3.py` | Baseline head-yaw gating implementation |
| `EyeContactMonitor.spec` | PyInstaller build specification for standalone Windows executable |
| `models/best_gaze_model.pt` | Pretrained GazeCNN PyTorch weights (~1.5MB) |
| `notebooks/` | Complete data preprocessing, label bug fix verification, & model training notebook |
| `sessions/` | Output directory for per-candidate folders (`candidate_photo.jpg` & `session_log.json`) |
| `requirements.txt` | Dependency requirements list |

---

## Roadmap Accomplishments

- [x] Geometric head-pose gating (yaw & pitch) for robust off-center head turns
- [x] Candidate registration form & automatic candidate photograph capture
- [x] Per-candidate isolated folder structure (`sessions/`) and JSON event logging
- [x] Always-on-top window overlay with real-time status banner
- [x] Standalone PyInstaller executable build spec (`EyeContactMonitor.spec`)

---

## Credits

* Training Dataset: **MPIIGaze** (Zhang et al.), CC BY-NC-SA 4.0
* Face Landmarks: **MediaPipe Face Mesh**