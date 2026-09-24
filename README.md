# EyeContactTracker

Real-time eye contact detection for video interviews. Built this to help an HR firm flag when a candidate's gaze drifts away from the camera during online interviews — could be checking notes, reading answers off a second screen, or just distraction. The tool watches, logs when it happens, how long, and how far off.

---

## Background

Started as an idea to build gaze detection from scratch rather than wrap an existing SDK — wanted to actually understand and own the pipeline: real dataset, real model, real debugging. Ended up being a solid crash course in appearance-based gaze estimation.

---

## Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Webcam Frame   │────▶│    MediaPipe     │────▶│     Eye Crop      │
│  (live video)   │     │    Face Mesh     │     │    Extraction     │
│                 │     │   (landmarks)    │     │  (L + R, 36x60)   │
└─────────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                           │
                                                           ▼
                                                 ┌───────────────────┐
                                                 │      GazeCNN      │
                                                 │  (5 conv layers,  │
                                                 │   ~394K params)   │
                                                 │   → theta, phi    │
                                                 └─────────┬─────────┘
                                                           │
        ┌──────────────────────────────────────────────────┼──────────────────────┐
        ▼                                                                         ▼
┌───────────────────────┐                                         ┌──────────────────────┐
│      Per-session      │                                         │  Rolling smoothing   │
│ calibration baseline  │───────────────── deviation calc ───────▶│     + debounce       │
│  (median, both eyes)  │                                         │   (avoid flicker)    │
└───────────────────────┘                                         └──────────┬───────────┘
                                                                             ▼
                                                                  ┌──────────────────────┐
                                                                  │    Eye Contact /     │
                                                                  │  Looking Away flag   │
                                                                  └──────────┬───────────┘
                                                                             ▼
                                                                  ┌──────────────────────┐
                                                                  │     Event Logger     │
                                                                  │  → session_log.json  │
                                                                  └──────────┬───────────┘
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

**~394K parameters total.** No head pose input — checked for correlation with gaze during EDA, found none (correlation ≈ 0.04–0.05), dropped it to keep the model simpler and faster.

---

## Data Pipeline

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
│   → 97.9% retained   │  removed soft images + a corruption pocket
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Bias / leakage check │  no participant-ID leakage
│ → head pose dropped  │  head pose uncorrelated with gaze
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Participant-level   │  train: 12 participants (397,739)
│ train/val/test split │  val:   2 participants (18,496)
│  (no frame leakage)  │  test:  1 participant (2,058)
└──────────┬───────────┘
           ▼
GazeCNN training
```

---

## Data

Used [MPIIGaze](https://perceptualui.org/research/datasets/MPIIGaze/) — 213,658 real webcam images from 15 people, collected over months of normal laptop use, each labeled with exact gaze direction (participants looked at a sequence of on-screen dots and confirmed each one). CC BY-NC-SA 4.0 licensed, not redistributed here.

Data comes as nested MATLAB `.mat` files — wrote a loader to pull out eye images, 3D gaze vectors, and head pose per participant per day, then converted the 3D vectors into two simple angles ($\theta, \phi$) using the standard conversion from the dataset's docs.

### Caught a Labeling Bug Before It Mattered

Before trusting any of it, ran a sanity check: does the "ground truth" gaze angle actually match where the pupil sits in the image? Measured pupil position directly (darkest region in the crop) and compared against computed gaze angle across thousands of samples, grouped into clear "looking left" vs "looking right" cohorts.

The horizontal angle was inverted — a sign bug in the angle conversion. Confirmed it wasn't noise (grouped comparison showed a large, consistent, wrong-direction gap), fixed it with a sign flip, re-verified the fix actually corrected it. Would've been an invisible bug otherwise — training on mirrored labels produces a model that's confidently wrong about left vs right, and nothing about the loss curve would have told us.

### Cleaned It Up

Measured blur (Laplacian variance) across all 427K samples. Found the expected soft/blurry tail at the low end, and — more useful — a pocket of genuinely corrupted images (sensor noise, not just blur) sitting at the *high* end of the blur score, concentrated in two participants. Filtered ~2% off both ends.

Checked for shortcuts the model could exploit instead of learning real gaze cues: whether gaze patterns differed enough per-person that the model could just be recognizing the person (no meaningful leak found), and whether head pose predicts gaze direction (essentially uncorrelated, so dropped head pose as a model input entirely).

Split into train/val/test **by participant**, not by random frame shuffling — otherwise validation numbers would be inflated by near-duplicate frames of the same person leaking across the split.

---

## Model Results

Trained on Colab (free GPU, 20 epochs, a few minutes). Watched for overfitting — val loss stalled around epoch 6 while train loss kept dropping — and kept that earlier checkpoint instead of the final one.

> **Result: 3.58° mean angular error on a fully held-out test participant**, never touched during training or model selection. Cross-person MPIIGaze benchmarks in the literature typically land around 4.5–6° without personal calibration, so this is a solid result for a from-scratch first pass. Sanity-checked visually too — plotted predicted vs. true gaze as arrows on test samples, consistently close with no systematic directional bias.

---

## Live App Flow

```text
Launch app
  │
  ▼
Camera opens, live preview shown
  │
  ▼
Press SPACE
  │
  ▼
┌───────────────────────────────────┐
│     Calibration (x3 rounds)       │
│    5-4-3-2-1 countdown + beep     │
│   → 2s "look at camera" capture   │
│   → beep (round done)             │
│   ... repeat x3, both eyes        │
│   → median baseline computed      │
└──────────────┬────────────────────┘
               ▼
Live monitoring begins
  │
  ▼
┌─────────────────────────┐
│       Per-frame:        │
│   predict → compare to  │
│   baseline → smooth →   │
│   debounce → flag       │
└───────────┬─────────────┘
            ▼
On state change (contact ↔ away):
log event → session_log.json
```

- **Live eye extraction** — MediaPipe pulling eye crops from an actual webcam feed, matched to the training format.
- **Per-session calibration** — the model's raw output space doesn't know your specific camera position, so each run starts with a quick calibration: countdown + beep, a few seconds of "look directly at the camera," repeated 3 rounds, both eyes calibrated separately, median taken as the personal baseline.
- **Temporal smoothing** — rolling average + a debounce requiring several consecutive off-camera frames before flagging, so normal eye jitter doesn't trigger constant false "looking away" flickers.
- **Event logging** — not per-frame (too noisy, too much data), but per-event: every look-away gets a start time, end time, duration, and peak deviation, written to `session_log.json`.

---

## Known Limitations

- **Works well for subtle drift with the head roughly forward** — checking notes just off-camera, glancing at a second screen. This is the realistic interview-proctoring case.
- **Degrades on large head rotation.** MPIIGaze's data is mostly eye movement with the head relatively still, so the model hasn't really seen extreme head-turned poses. Tested this directly — accuracy falls off noticeably when the face turns toward a far corner. Next step is likely a head-pose gate: detect large rotation directly from landmarks and flag "face turned away" without relying on the gaze model in that regime.
- **Runs as its own OpenCV window** right now, not a background overlay — fine for testing and demoing, not yet the "runs invisibly during a real interview call" experience.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Rajneessh/EyeContactTracker.git
cd EyeContactTracker

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows (PowerShell / Command Prompt):
# .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run live eye contact tracker
python app/live_eye_contact.py
```

Camera opens $\rightarrow$ press **SPACE** $\rightarrow$ calibration (countdown + beeps, look at camera on each round) $\rightarrow$ live monitoring starts. Press `q` to quit.

---

## Repo Structure

| Directory / File | Description |
| :--- | :--- |
| `notebooks/` | Full data pipeline + training, in order, with outputs (Colab notebook) |
| `app/` | Live calibration, detection, and logging code |
| `models/` | Contains `best_gaze_model.pt` (~1.5MB pretrained PyTorch model) |
| `training/` | Reserved for standalone training scripts (pipeline currently lives in notebook) |

---

## Roadmap

- [ ] Head-pose gating for large rotations (the main accuracy gap right now)
- [ ] Always-on-top minimal overlay instead of a full OpenCV window
- [ ] Session summary view — % eye contact, break count/duration, pulled from `session_log.json`
- [ ] Config file instead of hardcoded thresholds

---

## Credits

Training data: **MPIIGaze** (Zhang et al.), CC BY-NC-SA 4.0