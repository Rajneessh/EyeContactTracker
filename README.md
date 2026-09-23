# EyeContactTracker

Real-time eye contact detection for video calls. Tracks whether you're actually looking at your webcam or have drifted off, using a CNN trained from scratch on the MPIIGaze dataset.

## What it does

- Detects eye landmarks live via MediaPipe, crops both eyes
- Runs a small CNN to predict gaze direction (theta/phi) per eye
- Calibrates per-session to your specific camera position (3 rounds, both eyes, median baseline)
- Smooths predictions over a rolling window + debounce to avoid flicker
- Logs "looked away" events (start, end, duration, peak deviation) to `session_log.json`

## Model

5-layer CNN, ~394K params, single grayscale eye patch (36x60) as input. No head pose input — checked for correlation with gaze during EDA, found none, dropped it to keep the model simpler and faster.

Trained on MPIIGaze (213K labeled eye images, 15 participants), split by participant to avoid leakage. **3.58° mean angular error on a fully held-out test participant** — in line with published cross-person benchmarks on this dataset (typically 4.5–6° without calibration).

Full training pipeline — data loading, a label sign-bug fix (phi was inverted in the angle conversion, caught via pupil-position correlation check, see notebook), quality filtering, EDA, training — is in `notebooks/preprocessingAndVisualization.ipynb`.

## Known limitations

- Works well for subtle eye drift while facing roughly forward — the realistic case for "looking at chat/second monitor during a call."
- Degrades on large head rotation (turning to look at a far corner/monitor) — training data (MPIIGaze) mostly captures eye movement with the head relatively stable, so extreme head poses are out-of-distribution for the model. On the roadmap to address (either a head-pose gate that flags "face turned away" separately, or fine-tuning on a dataset with more pose variation).
- Runs as its own OpenCV window currently, not a background overlay — fine for testing, not yet a "leave it running during a real Meet call" experience.

## Setup

```bash
git clone https://github.com/Rajneessh/EyeContactTracker.git
cd EyeContactTracker
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/live_eye_contact.py
```

Camera opens, press SPACE to start calibration, follow the countdown/beeps, then live monitoring starts. Press `q` to quit.

## Repo structure
notebooks/ Data pipeline + training (Colab notebook)
app/ Live calibration + detection + logging
models/ best_gaze_model.pt (~1.5MB)
training/ reserved for standalone training scripts


## Roadmap

- [ ] Always-on-top minimal overlay instead of a full OpenCV window
- [ ] Session summary view (% eye contact, break count/duration from `session_log.json`)
- [ ] Head-pose gating for large rotations
- [ ] Config file instead of hardcoded thresholds

## Data

[MPIIGaze](https://perceptualui.org/research/datasets/MPIIGaze/) (Zhang et al.), CC BY-NC-SA 4.0. Not included in this repo.