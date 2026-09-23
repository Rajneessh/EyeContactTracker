# EyeContactTracker

A real-time eye-contact detector for video calls. It watches your webcam, figures out if you're actually looking at the camera or have drifted off to something else on screen, and quietly logs when that happens — so you can review it later.

This was built from scratch: no pretrained gaze models, no off-the-shelf eye-tracking SDKs. Just a public research dataset, a CNN trained from zero, and a lot of debugging along the way.

## Why this exists

During video calls it's easy to lose track of whether you're actually making eye contact or just staring at your own reflection, notifications, or a second monitor. This tool tries to catch that in real time and keep a record of it, so you can see patterns after the fact — was I looking away a lot in that meeting, and when?

## How it works, at a high level

1. **MediaPipe Face Mesh** finds your face and eye landmarks live from the webcam, frame by frame.
2. Both eye regions get cropped out and resized to small grayscale patches.
3. A **custom-trained CNN** looks at each eye patch and predicts a gaze direction (two angles — up/down and left/right).
4. At the start of each session, a quick **calibration** step figures out what "looking at the camera" actually looks like for you specifically — this matters because everyone's webcam sits in a slightly different spot relative to their eyes.
5. Every frame, the live prediction is compared against your calibrated baseline. If you drift too far for too long, it's logged as a "looking away" event.
6. A rolling average smooths out normal eye jitter so it doesn't flag every tiny movement as "you looked away."

## The journey

### Finding and understanding the data

We used **MPIIGaze**, a research dataset of 213,658 real webcam images from 15 people, collected over several months of normal laptop use. Each image is a tightly cropped eye patch, labeled with exactly where that person was looking at the time (participants had to look at a sequence of on-screen dots and confirm each one, so the labels are genuinely accurate, not guessed).

It's licensed CC BY-NC-SA — free to use and build on, not for commercial use, and attribution is required if this work is ever published or shared publicly.

### Getting it into a usable shape

The raw data ships as MATLAB `.mat` files with a fairly awkward nested structure. We wrote a loader to unpack it into plain arrays: for every eye, an image, a 3D gaze direction vector, and a head pose vector. The 3D gaze vectors got converted into two simple angles (theta = vertical, phi = horizontal) using the standard formula from the dataset's own documentation.

### Catching a labeling bug before it mattered

Before trusting any of this data for training, we ran a sanity check: does the model's "ground truth" angle actually match where the pupil is sitting in the image? We measured pupil position directly (via pixel darkness) and compared it against the calculated gaze angle across thousands of samples.

It didn't match. The horizontal angle was inverted — a sign error in the angle conversion. We confirmed this wasn't just noise by grouping samples into "clearly looking left" vs "clearly looking right" and checking that pupil position really did differ between the groups (it did, just in the opposite direction expected). Fixed with a single sign flip, then re-verified the fix actually corrected it.

This was probably the single most important step in the whole project — training on mirrored labels would have quietly produced a model that gets left and right backwards, and that kind of bug is invisible unless you go looking for it.

### Cleaning up the data

213k images collected over months on people's personal laptops inevitably includes some junk. We measured image sharpness (a standard blur-detection technique) and found two problems: some images were just too soft to be useful, and — more interestingly — a pocket of genuinely corrupted images (looked like sensor noise or a scanning glitch) hiding in two participants' data. Filtered out the worst ~2% at both ends, keeping the rest.

We also checked that the data didn't have any obvious “shortcuts” a model could exploit instead of actually learning to read eyes — like whether gaze patterns differed so much per-person that the model could cheat by recognizing the person rather than reading the eye. It didn't. We also checked whether head pose (which way someone's head is turned) predicts gaze direction — it turned out to be basically uncorrelated in this dataset, so we simplified the model to just look at the eye itself, no head pose needed.

Finally, the data was split into training, validation, and test sets **by person**, not by randomly shuffling images — otherwise the model could end up being tested on near-duplicate images of someone it already trained on, which would make it look more accurate than it really is.

### Training the model

A small convolutional neural network (roughly 394,000 parameters — tiny by modern standards, which matters because this needs to run in real time on a regular laptop) trained on the cleaned eye-patch data. Watched for overfitting — a point partway through training where the model starts memorizing quirks of the training data rather than learning anything generalizable — and kept the checkpoint from right before that happened, rather than just the final one.

**Result: 3.58° average error on a completely held-out person the model never saw during training or tuning.** For context, published research results on this same dataset (models tested on unseen people, same setup) typically land in the 4.5-6° range without any personal calibration — so this is a solid, competitive result for a first pass.

We also did a visual gut-check: plotted the model's predicted gaze direction as an arrow next to the true direction on a batch of test images. They lined up closely and consistently, with no obvious systematic bias in any direction — just small, expected noise.

### From a trained model to something that actually runs live

Training happened on Google Colab (free GPU access made this fast — full dataset load plus 20 training epochs took well under 30 minutes total). But a trained model sitting in a notebook doesn't do anything useful on its own — the rest of the work was building a local application around it:

- **Live eye extraction**: wired up MediaPipe Face Mesh to grab eye crops from an actual webcam feed in real time, matching the format the model was trained on.
- **Per-session calibration**: since the training data comes from other people's cameras in other positions, the raw model output on your face doesn't automatically mean "0° = looking at your camera." Each time the app starts, it runs a quick calibration: a countdown with beeps, a few seconds of "look directly at the camera," repeated three times, for both eyes separately. The median of all those readings becomes your personal baseline for that session.
- **Temporal smoothing**: raw per-frame predictions are naturally a little noisy — averaging over a short rolling window, plus requiring several consecutive "off camera" frames before actually flagging it, stops the system from flickering between "eye contact" and "looking away" on every tiny eye movement.
- **Event logging**: rather than recording every single frame (which would be an enormous, useless amount of data), the app logs discrete events — every time you look away, it records when it started, when it ended, how long it lasted, and how far off you were at the peak. Saved as a clean, readable JSON file you can look back through after a call.

## What's in this repo
notebooks/ The full Colab notebook — data loading, the label-bug investigation, quality
filtering, EDA, and model training, all in order with outputs.
app/ The live application — calibration, real-time detection, and logging.
models/ The trained model checkpoint (best_gaze_model.pt, ~1.5MB).
training/ Reserved for standalone training scripts (currently the pipeline lives in
the notebook).


## Current status

Working end to end: launches, calibrates against your specific camera setup, tracks eye contact live, and logs break events to a session file. Right now it runs as its own visible window rather than a background overlay you'd keep open during an actual call — that's the next piece being worked on, along with a simple after-the-fact summary view of each session's log (percentage of time in eye contact, number and length of look-away events, etc).

## Setup

```bash
git clone https://github.com/Rajneessh/EyeContactTracker.git
cd EyeContactTracker
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app/live_eye_contact.py
```

## Credits

Training data: [MPIIGaze](https://perceptualui.org/research/datasets/MPIIGaze/), Zhang et al. — licensed CC BY-NC-SA 4.0. Not redistributed in this repo; see the notebook for how to obtain and preprocess it.