# INF2009 Image & Video Analytics Lab (Raspberry Pi 5)

## 1. Overview

This lab introduces **image analytics** and **video analytics** on an edge device using the Raspberry Pi 5. You will progressively move from processing **single images** to analysing **continuous video streams**, mirroring how real-world edge vision systems are designed.

This lab is intentionally structured to be **highly guided** and **incremental**, similar to the Sound Analytics lab. Do not skip sections.

---

## 2. Learning Objectives

By the end of this lab, students should be able to:

* Capture images and video from a camera on Raspberry Pi 5
* Understand the difference between image-based and video-based analytics
* Apply basic OpenCV operations (colour conversion, thresholding, edge detection)
* Perform simple object and motion analysis on video streams
* Reason about computational cost and frame-rate constraints on edge devices

---

## 3. Hardware and Software Requirements

### Hardware

* Raspberry Pi 5
* USB webcam or Raspberry Pi Camera Module
* Keyboard, mouse, HDMI display

### Software

* Raspberry Pi OS (64-bit recommended)
* Python 3.9+
* OpenCV (cv2)

---

## 4. Environment Setup

This section assumes **no prior completion of the Sound Analytics lab**. Follow all steps exactly.

---

### 4.1 System Update

```bash
sudo apt update
sudo apt upgrade -y
```

---

### 4.2 Install System Dependencies

```bash
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  libcamera-dev
```

---

### 4.3 Python Environment Setup (Isolated Lab Environment)

This lab uses a **Python virtual environment**, but with a **descriptive, lab-specific name** instead of the generic `venv`.

Rationale (read this):

* Keeps all INF2009 dependencies isolated from the OS
* Avoids conflicts between image, video, and sound analytics labs
* Makes it explicit which environment belongs to which lab

---

### 4.4 Create the INF2009 Image/Video Analytics Environment

```bash
mkdir -p ~/inf2009
cd ~/inf2009

python3 -m venv imgvid_env
source imgvid_env/bin/activate
```

You should now see `(imgvid_env)` in your terminal prompt.

---

### 4.5 Install Required Python Packages

```bash
pip install --upgrade pip
pip install numpy opencv-python
```

Verify installation:

```bash
python - << EOF
import cv2
import numpy
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
EOF
```

If no errors appear, the Python environment is correctly set up.

---

## 5. Conceptual Background (Read This First)

### Image vs Video Analytics

| Image Analytics                          | Video Analytics                         |
| ---------------------------------------- | --------------------------------------- |
| Operates on a single frame               | Operates on a sequence of frames        |
| Stateless                                | Temporal / stateful                     |
| Lower compute cost                       | Higher compute cost                     |
| Examples: edge detection, face detection | Examples: motion tracking, optical flow |

Key idea: **Video analytics is image analytics + time**.

---

## 6. Part A – Image Analytics

In this section, you will process **single images** to extract useful information. The key to avoiding confusion is that you will work from a **single starter script** and progressively add small blocks at clearly marked locations.

---

### 6.1 Create the Starter Script (image_lab.py)

Create a file named `image_lab.py` and paste **this entire starter code** first:

```python
import cv2

# ==============================
# INF2009 Image Analytics Lab
# File: image_lab.py
# ==============================

# STEP 1: Camera capture (do not modify yet)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to capture image from camera")

cv2.imwrite("capture.jpg", frame)
print("[OK] Saved capture.jpg")

# STEP 2: Add HSV conversion here (you will add code below this line)
# --- ADD STEP 2 CODE BELOW ---


# STEP 3: Add colour segmentation here
# --- ADD STEP 3 CODE BELOW ---


# STEP 4: Add morphological cleanup here
# --- ADD STEP 4 CODE BELOW ---


# STEP 5: Add contour detection + drawing here
# --- ADD STEP 5 CODE BELOW ---


# STEP 6: Add feature extraction + printing here
# --- ADD STEP 6 CODE BELOW ---


print("[DONE] Image analytics pipeline complete")
```

Run it once:

```bash
python image_lab.py
```

Checkpoint:

* `capture.jpg` is created
* No Python errors

---

### 6.2 STEP 2 — Colour Space Conversion (BGR → HSV)

**Where to put this:** in `image_lab.py`, directly under:
`# --- ADD STEP 2 CODE BELOW ---`

Add this:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

Explanation:

* OpenCV frames from cameras arrive in **BGR** by default
* Colour segmentation is more stable in **HSV** (separates colour from brightness)

Checkpoint:

* Script still runs without errors

---

### 6.3 STEP 3 — Colour Segmentation

**Where to put this:** under `# --- ADD STEP 3 CODE BELOW ---`

Add this example (segment **blue** objects):

```python
lower_blue = (100, 150, 50)
upper_blue = (140, 255, 255)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imwrite("mask.jpg", mask)
print("[OK] Saved mask.jpg")
```

Explanation (bit by bit):

* `lower_blue` / `upper_blue` define the HSV range you consider “blue”
* `inRange()` outputs a **binary mask**: white = match, black = no match

Checkpoint:

* `mask.jpg` is created
* White regions correspond roughly to blue objects in the scene

---

### 6.4 STEP 4 — Noise Reduction (Morphological Operations)

**Where to put this:** under `# --- ADD STEP 4 CODE BELOW ---`

Add this:

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imwrite("clean_mask.jpg", clean)
print("[OK] Saved clean_mask.jpg")
```

Explanation:

* Real masks contain speckles and holes
* `MORPH_OPEN` = erosion then dilation → removes small noise

Checkpoint:

* `clean_mask.jpg` has fewer speckles than `mask.jpg`

---

### 6.5 STEP 5 — Contour Detection and Bounding Boxes

**Where to put this:** under `# --- ADD STEP 5 CODE BELOW ---`

Add this:

```python
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 500:  # ignore tiny blobs
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite("contours.jpg", frame)
print("[OK] Saved contours.jpg")
```

Explanation:

* `findContours()` finds connected “islands” of white pixels
* `boundingRect()` converts each island to a rectangle (useful for tracking)

Checkpoint:

* `contours.jpg` shows green rectangles around the segmented object(s)

---

### 6.6 STEP 6 — Feature Extraction (Area, Bounding Box)

**Where to put this:** under `# --- ADD STEP 6 CODE BELOW ---`

Add this:

```python
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        print(f"Feature: area={area:.1f}, bbox=({x},{y},{w},{h}), center=({cx},{cy})")
```

Explanation:

* These numbers are “features” that rules or ML models can consume
* The centroid `(cx, cy)` is a common tracking feature

Checkpoint:

* Terminal prints feature lines

---

## 7. Part B – Video Analytics

In this section you will reuse the same ideas from Part A, but apply them **continuously per frame**.

---

### 7.1 Create the Starter Script (video_lab.py)

Create a file named `video_lab.py` and paste **this entire starter code** first:

```python
import cv2

# ==============================
# INF2009 Video Analytics Lab
# File: video_lab.py
# ==============================

cap = cv2.VideoCapture(0)

# STEP 1: Add any one-time initialisation here
# --- ADD STEP 1 CODE BELOW ---


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # STEP 2: Add per-frame preprocessing here (e.g., HSV conversion)
    # --- ADD STEP 2 CODE BELOW ---


    # STEP 3: Add per-frame segmentation here
    # --- ADD STEP 3 CODE BELOW ---


    # STEP 4: Add tracking / visualisation here
    # --- ADD STEP 4 CODE BELOW ---


    cv2.imshow("Video Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Video analytics pipeline complete")
```

Run it once:

```bash
python video_lab.py
```

Checkpoint:

* Live feed appears
* Press `q` to quit

---

### 7.2 STEP 2 — Per-frame HSV Conversion

**Where to put this:** under `# --- ADD STEP 2 CODE BELOW ---`

Add this:

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

Explanation:

* Same as Part A, but repeated every frame

---

### 7.3 STEP 3 — Per-frame Colour Segmentation

**Where to put this:** under `# --- ADD STEP 3 CODE BELOW ---`

Add this (reuse the same HSV thresholds from Part A):

```python
lower_blue = (100, 150, 50)
upper_blue = (140, 255, 255)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

Explanation:

* Produces a mask per frame

---

### 7.4 STEP 4 — Colour-based Object Tracking (Contours + Centre)

**Where to put this:** under `# --- ADD STEP 4 CODE BELOW ---`

Add this:

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"area={area:.0f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
```

Explanation:

* Bounding box and centroid are the simplest form of “tracking”
* You are not predicting motion yet; you are re-detecting each frame

Checkpoint:

* Box and centroid follow the coloured object

---

### 7.5 Motion Detection (Frame Differencing)

This requires **state** (the previous frame). You will add one variable outside the loop, then add the processing inside the loop.

**Step 7.5a — Add initialisation**

Where to put this: under `# --- ADD STEP 1 CODE BELOW ---`

```python
prev_gray = None
```

**Step 7.5b — Add processing**

Where to put this: under `# --- ADD STEP 2 CODE BELOW ---` (after HSV conversion is fine, but this uses grayscale)

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if prev_gray is not None:
    diff = cv2.absdiff(prev_gray, gray)
    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow("Motion", motion)

prev_gray = gray
```

Explanation:

* `prev_gray` is your memory of the previous frame
* Big differences between frames imply motion

Checkpoint:

* A second window appears showing moving regions

---

### 7.6 Optical Flow (Advanced)

Optical flow estimates motion vectors, not just “changed pixels”. It is heavier and may reduce FPS.

**Step 7.6a — Add initialisation**

Where to put this: under `# --- ADD STEP 1 CODE BELOW ---` (if you are doing both motion and optical flow, keep them separate variables)

```python
prev_flow_gray = None
```

**Step 7.6b — Add processing**

Where to put this: under `# --- ADD STEP 2 CODE BELOW ---`

```python
gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if prev_flow_gray is not None:
    flow = cv2.calcOpticalFlowFarneback(prev_flow_gray, gray2, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag_norm = (mag / (mag.max() + 1e-6))
    cv2.imshow("Optical Flow Magnitude", mag_norm)

prev_flow_gray = gray2
```

Explanation:

* `calcOpticalFlowFarneback()` estimates motion between consecutive frames
* Magnitude image shows “how much motion” occurred

Checkpoint:

* Moving objects produce brighter regions in the magnitude view

---

## 8. Performance Discussion (Critical Thinking)

Answer the following:

1. Why does video analytics consume more CPU than image analytics?
2. What happens if frame resolution increases?
3. Why is edge processing preferred over cloud for real-time analytics?

---

## 9. Common Pitfalls

* Forgetting to release the camera resource
* Running multiple camera programs simultaneously
* Using high resolutions unnecessarily

---

## 10. Extensions (Optional)

These extensions are **not required** to pass the lab. They are intended for students who finish early and want to explore more realistic, model-based analytics.

---

### 10.1 Measure FPS (Frames Per Second)

Add this to `video_lab.py`:

**Where to put this (initialisation):** under `# --- ADD STEP 1 CODE BELOW ---`

```python
import time
frame_count = 0
start_time = time.time()
```

**Where to put this (inside the while-loop, near the end):** just before `cv2.imshow("Video Analytics", frame)`

```python
frame_count += 1
elapsed = time.time() - start_time
if elapsed >= 2.0:
    fps = frame_count / elapsed
    print(f"[FPS] {fps:.1f}")
    frame_count = 0
    start_time = time.time()
```

Checkpoint:

* Terminal prints FPS every ~2 seconds

---

### 10.2 Lower Resolution for Better Performance

**Where to put this:** right after `cap = cv2.VideoCapture(0)`

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

Checkpoint:

* FPS increases compared to default resolution

---

### 10.3 Advanced Extension: MediaPipe (Pose / Face / Object Detection)

⚠ **Not required.** This section introduces pretrained model pipelines. You will run **one** of the following (or all, if you have time):

* **Pose** (body keypoints)
* **Face mesh** (dense facial landmarks)
* **Object detection** (bounding boxes + labels)

The goal is not “wow AI”. The goal is to compare:

* classical pipeline (HSV/contours/features)
  vs
* model pipeline (pretrained landmarks/detections)

---

#### 10.3.1 Install MediaPipe (inside `imgvid_env`)

Make sure you are in the environment:

```bash
cd ~/inf2009
source imgvid_env/bin/activate
```

Install:

```bash
pip install mediapipe
```

If this step actually worked, then congratulations. You have just accomplished something borderline miraculous. Please take a moment to appreciate your own genius. Why? Because the latest Mediapipe is not available on the RPi5 OS with the newest Python version.

But do not worry. Mediapipe does behave properly on x86 machines. So if your Raspberry Pi adventure ends in frustration, try installing it on your PC or laptop instead. Those devices are far less judgmental. 

Thus, subsequent steps will only work on an x86 until someone builds the model for ARM platforms. Whoever that hero is, we salute them in advance.

Quick import test:

```bash
python -c "import mediapipe as mp; print('mediapipe OK')"
```

---

#### 10.3.2 MediaPipe Pose (Pose Estimation)

Create `mediapipe_pose.py` and paste this full script:

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

Run:

```bash
python mediapipe_pose.py
```

Checkpoint:

* Skeleton landmarks appear on your body

What this gives you:

* 33 pose landmarks (a structured feature set)

---

#### 10.3.3 MediaPipe Face Mesh (Facial Landmarks)

Create `mediapipe_face_mesh.py` and paste this full script:

```python
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )

        cv2.imshow("MediaPipe Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

Run:

```bash
python mediapipe_face_mesh.py
```

Checkpoint:

* Face mesh overlay appears on your face

What this gives you:

* 468 face landmarks (dense geometry features)

---

#### 10.3.4 MediaPipe Object Detection (Bounding Boxes + Labels)

There are two common ways to do object detection with MediaPipe:

1. **MediaPipe Tasks API** (recommended conceptually, but needs a model file)
2. Alternative libraries (not covered here)

We will do (1).

**Step A — Download a lightweight object detector model**

Create a folder for models:

```bash
mkdir -p models
```

Download an EfficientDet-Lite model (TFLite). Use the provided class link (or your TA-provided link) and save it as:

* `models/efficientdet_lite0.tflite`

(If your class does not provide a model link, skip object detection and do Pose/Face Mesh instead.)

**Step B — Create the script**

Create `mediapipe_object_detector.py` and paste this full script:

```python
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/efficientdet_lite0.tflite"

# Create detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5
)

detector = vision.ObjectDetector.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe Tasks expects RGB image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp_image = mp_image = None
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    # Draw detections
    for det in result.detections:
        bbox = det.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        label = det.categories[0].category_name if det.categories else "object"
        score = det.categories[0].score if det.categories else 0.0

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("MediaPipe Object Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run:

```bash
python mediapipe_object_detector.py
```

Checkpoint:

* Boxes and labels appear on detected objects

Important note:

* If the model file is missing, this script will fail immediately.

---

#### 10.3.5 What MediaPipe Is Doing (and what it hides)

Model pipeline (high level):

* Captures frame
* Runs a pretrained model
* Outputs structured results (landmarks / boxes / labels)

What it **hides** from you:

* how features were chosen
* how the model was trained
* why it fails on some poses/lighting/occlusions
* the real compute/memory costs until you measure them

---

#### 10.3.6 Required Comparison (Write-up)

Compare your **classical pipeline** vs **one MediaPipe pipeline** (Pose OR Face Mesh OR Object Detection):

1. **Robustness:** which survives lighting changes better?
2. **Compute cost:** which has higher CPU usage / lower FPS?
3. **Explainability:** which is easier to debug?
4. **Failure cases:** show one failure example for each.

Write a clear trade-off statement:

* Classical methods: simpler, explainable, requires tuning
* Model methods: more capable, less explainable, heavier dependencies

---

## 11. Submission Checklist

* Screenshots of image outputs
* Short answers to performance questions
* Annotated code snippets

---

## 12. Key Takeaway

Edge vision systems must balance **accuracy**, **latency**, and **compute cost**. Image analytics builds the foundation; video analytics introduces real-world constraints.

Do not treat them as separate topics—they are part of the same pipeline.
