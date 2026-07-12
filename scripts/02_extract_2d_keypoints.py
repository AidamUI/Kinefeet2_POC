"""
02_extract_2d_keypoints.py
----------------------------
Runs MediaPipe's Pose Landmarker on every photo you captured and saves the
33 body landmarks (pixel x, y + visibility/confidence) to a JSON file next
to each image's output slot.

Works with modern MediaPipe (>=0.10, the "Tasks" API, which needs a
downloaded .task model file - this script downloads it automatically the
first time) as well as older MediaPipe (<0.10, the legacy
`mp.solutions.pose` API) by detecting which one is installed.

INPUT   data/<version>/<pose>/*.jpg   (version = full_body | waist_down)
OUTPUT  output/<version>/<pose>/keypoints_2d/<image_stem>.json
        output/<version>/<pose>/annotated/<image_stem>.jpg   (visual sanity check)

Run this AFTER you've taken all your photos and organized them into the
data/ folders as described in the README.
"""

import glob
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import LANDMARK_NAMES, NUM_LANDMARKS, FULL_BODY_CONNECTIONS

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# "lite" / "full" / "heavy" - heavy is the most accurate and is fine here
# since we're processing photos offline, not doing realtime video.
MODEL_VARIANT = "heavy"
MODEL_URL = (
    f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{MODEL_VARIANT}/float16/latest/pose_landmarker_{MODEL_VARIANT}.task"
)
MODEL_PATH = os.path.join(MODEL_DIR, f"pose_landmarker_{MODEL_VARIANT}.task")

VERSIONS = ["full_body", "waist_down"]
POSES = ["pose1", "pose2", "pose3"]


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading pose landmarker model ({MODEL_VARIANT}) ...")
    print(f"  {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  saved -> {MODEL_PATH}")


def make_detector_new_api():
    """Modern MediaPipe (>=0.10) Tasks API."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    detector = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        # normalized landmarks: x,y in [0,1] relative to image size, z relative depth
        lm = result.pose_landmarks[0]
        return [(p.x, p.y, p.visibility) for p in lm]

    return detect


def make_detector_legacy_api():
    """Older MediaPipe (<0.10) mp.solutions.pose API, kept as a fallback."""
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                         min_detection_confidence=0.5)

    def detect(bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks.landmark
        return [(p.x, p.y, p.visibility) for p in lm]

    return detect


def get_detector():
    try:
        return make_detector_new_api()
    except Exception as e:
        print(f"[i] New MediaPipe Tasks API unavailable ({e}); trying legacy API...")
        return make_detector_legacy_api()


def draw_annotated(image, landmarks_px):
    out = image.copy()
    for a, b in FULL_BODY_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            xa, ya, va = landmarks_px[a]
            xb, yb, vb = landmarks_px[b]
            if va > 0.3 and vb > 0.3:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
    for i, (x, y, v) in enumerate(landmarks_px):
        if v > 0.3:
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    return out


def process_folder(detect_fn, version, pose_name):
    img_dir = os.path.join(DATA_ROOT, version, pose_name)
    images = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    images = [f for f in images if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print(f"  (no images in {img_dir}, skipping)")
        return

    kp_dir = os.path.join(OUTPUT_ROOT, version, pose_name, "keypoints_2d")
    ann_dir = os.path.join(OUTPUT_ROOT, version, pose_name, "annotated")
    os.makedirs(kp_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"  [!] could not read {path}")
            continue
        h, w = img.shape[:2]

        landmarks_norm = detect_fn(img)
        if landmarks_norm is None:
            print(f"  [!] NO POSE DETECTED: {os.path.basename(path)}  "
                  f"(retake this photo, or check lighting/framing)")
            continue

        # Convert normalized [0,1] coords to pixel coords for this image.
        landmarks_px = [(x * w, y * h, v) for (x, y, v) in landmarks_norm]

        record = {
            "image": os.path.basename(path),
            "image_width": w,
            "image_height": h,
            "landmarks": [
                {"index": i, "name": LANDMARK_NAMES[i],
                 "x_px": lx, "y_px": ly, "visibility": lv}
                for i, (lx, ly, lv) in enumerate(landmarks_px)
            ],
        }
        with open(os.path.join(kp_dir, f"{stem}.json"), "w") as f:
            json.dump(record, f, indent=2)

        cv2.imwrite(os.path.join(ann_dir, f"{stem}.jpg"), draw_annotated(img, landmarks_px))
        print(f"  ok: {os.path.basename(path)}  -> {stem}.json")


def main():
    print("Loading MediaPipe Pose Landmarker...")
    detect_fn = get_detector()

    for version in VERSIONS:
        for pose_name in POSES:
            print(f"\n[{version}/{pose_name}]")
            process_folder(detect_fn, version, pose_name)

    print("\nDone. Check the 'annotated' folders in output/ to visually verify "
          "every photo was detected correctly before moving to the next step.")


if __name__ == "__main__":
    main()
