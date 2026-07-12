"""
03_setup_camera_rig.py
------------------------
Turns the rig measurements in config.yaml (radius / camera height / target
height / the 8 shooting angles) into a full camera projection matrix P for
every photo, and saves them to output/<version>/cameras.json.

This is the step that encodes WHERE each of your 8 photos was taken from.
It assumes the "orbit rig" capture method described in the README: you
(or 8 fixed cameras) shot from 8 roughly evenly-spaced angles around the
subject, all at the same distance and height, all aimed at the same point
on the subject's body.

If you calibrated your camera with 01_calibrate_camera.py, that K and
lens distortion are reused for every camera here (same physical camera).
Otherwise an approximate K is built from a guessed field-of-view.

Run this once per version (full_body / waist_down) since they typically
use different radius/height (waist_down is usually shot closer / lower).
"""

import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils import build_intrinsics, build_extrinsics, projection_matrix

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")
INTRINSICS_PATH = os.path.join(OUTPUT_ROOT, "camera_intrinsics.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_image_size(version, pose_name):
    """Peek at one real photo to know the pixel resolution for K's cx,cy."""
    img_dir = os.path.join(DATA_ROOT, version, pose_name)
    files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        return None
    import cv2
    img = cv2.imread(files[0])
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def build_K(cfg, image_width, image_height):
    if os.path.exists(INTRINSICS_PATH):
        with open(INTRINSICS_PATH) as f:
            calib = json.load(f)
        print("  using checkerboard-calibrated intrinsics from "
              "output/camera_intrinsics.json")
        # NOTE: if your body photos were taken at a different resolution
        # than the calibration photos, scale K's focal length / center
        # accordingly. This assumes the same resolution for simplicity.
        return calib["K"]
    else:
        print("  no checkerboard calibration found - using approximate "
              "field-of-view intrinsics (see config.yaml -> camera.assumed_fov_deg)")
        K = build_intrinsics(image_width, image_height, cfg["camera"]["assumed_fov_deg"])
        return K.tolist()


def main():
    cfg = load_config()

    for version in ["full_body", "waist_down"]:
        rig = cfg["rig"][version]
        angles = rig["angles_deg"]
        radius = rig["radius_m"]
        cam_height = rig["camera_height_m"]
        target_height = rig["target_height_m"]

        # use any pose folder just to read the photo resolution
        image_size = None
        for pose_name in ["pose1", "pose2", "pose3"]:
            image_size = get_image_size(version, pose_name)
            if image_size:
                break
        if image_size is None:
            print(f"[!] No images found for '{version}' yet - skipping camera setup. "
                  f"Add photos to data/{version}/pose1|2|3/ first.")
            continue

        w, h = image_size
        print(f"\n[{version}]  image size = {w}x{h}, {len(angles)} camera angles: {angles}")
        K = build_K(cfg, w, h)

        cameras = {}
        for i, angle in enumerate(angles):
            R, t, cam_pos = build_extrinsics(angle, radius, cam_height, target_height)
            P = projection_matrix(np.array(K), R, t)
            cameras[f"cam_{i:02d}"] = {
                "angle_deg": angle,
                "image_width": w,
                "image_height": h,
                "K": K,
                "R": R.tolist(),
                "t": t.tolist(),
                "camera_position_world": cam_pos.tolist(),
                "P": P.tolist(),
            }

        os.makedirs(os.path.join(OUTPUT_ROOT, version), exist_ok=True)
        out_path = os.path.join(OUTPUT_ROOT, version, "cameras.json")
        with open(out_path, "w") as f:
            json.dump(cameras, f, indent=2)
        print(f"  saved {len(cameras)} camera matrices -> {out_path}")

    print("\nDone. IMPORTANT: cam_00, cam_01, ... map to your photos in the ORDER "
          "you listed angles_deg in config.yaml AND the alphabetical order of your "
          "image filenames within each pose folder (see README naming convention, "
          "e.g. 01.jpg = 0deg, 02.jpg = 45deg, etc).")


if __name__ == "__main__":
    main()
