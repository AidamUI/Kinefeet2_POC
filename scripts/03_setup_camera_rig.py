"""
03_setup_camera_rig.py
------------------------
Turns the rig measurements in config.yaml (radius, camera height, target
height and the shooting angles) into a full camera projection matrix P for
every photo, and saves them to output/<version>/cameras.json.

This is the step that encodes where each photo was taken from. It assumes
the "orbit rig" capture method described in the README: a camera (or several
fixed cameras) shot from evenly-spaced angles around the subject, all at the
same distance and height, all aimed at the same point on the subject's body.

This is a fallback. If camera_calibration/ has already produced a
cameras.json for this version, that file holds the real measured camera
poses and this script leaves it alone: an idealised ring built from typed-in
numbers is always less accurate than a real calibration. Running
camera_calibration/ removes the need for this script entirely.

Run this once per version (full_body / waist_down), since they typically use
different radius and height values (waist_down is usually shot closer and
lower).
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


CALIBRATED_MARKER = "camera_calibration/export_cameras.py"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_calibrated(path):
    """True if this cameras.json came from a real calibration.

    The poses this script builds are an idealised ring derived from numbers
    typed into config.yaml - a placeholder for when nothing better exists. Once
    camera_calibration/ has measured where the cameras actually are, silently
    overwriting that with the placeholder would undo the calibration and leave
    no trace, so refuse instead.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            cameras = json.load(f)
        return any(
            isinstance(cam, dict) and cam.get("source") == CALIBRATED_MARKER
            for cam in cameras.values()
        )
    except (ValueError, OSError, AttributeError):
        return False


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
    print("  no real calibration for this rig - using approximate "
          "field-of-view intrinsics (see config.yaml -> camera.assumed_fov_deg).\n"
          "  Run camera_calibration/ for a real K; see its README.")
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
        if is_calibrated(out_path):
            print(f"  cameras.json here came from camera_calibration/ - keeping it.")
            print(f"    These are the real measured camera poses; the idealised ring")
            print(f"    this script builds from config.yaml would be a downgrade.")
            print(f"    Delete {out_path} first if you really want to replace it.")
            continue
        with open(out_path, "w") as f:
            json.dump(cameras, f, indent=2)
        print(f"  saved {len(cameras)} camera matrices -> {out_path}")

    print("\nDone. cam_00, cam_01, ... map to your photos in the order you listed "
          "angles_deg in config.yaml, matched to the alphabetical order of the "
          "image filenames within each pose folder (see the README naming "
          "convention, e.g. 01.jpg = 0deg, 02.jpg = 45deg, etc).")


if __name__ == "__main__":
    main()
