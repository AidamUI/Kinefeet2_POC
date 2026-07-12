"""
01_calibrate_camera.py
-----------------------
OPTIONAL BUT RECOMMENDED. Computes your camera's intrinsic matrix (focal
length + optical center + lens distortion) from photos of a checkerboard.

Why bother: triangulation accuracy depends directly on how correct your
camera intrinsics (K) are. Guessing a field-of-view (what happens if you
skip this step) works, but a real calibration is noticeably more accurate
and only takes 5 minutes.

HOW TO SHOOT THE CALIBRATION PHOTOS
  1. Print or display a standard checkerboard pattern (a 9x6-INNER-CORNER
     board is assumed by default - e.g. search "opencv checkerboard
     pattern 9x6" and print it on A4, keep it flat, e.g. taped to a
     clipboard or cardboard).
  2. Using the SAME phone, SAME lens (no zoom change!) and roughly the
     SAME settings you'll use for the actual body photos, take 12-20
     photos of the checkerboard from different angles/distances/tilts,
     making sure the whole board is inside the frame each time.
  3. Put those photos in data/calibration_images/.
  4. Run this script.

OUTPUT
  output/camera_intrinsics.json  containing K (3x3) and distortion
  coefficients. 04_triangulate_3d.py will automatically use this file if
  it exists (see config.yaml -> calibration.use_checkerboard).
"""

import json
import glob
import os
import sys
from typing import Any, cast

try:
    import cv2  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for camera calibration. Install it with "
        "'pip install opencv-python' or install dependencies from "
        "'Claude/mp3d/requirements.txt'."
    ) from exc

try:
    import numpy as np  # pyright: ignore[reportMissingImports]
except ImportError as exc:
    raise ImportError(
        "NumPy is required for camera calibration. Install it with "
        "'pip install numpy' or install dependencies from "
        "'Claude/mp3d/requirements.txt'."
    ) from exc

cv2 = cast(Any, cv2)
np = cast(Any, np)

# --- Configuration -----------------------------------------------------
CHECKERBOARD_INNER_CORNERS = (9, 6)  # (columns, rows) of INNER corners, not squares
SQUARE_SIZE_MM = 25.0                # physical size of one checkerboard square, edit to match your printout
CALIB_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "calibration_images")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "camera_intrinsics.json")


def main():
    if SQUARE_SIZE_MM <= 0:
        raise ValueError("SQUARE_SIZE_MM must be greater than 0.")

    images = sorted(glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.*")))
    images = [f for f in images if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(images) < 5:
        print(f"[!] Found only {len(images)} images in {CALIB_IMAGES_DIR}.")
        print("    Add at least 10-15 checkerboard photos and re-run.")
        print("    Skipping calibration - the triangulation script will fall back")
        print("    to an approximate field-of-view based intrinsic matrix instead.")
        sys.exit(0)

    # 3D coordinates of checkerboard inner corners in the board's own coordinate
    # system (Z=0 plane), e.g. (0,0,0), (1,0,0), (2,0,0) ... scaled by square size.
    cols, rows = CHECKERBOARD_INNER_CORNERS
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []   # 3D points in board space, one array per accepted image
    imgpoints = []   # 2D corner detections in pixel space, one array per accepted image
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    accepted = 0
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            print(f"    corners NOT found: {os.path.basename(path)}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        accepted += 1
        print(f"    corners found:     {os.path.basename(path)}")

    if accepted < 5:
        print(f"[!] Only {accepted} usable images (need >= 5). Calibration aborted.")
        sys.exit(0)

    if image_size is None:
        print("[!] No readable calibration images were found. Calibration aborted.")
        sys.exit(0)

    print(f"\nCalibrating from {accepted} images...")
    ret, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print(f"Re-projection RMS error: {ret:.4f} px (lower is better; <1.0 is great, <2.0 is fine)")
    print(f"K =\n{K}")
    print(f"distortion coeffs = {dist.ravel()}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "image_width": image_size[0],
            "image_height": image_size[1],
            "K": K.tolist(),
            "dist_coeffs": dist.ravel().tolist(),
            "rms_reprojection_error_px": ret,
        }, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
