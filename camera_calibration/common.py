"""
common.py
---------
Shared plumbing for the calibration stages: config loading, image discovery,
board detection with a small on-disk cache, and the image-size sanity check
that every stage depends on.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
from collections import Counter

import cv2
import numpy as np
import yaml

import squaregrid as sg

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.yaml")
CACHE_DIR = os.path.join(HERE, "output", ".detection_cache")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Object-point unit used throughout. Metres, because the downstream Kinefeet
# pipeline (triangulation, meshes, config.yaml rig numbers) is all in metres.
UNIT_M = sg.PITCH_MM / 1000.0


def load_config(path=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    """Resolve a config path relative to the camera_calibration folder."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(HERE, path))


def list_images(folder):
    folder = resolve(folder)
    files = sorted(glob.glob(os.path.join(folder, "*")))
    return [f for f in files if f.lower().endswith(IMAGE_EXTS)]


# --------------------------------------------------------------------------
# Detection with cache
# --------------------------------------------------------------------------


def _cache_key(path, min_cells):
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{min_cells}|v3"
    return hashlib.sha1(raw.encode()).hexdigest()


def detect_board(path, min_cells=12, use_cache=True):
    """Detect the board in one image file. Returns a ``Detection`` or ``None``."""
    key = _cache_key(path, min_cells)
    cache_file = os.path.join(CACHE_DIR, key + ".pkl")
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    detection = sg.detect(image, unit_mm=UNIT_M, min_cells=min_cells) if image is not None else None

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(detection, f)
    return detection


def detect_folder(folder, min_cells=12, use_cache=True, verbose=True):
    """Detect the board in every image of a folder.

    Returns ``(detections, paths)`` for the images where the board was found.
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images in {resolve(folder)}")

    detections, kept = [], []
    for path in paths:
        detection = detect_board(path, min_cells=min_cells, use_cache=use_cache)
        name = os.path.basename(path)
        if detection is None:
            if verbose:
                print(f"    board NOT found  {name}")
            continue
        detections.append(detection)
        kept.append(path)
        if verbose:
            n = detection.debug["n_cells"]
            flag = "  [MIRRORED]" if detection.mirrored else ""
            print(f"    found {n:2d}/24 squares  {name}{flag}")
    return detections, kept


# --------------------------------------------------------------------------
# Image size handling
# --------------------------------------------------------------------------


def check_image_sizes(detections, paths, tolerance=0.01):
    """Split detections into those matching the modal image size and the rest.

    Intrinsics are meaningless across mixed image sizes: ``cx``/``cy`` are
    pixel coordinates in one specific frame, and ``fx``/``fy`` scale with
    resolution. A folder of differently-sized screenshots cannot be calibrated
    as one camera, so we pick the modal size and report everything else.

    Returns ``(size, keep_idx, reject_idx)`` where ``size`` is ``(w, h)``.
    """
    sizes = [d.image_size for d in detections]
    modal, _ = Counter(sizes).most_common(1)[0]
    mw, mh = modal

    keep, reject = [], []
    for i, (w, h) in enumerate(sizes):
        if abs(w - mw) <= tolerance * mw and abs(h - mh) <= tolerance * mh:
            keep.append(i)
        else:
            reject.append(i)
    return modal, keep, reject


def report_image_sizes(detections, paths):
    """Print a size histogram and warn when the set is not uniform."""
    counts = Counter(d.image_size for d in detections)
    if len(counts) == 1:
        (w, h), n = next(iter(counts.items()))
        print(f"  image size: {w}x{h} for all {n} images")
        return True

    print("  [!] images are NOT all the same size:")
    for (w, h), n in counts.most_common():
        print(f"        {w}x{h}  ({n} image{'s' if n != 1 else ''})")
    print(
        "      Intrinsics are tied to one exact frame geometry, so cropped or\n"
        "      rescaled frames cannot share a single K. Re-export every frame\n"
        "      straight from the camera at its native resolution, uncropped."
    )
    return False


# --------------------------------------------------------------------------
# Point extraction
# --------------------------------------------------------------------------


def points_from_detection(detection, use_corners=True):
    """(object_points Nx3 float32, image_points Nx2 float32) for one view.

    With ``use_corners`` each square contributes its 4 corners (up to 96 points
    per view); otherwise only the 24 square centres are used.
    """
    if use_corners:
        obj = sg.object_corners(UNIT_M).reshape(-1, 3)
        img = detection.corners.reshape(-1, 2)
    else:
        obj = sg.object_centres(UNIT_M)
        img = detection.centres

    good = ~np.isnan(img[:, 0])
    return (
        np.ascontiguousarray(obj[good], dtype=np.float32),
        np.ascontiguousarray(img[good], dtype=np.float32),
    )


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------


def planar_residual_px(detection, use_corners=False):
    """RMS residual of a single-view homography fit, in pixels.

    The board is flat, so one homography explains a view *exactly* up to lens
    distortion and detection noise - no camera pose or intrinsics involved.
    A view that fits badly is telling you the physical board was bowed, or the
    frame was blurred/torn, and including it will drag the whole calibration
    off. This is the cheapest pre-flight check there is.
    """
    obj, img = points_from_detection(detection, use_corners)
    if len(obj) < 4:
        return float("inf")
    h, _ = cv2.findHomography(obj[:, :2].astype(np.float64), img.astype(np.float64), 0)
    if h is None:
        return float("inf")
    projected = cv2.perspectiveTransform(
        obj[:, :2].reshape(-1, 1, 2).astype(np.float64), h
    ).reshape(-1, 2)
    return float(np.sqrt((np.linalg.norm(projected - img, axis=1) ** 2).mean()))


def focal_from_board(detection):
    """Focal length implied by the board's own perspective in one image, in px.

    Zhang's single-view estimate, assuming square pixels and the principal point
    at the image centre. A flat target's homography carries 8 degrees of freedom
    against 6 for the pose, so the leftovers say something about the lens - not
    enough to calibrate from, but more than enough to catch intrinsics that
    belong to a *different* set of frames. That failure is invisible in
    reprojection error (a wrong focal is absorbed by placing the board further
    away) and shows up only as every distance in the scene being wrong by a
    constant factor, so it is worth checking explicitly.

    Returns the estimate, or ``None`` when the view is too fronto-parallel to
    say anything - a homography with no perspective in it constrains no focal.
    """
    obj, img = points_from_detection(detection, use_corners=False)
    if len(obj) < 6:
        return None
    h, _ = cv2.findHomography(obj[:, :2].astype(np.float64), img.astype(np.float64), 0)
    if h is None:
        return None

    width, height = detection.image_size
    centred = np.array([[1, 0, -width / 2.0], [0, 1, -height / 2.0], [0, 0, 1]]) @ h
    h1, h2 = centred[:, 0], centred[:, 1]

    estimates = []
    denominator = h1[2] * h2[2]
    if abs(denominator) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / denominator
        if f2 > 0:
            estimates.append(np.sqrt(f2))
    denominator = h1[2] ** 2 - h2[2] ** 2
    if abs(denominator) > 1e-12:
        f2 = ((h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)) / denominator
        if f2 > 0:
            estimates.append(np.sqrt(f2))
    return float(np.median(estimates)) if estimates else None


def rescale_intrinsics(K, calib_size, image_size):
    """Move a calibration from the resolution it was measured at to another.

    Exact for a pure resize of the same framing, and nothing else: a crop moves
    the principal point by an unknown amount and a digital zoom changes the
    field of view, neither of which any rescaling can recover.
    """
    sx = image_size[0] / calib_size[0]
    sy = image_size[1] / calib_size[1]
    out = K.copy()
    out[0, :] *= sx
    out[1, :] *= sy
    return out, (sx, sy)


def draw_reprojection(image, observed, projected, header_lines):
    """Overlay observed vs. reprojected points on an image, so calibration
    accuracy can be checked by eye instead of by reading numbers out of JSON.

    Green circles are where the board detector actually found each point;
    red crosses are where the fitted camera model predicts that same point
    should land; the yellow line between them is the reprojection error for
    that point. ``header_lines`` (a list of strings, e.g. camera name and
    RMS/max error) is drawn as a text panel in the top-left corner.
    """
    out = image.copy()
    for (ox, oy), (px, py) in zip(observed, projected):
        o = (int(round(ox)), int(round(oy)))
        p = (int(round(px)), int(round(py)))
        cv2.line(out, o, p, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, o, 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(out, p, (0, 0, 255), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)

    if header_lines:
        pad, line_h = 10, 24
        box_w = max(260, 12 * max(len(line) for line in header_lines))
        box_h = pad * 2 + line_h * len(header_lines)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
        for i, line in enumerate(header_lines):
            y = pad + line_h * i + 18
            cv2.putText(out, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved -> {os.path.relpath(path, HERE)}")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
