"""
synthetic.py
------------
Renders the square-grid board with a known camera, so the detector and the
calibration can be checked against ground truth rather than against the one set
of photos they were developed on.

The point is to be able to answer "is this tuned to those 40 images, or does it
work?" with a number. Every threshold in squaregrid.py was chosen while looking
at real data; nothing here was, so agreement between the two is evidence and
disagreement is a bug.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import squaregrid as sg  # noqa: E402

# Board colours, roughly matching a real print under indoor light.
INK = (28, 28, 30)
SHEET = (232, 232, 230)
BORDER = (35, 35, 40)
MARKER = {sg.CELL_RED: (42, 44, 188), sg.CELL_GREEN: (66, 148, 74), sg.CELL_BLUE: (188, 92, 48)}

SHEET_W_MM, SHEET_H_MM = 600.0, 400.0

# Metres per pitch, matching common.UNIT_M - render and pose in the same unit.
UNIT_M = sg.PITCH_MM / 1000.0


def board_polygons(unit_mm=sg.PITCH_MM):
    """Sheet outline and every square, as (points_mm, colour) in board coords."""
    scale = unit_mm / sg.PITCH_MM
    half = 0.5 * sg.SQUARE_MM * scale
    # Grid spans 5 and 3 pitches; the sheet is centred on it.
    margin_x = (SHEET_W_MM * scale - (5 * unit_mm + 2 * half)) / 2
    margin_y = (SHEET_H_MM * scale - (3 * unit_mm + 2 * half)) / 2
    x0, y0 = -half - margin_x, -half - margin_y
    x1, y1 = 5 * unit_mm + half + margin_x, 3 * unit_mm + half + margin_y

    out = [
        (np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), BORDER),
        (
            np.array(
                [
                    [x0 + 6 * scale, y0 + 6 * scale],
                    [x1 - 6 * scale, y0 + 6 * scale],
                    [x1 - 6 * scale, y1 - 6 * scale],
                    [x0 + 6 * scale, y1 - 6 * scale],
                ]
            ),
            SHEET,
        ),
    ]
    for (c, r) in sg.cells():
        cx, cy = c * unit_mm, r * unit_mm
        corners = np.array(
            [[cx - half, cy - half], [cx + half, cy - half],
             [cx + half, cy + half], [cx - half, cy + half]]
        )
        out.append((corners, MARKER.get((c, r), INK)))
    return out


def render(
    K, rvec, tvec, size, unit_mm=sg.PITCH_MM, background=None,
    supersample=3, blur=1.0, noise=2.5, clutter=0, seed=0,
):
    """Render the board seen by a camera. Returns ``(bgr, truth_centres)``.

    ``truth_centres`` is where each of the 24 square centres really projects,
    in canonical cell order - the thing the detector has to recover.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    big = (w * supersample, h * supersample)
    Ks = K.copy()
    Ks[:2] *= supersample

    canvas = np.full((big[1], big[0], 3), 150, np.uint8)
    if background is not None:
        canvas[:] = np.array(background, np.uint8)
    # Soft lighting gradient, so a single global threshold is never quite right.
    ramp = np.linspace(0.82, 1.15, big[0], dtype=np.float32)[None, :, None]
    canvas = np.clip(canvas.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

    for _ in range(clutter):
        pt = rng.integers([0, 0], big, size=2)
        side = int(rng.integers(8 * supersample, 40 * supersample))
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        angle = rng.uniform(0, np.pi)
        box = cv2.boxPoints(((int(pt[0]), int(pt[1])), (side, side), np.degrees(angle)))
        cv2.fillConvexPoly(canvas, box.astype(np.int32), colour)

    for points_mm, colour in board_polygons(unit_mm):
        pts3d = np.hstack([points_mm, np.zeros((len(points_mm), 1))]).astype(np.float64)
        projected, _ = cv2.projectPoints(pts3d, rvec, tvec, Ks, None)
        cv2.fillConvexPoly(
            canvas, np.round(projected.reshape(-1, 2)).astype(np.int32), colour,
            lineType=cv2.LINE_AA,
        )

    image = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    if blur > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    if noise > 0:
        image = np.clip(
            image.astype(np.float32) + rng.normal(0, noise, image.shape), 0, 255
        ).astype(np.uint8)

    truth, _ = cv2.projectPoints(sg.object_centres(unit_mm), rvec, tvec, K, None)
    return image, truth.reshape(-1, 2)


def random_pose(rng, distance, tilt_deg=45.0, offset=0.18):
    """A random board pose at roughly ``distance`` metres, tilted up to tilt_deg.

    Distances are metres, so render with ``unit_mm=UNIT_M``.
    """
    axis = rng.normal(size=3)
    axis[2] *= 0.3  # mostly tilt, not spin, so views vary in perspective
    axis = axis / np.linalg.norm(axis)
    rvec = (axis * np.radians(rng.uniform(5, tilt_deg))).reshape(3, 1)
    tvec = np.array(
        [
            rng.uniform(-offset, offset) * distance,
            rng.uniform(-offset, offset) * distance,
            distance,
        ]
    ).reshape(3, 1)
    return rvec, tvec


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """World->camera (R, t) for a camera at ``eye`` looking at ``target``."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    return R, (-R @ eye).reshape(3, 1)
