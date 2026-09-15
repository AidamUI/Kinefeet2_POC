"""
test_calibration.py
-------------------
Checks the calibration itself against ground truth, not just the detector.

    python tests/test_calibration.py

Renders synthetic board views with a camera whose K is known exactly, runs the
same calibrateCamera path calibrate_intrinsics.py uses, and compares. Then
places four cameras at known positions around a board on the floor, runs the
extrinsic solve, and compares the recovered rig to the one that was rendered.

Reprojection error cannot catch a calibration that is wrong in a
self-consistent way - a focal length that is too long is absorbed exactly by
putting everything further away. Only ground truth catches that, which is what
this file is for.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibrate_extrinsics as ext  # noqa: E402
import common  # noqa: E402
import squaregrid as sg  # noqa: E402
import synthetic as syn  # noqa: E402

SIZE = (1280, 720)


def test_intrinsics(n_views=22, seed=1):
    """Recover a known K from synthetic views."""
    true_K = np.array([[980.0, 0, 655.0], [0, 972.0, 345.0], [0, 0, 1]])
    rng = np.random.default_rng(seed)

    obj_pts, img_pts = [], []
    for i in range(n_views):
        distance = rng.uniform(0.9, 2.2)
        rvec, tvec = syn.random_pose(rng, distance, tilt_deg=48)
        image, _ = syn.render(
            true_K, rvec, tvec, SIZE, unit_mm=syn.UNIT_M, seed=int(rng.integers(1 << 30))
        )
        detection = sg.detect(image, unit_mm=syn.UNIT_M)
        if detection is None or detection.mirrored:
            continue
        obj, img = common.points_from_detection(detection, use_corners=False)
        obj_pts.append(obj)
        img_pts.append(img)

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, SIZE, None, None, flags=cv2.CALIB_FIX_K3
    )

    print("\n  INTRINSICS (known camera, synthetic views)")
    print(f"    views used          {len(obj_pts)}/{n_views}")
    print(f"    reprojection RMS    {rms:.4f} px")
    labels = ("fx", "fy", "cx", "cy")
    truth = (true_K[0, 0], true_K[1, 1], true_K[0, 2], true_K[1, 2])
    got = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    worst = 0.0
    for name, t, g in zip(labels, truth, got):
        print(f"    {name:3s} true {t:8.2f}   recovered {g:8.2f}   "
              f"error {g - t:+6.2f} px")
        worst = max(worst, abs(g - t))
    print(f"    distortion (true 0) {np.round(dist.ravel()[:4], 4)}")
    return worst


def test_extrinsics(seed=2):
    """Recover a known 4-camera rig from synthetic views of a board on the floor."""
    true_K = np.array([[980.0, 0, 640.0], [0, 980.0, 360.0], [0, 0, 1]])
    radius, height = 3.2, 1.35
    eyes = [
        np.array([radius * np.cos(a), radius * np.sin(a), height])
        for a in np.radians([0, 90, 180, 270])
    ]
    target = np.array([0.0, 0.0, 0.0])

    rng = np.random.default_rng(seed)
    observations = []
    intrinsics = {c: (true_K, np.zeros(5), SIZE, "synthetic") for c in range(4)}

    # Three board placements on the floor, in the world Z=0 plane.
    placements = [
        (np.array([0.0, 0.0, 0.0]), 0.0),
        (np.array([0.7, -0.5, 0.0]), 35.0),
        (np.array([-0.6, 0.4, 0.0]), -50.0),
    ]
    for p, (offset, spin) in enumerate(placements):
        # Board->world: rotate about Z, translate. The board's own +Z points
        # into the floor, which ext.FLIP_Z_UP accounts for.
        angle = np.radians(spin)
        Rz = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        R_world_board = Rz @ ext.FLIP_Z_UP
        t_world_board = offset.reshape(3, 1)

        for c, eye in enumerate(eyes):
            R_cam_world, t_cam_world = syn.look_at(eye, target)
            R = R_cam_world @ R_world_board
            t = R_cam_world @ t_world_board + t_cam_world
            image, _ = syn.render(
                true_K, cv2.Rodrigues(R)[0], t, SIZE, unit_mm=syn.UNIT_M,
                clutter=20, seed=int(rng.integers(1 << 30)),
            )
            detection = sg.detect(image, unit_mm=syn.UNIT_M)
            if detection is None or detection.mirrored:
                print(f"    [!] camera {c} placement {p}: not detected")
                continue
            obj, img = common.points_from_detection(detection, use_corners=False)
            observations.append((c, p, obj.astype(np.float64), img.astype(np.float64)))

    T_cam, T_board, _ = ext.initialise(
        observations, intrinsics, 4, len(placements), "z_up"
    )
    T_cam, T_board, _ = ext.bundle_adjust(observations, intrinsics, T_cam, T_board)

    print("\n  EXTRINSICS (known 4-camera rig, synthetic views)")
    _, errors = ext.errors_per_camera(
        observations, intrinsics, T_cam, T_board, [f"cam{c}" for c in range(4)]
    )
    print(f"    reprojection RMS    {np.sqrt((errors ** 2).mean()):.4f} px")

    # The world frame is pinned to placement 0, which sits at the origin here,
    # so recovered positions are directly comparable to the rendered ones.
    worst = 0.0
    for c in sorted(T_cam):
        R, t = T_cam[c]
        centre = (-R.T @ t).ravel()
        error = np.linalg.norm(centre - eyes[c])
        worst = max(worst, error)
        print(f"    cam{c} true ({eyes[c][0]:+6.3f},{eyes[c][1]:+6.3f},"
              f"{eyes[c][2]:+6.3f})  got ({centre[0]:+6.3f},{centre[1]:+6.3f},"
              f"{centre[2]:+6.3f})  off {1000 * error:5.1f} mm")

    residuals, scale = ext.triangulation_check(observations, intrinsics, T_cam, T_board)
    if len(residuals):
        print(f"    board triangulation {1000 * np.sqrt((residuals ** 2).mean()):.2f} mm"
              f"   scale {100 * (np.mean(scale) - 1):+.3f}%")
    return worst


def main():
    print("=" * 72)
    print("CALIBRATION AGAINST SYNTHETIC GROUND TRUTH")
    print("=" * 72)
    intrinsic_error = test_intrinsics()
    extrinsic_error = test_extrinsics()

    print("\n  VERDICT")
    print(f"    worst intrinsic parameter error  {intrinsic_error:.2f} px")
    print(f"    worst camera position error      {1000 * extrinsic_error:.1f} mm")
    ok = intrinsic_error < 8.0 and extrinsic_error < 0.02
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
