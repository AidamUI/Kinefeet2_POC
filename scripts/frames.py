"""
frames.py
---------
Coordinate-frame helpers for the motion / global-reconstruction stages.

The pipeline works in one **world** frame: metres, Z up, origin set by the
calibration board lying on the floor (see camera_calibration/README.md).
This module converts joint arrays between that frame and the ones a viewer
or biomechanics tool wants:

  world        the calibrated frame (Z up).
  camera i     the frame of camera i: X_c = R_i X + t_i, OpenCV axes
               (x right, y down, z forward). "Motion with respect to
               camera 1" means this frame for camera 1.
  pelvis       world axes, origin at the mid-hip point of each frame
               (removes translation, keeps orientation).
  y-up         a display/OpenSim axis convention: (x, y, z) -> (x, z, -y),
               a proper rotation, so handedness is preserved.

Pure numpy; works on any array whose last axis is 3, so one frame (J, 3) or
a whole sequence (F, J, 3) both work. NaN (a joint that was not
triangulated) passes through untouched.
"""

import numpy as np


def world_to_camera(points, R, t):
    """X_c = R X + t for (..., 3) points."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return np.asarray(points, dtype=np.float64) @ R.T + t


def camera_to_world(points, R, t):
    """Inverse of world_to_camera: X = R^T (X_c - t)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (np.asarray(points, dtype=np.float64) - t) @ R


def camera_center(R, t):
    """Camera position in world coordinates: C = -R^T t."""
    R = np.asarray(R, dtype=np.float64)
    return -R.T @ np.asarray(t, dtype=np.float64).reshape(3)


def pelvis_centre(joints, names, left="left_hip", right="right_hip"):
    """Mid-hip point for each frame; joints is (F, J, 3), returns (F, 3)."""
    li, ri = list(names).index(left), list(names).index(right)
    return 0.5 * (joints[:, li, :] + joints[:, ri, :])


def to_pelvis_frame(joints, names):
    """Subtract the mid-hip point of each frame (world axes kept)."""
    return joints - pelvis_centre(joints, names)[:, None, :]


# (x, y, z) -> (x, z, -y): rotation of -90 deg about x, Z-up -> Y-up.
ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [0.0, -1.0, 0.0]])


def zup_to_yup(points):
    return np.asarray(points, dtype=np.float64) @ ZUP_TO_YUP.T


def yup_to_zup(points):
    return np.asarray(points, dtype=np.float64) @ ZUP_TO_YUP


def camera_to_display(points_cam):
    """OpenCV camera axes (x right, y down, z fwd) -> a y-up viewer frame
    (x right, y up, z backward): (x, y, z) -> (x, -y, -z)."""
    p = np.asarray(points_cam, dtype=np.float64)
    return p * np.array([1.0, -1.0, -1.0])
