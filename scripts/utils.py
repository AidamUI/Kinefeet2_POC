"""
utils.py
--------
Shared helper functions used by every stage of the pipeline:

  1. MediaPipe BlazePose landmark names / skeleton connections
  2. Synthetic camera rig math (turning "camera #5, 180 degrees around
     the subject" into a proper 3x4 projection matrix)
  3. Multi-view triangulation (Direct Linear Transform / DLT)

Nothing in this file talks to disk or to MediaPipe directly - it's pure
math + constants, which makes it easy to unit-test and reuse across
scripts 02-05.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. LANDMARK DEFINITIONS
# ---------------------------------------------------------------------------
# BlazePose (the model MediaPipe Pose uses) always outputs the SAME 33
# landmarks, in the SAME order, no matter which image you feed it. Indices
# below match MediaPipe's official documentation.

LANDMARK_NAMES = {
    0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky", 19: "left_index",
    20: "right_index", 21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle", 29: "left_heel",
    30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
}
NUM_LANDMARKS = 33

# Full BlazePose skeleton, as (start_index, end_index) pairs.
# This is a hard-coded copy of mediapipe's PoseLandmarksConnections so the
# rest of the pipeline does not depend on which MediaPipe API version
# (legacy `mp.solutions.pose` or the new Tasks API) is installed.
POSE_CONNECTIONS_ALL = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# For the "full body" model we drop the fine face landmarks (1-10, the
# eyes/ears/mouth points) because they add visual clutter and MediaPipe's
# face localisation is not what we care about for a body model. We KEEP
# the nose (0) as a single head marker.
FACE_DETAIL_LANDMARKS = set(range(1, 11))
FULL_BODY_LANDMARKS = [i for i in range(NUM_LANDMARKS) if i not in FACE_DETAIL_LANDMARKS]
FULL_BODY_CONNECTIONS = [
    (a, b) for (a, b) in POSE_CONNECTIONS_ALL
    if a not in FACE_DETAIL_LANDMARKS and b not in FACE_DETAIL_LANDMARKS
]

# Waist-down subset: both hips, knees, ankles, heels, foot indices.
WAIST_DOWN_LANDMARKS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
WAIST_DOWN_CONNECTIONS = [
    (a, b) for (a, b) in POSE_CONNECTIONS_ALL
    if a in WAIST_DOWN_LANDMARKS and b in WAIST_DOWN_LANDMARKS
]


def landmarks_for_version(version: str):
    """Return (landmark_indices, connections) for 'full_body' or 'waist_down'."""
    if version == "full_body":
        return FULL_BODY_LANDMARKS, FULL_BODY_CONNECTIONS
    elif version == "waist_down":
        return WAIST_DOWN_LANDMARKS, WAIST_DOWN_CONNECTIONS
    raise ValueError(f"Unknown version '{version}', expected 'full_body' or 'waist_down'")


# ---------------------------------------------------------------------------
# 2. CAMERA RIG MATH
# ---------------------------------------------------------------------------
# We assume photos were taken by walking a single camera around the subject
# on an (approximate) circle, evenly spaced in angle, all pointed at a fixed
# target point on the subject's body. This is the "turntable" or "orbit
# rig" assumption. It is not as accurate as a full structure-from-motion
# solve (see README "Improving accuracy"), but it is simple, requires no
# feature matching, and works well enough for a body skeleton as long as
# the measurements (radius / height / target height) are reasonably
# accurate and the subject does not move between shots.

def build_intrinsics(image_width, image_height, sensor_fov_deg=60.0):
    """
    Build a pinhole camera intrinsic matrix K.

    This is a fallback. camera_calibration/ measures K properly from the
    square-grid board and writes it into cameras.json, which is what
    04_triangulate_3d.py reads.

    Otherwise, this builds an *approximate* K from a guessed horizontal
    field of view. Most modern phone main cameras are roughly 60-75
    degrees horizontal FOV - check your phone's camera specs for a
    better number if you can find it.
    """
    fx = fy = (image_width / 2.0) / np.tan(np.radians(sensor_fov_deg) / 2.0)
    cx, cy = image_width / 2.0, image_height / 2.0
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0],
    ])
    return K


def build_extrinsics(angle_deg, radius_m, camera_height_m, target_height_m):
    """
    Compute [R | t] (world -> camera) for one camera on the orbit rig.

    World coordinate system:
        - Origin: on the floor, directly below the subject's center.
        - +Z: up.
        - The subject stands at (0, 0, *).
        - Angle 0 deg is looking at the subject from +X, angles increase
          counter-clockwise when viewed from above (i.e. walking around
          the subject).

    Parameters
    ----------
    angle_deg : float           position of this camera on the circle
    radius_m  : float           distance from camera to the subject's center line (meters)
    camera_height_m : float     height of the camera lens off the floor (meters)
    target_height_m : float     height of the point the camera is aimed at (meters),
                                 e.g. mid-torso for full body, mid-thigh for waist-down

    Returns
    -------
    R : (3,3) rotation matrix, world -> camera
    t : (3,1) translation vector, world -> camera
    cam_pos : (3,) camera position in world coordinates (handy for debugging/plotting)
    """
    theta = np.radians(angle_deg)
    cam_pos = np.array([radius_m * np.cos(theta),
                         radius_m * np.sin(theta),
                         camera_height_m])
    target = np.array([0.0, 0.0, target_height_m])

    # OpenCV camera convention: +X right, +Y down, +Z forward (into the scene).
    world_up = np.array([0.0, 0.0, 1.0])

    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)          # camera forward
    x_axis = np.cross(z_axis, world_up)
    x_axis = x_axis / np.linalg.norm(x_axis)           # camera right
    y_axis = np.cross(z_axis, x_axis)                  # camera down (right-handed: y = z cross x)

    R = np.vstack([x_axis, y_axis, z_axis])            # world -> camera rotation
    t = -R @ cam_pos                                    # world -> camera translation
    return R, t.reshape(3, 1), cam_pos


def projection_matrix(K, R, t):
    """Combine intrinsics + extrinsics into the 3x4 camera projection matrix P = K[R|t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


# ---------------------------------------------------------------------------
# 3. MULTI-VIEW TRIANGULATION (DIRECT LINEAR TRANSFORM)
# ---------------------------------------------------------------------------

def triangulate_point_dlt(P_list, uv_list, weights=None):
    """
    Triangulate a single 3D point from N >= 2 views using the Direct Linear
    Transform. This is the classic least-squares multi-view triangulation
    method (Hartley & Zisserman, "Multiple View Geometry", ch. 12).

    Parameters
    ----------
    P_list : list of (3,4) ndarray       camera projection matrices
    uv_list : list of (u, v) tuples      2D pixel observations, same order as P_list
    weights : list of float or None      per-view confidence weight (e.g. MediaPipe
                                          visibility score). Higher = trusted more.

    Returns
    -------
    (3,) ndarray world XYZ, or None if fewer than 2 usable views were given.
    """
    n = len(P_list)
    if n < 2:
        return None
    if weights is None:
        weights = [1.0] * n

    A_rows = []
    for P, (u, v), w in zip(P_list, uv_list, weights):
        # Each view contributes 2 equations from  x = PX  (homogeneous):
        #   u*(P[2,:]) - P[0,:] = 0
        #   v*(P[2,:]) - P[1,:] = 0
        A_rows.append(w * (u * P[2, :] - P[0, :]))
        A_rows.append(w * (v * P[2, :] - P[1, :]))
    A = np.stack(A_rows, axis=0)

    # Solve via SVD: the 3D point (homogeneous) is the singular vector
    # associated with the smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]
    return X


def reprojection_error(P, point3d, uv):
    """Pixel distance between the projection of point3d through P and the observed uv."""
    Xh = np.append(point3d, 1.0)
    proj = P @ Xh
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - np.array(uv)))
