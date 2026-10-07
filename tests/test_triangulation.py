import numpy as np
import pytest

from _load import load_script
from utils import camera_key_for_image, undistort_pixels, triangulate_point_dlt

CAMERAS = {f"cam_{i:02d}": {} for i in range(8)}


def test_camera_key_matches_by_name_not_position():
    # views 03 and 08 missing: 04 must still map to cam_03, not cam_02
    assert camera_key_for_image("04.png", CAMERAS) == "cam_03"
    assert camera_key_for_image("01.jpeg", CAMERAS) == "cam_00"


def test_camera_key_fails_loudly():
    with pytest.raises(ValueError):
        camera_key_for_image("front.png", CAMERAS)
    with pytest.raises(ValueError):
        camera_key_for_image("09.png", CAMERAS)


def test_undistort_without_coefficients_is_identity():
    uv = np.array([[100.0, 200.0], [300.5, 40.25]])
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    np.testing.assert_allclose(undistort_pixels(uv, K, []), uv)
    np.testing.assert_allclose(undistort_pixels(uv, K, [0, 0, 0, 0, 0]), uv)


def test_undistort_inverts_distortion():
    import cv2
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    dist = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])
    pts = np.array([[[100.0, 80.0]], [[500.0, 400.0]], [[320.0, 240.0]]])
    # distort ideal pixels with cv2, then undistort and compare
    norm = cv2.undistortPoints(pts, K, None)
    distorted, _ = cv2.projectPoints(
        np.concatenate([norm.reshape(-1, 2), np.ones((3, 1))], axis=1),
        np.zeros(3), np.zeros(3), K, dist)
    recovered = undistort_pixels(distorted.reshape(-1, 2), K, dist)
    np.testing.assert_allclose(recovered, pts.reshape(-1, 2), atol=0.05)


def test_dlt_recovers_point():
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    X = np.array([0.1, -0.2, 3.0])
    Ps = []
    for tx in (-0.5, 0.0, 0.5):
        Ps.append(K @ np.hstack([np.eye(3), np.array([[tx], [0], [0]])]))
    uvs = []
    for P in Ps:
        q = P @ np.append(X, 1)
        uvs.append(q[:2] / q[2])
    np.testing.assert_allclose(triangulate_point_dlt(Ps, uvs), X, atol=1e-6)


def test_triangulate_pose_uses_filename_mapping(tmp_path):
    """A missing view must not shift later views onto the wrong camera."""
    import json
    t = load_script("04_triangulate_3d.py")
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    cams, Ps = {}, {}
    for i, tx in enumerate([-1.0, -0.5, 0.0, 0.5]):
        R, tv = np.eye(3), np.array([tx, 0, 0.0])
        P = K @ np.hstack([R, tv.reshape(3, 1)])
        cams[f"cam_{i:02d}"] = {"K": K.tolist(), "R": R.tolist(), "t": tv.tolist(),
                                "P": P.tolist(), "image_width": 640, "image_height": 480}
        Ps[i] = P
    X = np.array([0.2, 0.1, 4.0])
    kp = tmp_path / "full_body" / "pose1" / "keypoints_2d"
    kp.mkdir(parents=True)
    for n in (1, 2, 4):  # view 03 missing
        q = Ps[n - 1] @ np.append(X, 1)
        lm = [{"index": j, "x_px": float(q[0] / q[2]), "y_px": float(q[1] / q[2]),
               "visibility": 1.0} for j in range(33)]
        (kp / f"{n:02d}.json").write_text(json.dumps(
            {"image": f"{n:02d}.png", "image_width": 640, "image_height": 480, "landmarks": lm}))
    t.OUTPUT_ROOT = str(tmp_path)
    res = t.triangulate_pose("full_body", "pose1", cams, 0.1, 2)
    j = res["joints"]["left_knee"]
    np.testing.assert_allclose([j["x"], j["y"], j["z"]], X, atol=1e-6)
    assert res["cameras_used"] == ["cam_00", "cam_01", "cam_03"]
