import importlib.util
import json
import os

import numpy as np

from _load import ROOT, load_script

MARKER = "camera_calibration/export_cameras.py"


def load_8cam_common():
    path = os.path.join(ROOT, "8camera", "scripts", "_pipeline_common.py")
    spec = importlib.util.spec_from_file_location("eight_cam_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_cameras(path, n, source=None):
    K = [[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]
    cams = {}
    for i in range(n):
        cam = {"K": K, "R": np.eye(3).tolist(), "t": [0, 0, float(i)], "image_width": 640, "image_height": 480}
        if source:
            cam["source"] = source
        cams[f"cam_{i:02d}"] = cam
    path.write_text(json.dumps(cams))


def test_eight_camera_runner_recognises_a_calibration(tmp_path):
    common = load_8cam_common()
    calibrated, assumed, missing = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    write_cameras(calibrated, 8, MARKER)
    write_cameras(assumed, 8)
    assert common.is_calibrated(str(calibrated))
    assert not common.is_calibrated(str(assumed))
    assert not common.is_calibrated(str(missing))


def test_setup_script_keeps_a_calibrated_file(tmp_path):
    s03 = load_script("03_setup_camera_rig.py")
    calibrated = tmp_path / "cameras.json"
    write_cameras(calibrated, 8, MARKER)
    assert s03.is_calibrated(str(calibrated))   # step 03 therefore leaves an 8-camera calibration alone


def test_triangulation_uses_all_eight_calibrated_cameras(tmp_path):
    t = load_script("04_triangulate_3d.py")
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    X = np.array([0.1, -0.1, 0.2])
    cams, kp = {}, tmp_path / "full_body" / "pose1" / "keypoints_2d"
    kp.mkdir(parents=True)
    for i in range(8):
        a = np.radians(i * 45)
        eye = np.array([3 * np.cos(a), 3 * np.sin(a), 1.2])
        z = -eye / np.linalg.norm(eye)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        tv = -R @ eye
        P = K @ np.hstack([R, tv.reshape(3, 1)])
        cams[f"cam_{i:02d}"] = {"K": K.tolist(), "R": R.tolist(), "t": tv.tolist(), "P": P.tolist(),
                                "image_width": 640, "image_height": 480, "source": MARKER}
        q = P @ np.append(X, 1)
        lms = [{"index": j, "x_px": float(q[0] / q[2]), "y_px": float(q[1] / q[2]), "visibility": 1.0}
               for j in range(33)]
        (kp / f"{i + 1:02d}.json").write_text(json.dumps(
            {"image": f"{i + 1:02d}.jpeg", "image_width": 640, "image_height": 480, "landmarks": lms}))
    t.OUTPUT_ROOT = str(tmp_path)
    res = t.triangulate_pose("full_body", "pose1", cams, 0.1, 3)
    j = res["joints"]["left_knee"]
    np.testing.assert_allclose([j["x"], j["y"], j["z"]], X, atol=1e-6)
    assert res["cameras_used"] == [f"cam_{i:02d}" for i in range(8)]
