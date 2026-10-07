import json

import numpy as np
import pytest

from _load import load_script
from utils import undistort_pixels

ver = load_script("08_verify_reprojection.py")
K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1.0]])
DIST = np.array([-0.2, 0.05, 0.001, -0.001, 0.0])


def test_no_distortion_is_a_no_op():
    p = np.array([100.0, 50.0])
    np.testing.assert_allclose(ver.apply_lens_distortion(p, K, np.zeros(5)), p)
    np.testing.assert_allclose(ver.apply_lens_distortion(p, None, DIST), p)


def test_distortion_is_inverse_of_undistort():
    ideal = np.array([[520.0, 400.0]])
    distorted = ver.apply_lens_distortion(ideal[0], K, DIST)
    assert np.linalg.norm(distorted - ideal[0]) > 1.0          # the lens moves the pixel
    np.testing.assert_allclose(undistort_pixels(distorted[None], K, DIST), ideal, atol=0.05)


def test_load_3d_joints_skips_missing_joints(tmp_path):
    path = tmp_path / "joints_3d.json"
    path.write_text(json.dumps({"joints": {"left_knee": None,
                                           "left_hip": {"x": 1.0, "y": 2.0, "z": 3.0}},
                                "diagnostics": {}}))
    joints, _ = ver.load_3d_joints(str(path))
    assert list(joints) == ["left_hip"]
    np.testing.assert_allclose(joints["left_hip"], [1, 2, 3])
