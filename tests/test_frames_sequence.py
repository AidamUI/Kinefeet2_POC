import json

import numpy as np
import pytest

from _load import load_script
from frames import (world_to_camera, camera_to_world, camera_center, pelvis_centre,
                    to_pelvis_frame, zup_to_yup, yup_to_zup, camera_to_display)


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_world_camera_round_trip():
    rng = np.random.default_rng(0)
    R, t = random_rotation(rng), rng.normal(size=3)
    X = rng.normal(size=(5, 23, 3))
    np.testing.assert_allclose(camera_to_world(world_to_camera(X, R, t), R, t), X, atol=1e-12)


def test_camera_centre_maps_to_origin():
    rng = np.random.default_rng(1)
    R, t = random_rotation(rng), rng.normal(size=3)
    np.testing.assert_allclose(world_to_camera(camera_center(R, t), R, t), 0, atol=1e-12)


def test_nan_passes_through():
    X = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    out = world_to_camera(X, np.eye(3), np.zeros(3))
    assert np.isnan(out[1]).all() and not np.isnan(out[0]).any()


def test_yup_is_proper_rotation_and_invertible():
    from frames import ZUP_TO_YUP
    assert np.linalg.det(ZUP_TO_YUP) == pytest.approx(1.0)
    X = np.random.default_rng(2).normal(size=(4, 3))
    np.testing.assert_allclose(yup_to_zup(zup_to_yup(X)), X, atol=1e-12)
    # world up (+Z) must become display up (+Y)
    np.testing.assert_allclose(zup_to_yup([0, 0, 1.0]), [0, 1, 0])


def test_camera_to_display_flips_y_and_z():
    np.testing.assert_allclose(camera_to_display([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_pelvis_frame_centres_hips():
    names = ["left_hip", "right_hip", "left_knee"]
    joints = np.array([[[1.0, 0, 1], [3.0, 0, 1], [2.0, 0, 0.5]]])
    np.testing.assert_allclose(pelvis_centre(joints, names), [[2.0, 0, 1]])
    local = to_pelvis_frame(joints, names)
    np.testing.assert_allclose(local[0, 2], [0, 0, -0.5])


def test_build_sequence_and_consistency(tmp_path):
    seq = load_script("09_build_sequence.py")
    version_dir = tmp_path / "full_body"
    names = ["left_hip", "right_hip", "left_knee", "left_ankle"]
    for k, shift in enumerate([0.0, 0.3]):
        d = version_dir / f"pose{k + 1}"
        d.mkdir(parents=True)
        pts = {"left_hip": (0.1 + shift, 0, 0.9), "right_hip": (-0.1 + shift, 0, 0.9),
               "left_knee": (0.1 + shift, 0, 0.5), "left_ankle": (0.1 + shift, 0, 0.1)}
        joints = {n: {"index": i, "x": p[0], "y": p[1], "z": p[2]} for i, (n, p) in enumerate(pts.items())}
        (d / "joints_3d.json").write_text(json.dumps({"joints": joints, "diagnostics": {}}))
    # patch the landmark list so only our four joints are read
    seq.landmarks_for_version = lambda v: ([23, 24, 25, 27], None)
    got_names, frames, joints = seq.load_sequence(str(tmp_path), "full_body")
    assert got_names == names and frames == ["pose1", "pose2"] and joints.shape == (2, 4, 3)
    rep = seq.consistency_report(joints, got_names, frames, True)
    assert rep["bones"]["thigh_L"]["cv_percent"] == pytest.approx(0.0, abs=1e-9)
    assert rep["bones"]["thigh_L"]["mean_m"] == pytest.approx(0.4)
    assert rep["lowest_foot_joint_z_m"] == [0.1, 0.1]
