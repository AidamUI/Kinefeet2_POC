import numpy as np
import pytest

from _load import load_script

exp = load_script("11_export_opensim.py")
NAMES = ["left_hip", "right_hip", "left_knee", "nose"]


def test_heading_puts_right_hip_on_plus_z():
    rng = np.random.default_rng(3)
    for angle in rng.uniform(-np.pi, np.pi, 5):
        c, s = np.cos(angle), np.sin(angle)
        # a subject whose lateral axis points along an arbitrary floor direction (Z up world)
        lat = np.array([c, s, 0.0]) * 0.2
        joints = np.array([[[-lat[0] / 2, -lat[1] / 2, 0.9], [lat[0] / 2, lat[1] / 2, 0.9],
                            [0.0, 0.0, 0.5], [0.0, 0.0, 1.5]]])
        os_pts = exp.to_opensim_axes(joints, NAMES)
        right_minus_left = os_pts[0, 1] - os_pts[0, 0]
        np.testing.assert_allclose(right_minus_left, [0, 0, 0.2], atol=1e-9)
        assert os_pts[0, 3, 1] == pytest.approx(1.5)  # nose height ends up on Y (up)


def test_conversion_is_rigid():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(2, 4, 3))
    out = exp.to_opensim_axes(pts, NAMES)
    d_in = np.linalg.norm(pts[0, 0] - pts[0, 3])
    d_out = np.linalg.norm(out[0, 0] - out[0, 3])
    assert d_in == pytest.approx(d_out)


def test_trc_layout(tmp_path):
    pts = np.zeros((2, 4, 3))
    pts[0, 3] = [0.1, 1.5, 0.0]
    pts[1, 2] = np.nan
    path = tmp_path / "m.trc"
    markers = exp.write_trc(str(path), pts, NAMES)
    assert markers == ["LHip", "RHip", "LKnee", "Nose"]
    lines = path.read_text().splitlines()
    assert lines[2].split("\t")[2:5] == ["2", "4", "m"]
    first = lines[6].split("\t")
    assert first[0] == "1" and first[1] == "0.000000"
    assert len(first) == 2 + 3 * 4
    second = lines[7].split("\t")
    assert second[8:11] == ["", "", ""]  # NaN LKnee written as blanks
