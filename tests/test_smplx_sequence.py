import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scipy.spatial.transform import Rotation  # noqa: E402

from _load import load_script  # noqa: E402
from fake_smplx import FakeSMPLX  # noqa: E402

fit = load_script("10_fit_smplx_sequence.py")
s06 = fit.load_script06()
INDICES = list(s06.CORRESPONDENCES)          # SMPL-X joints that get targets


def make_truth(betas0=1.5, floor_shift=0.0, seed=0, frames=3):
    rng = np.random.default_rng(seed)
    model = FakeSMPLX()
    # Y-up body stood up into a Z-up world, a different heading each frame
    up = Rotation.from_euler("x", 90, degrees=True)
    orient = np.stack([(Rotation.from_euler("z", 40 * f, degrees=True) * up).as_rotvec() for f in range(frames)])
    pose = rng.normal(scale=0.15, size=(frames, 63)).astype(np.float32)
    transl = np.array([[0.3 * f, 0.1 * f, 0.95 + floor_shift] for f in range(frames)], dtype=np.float32)
    betas = torch.zeros(frames, 10)
    betas[:, 0] = betas0
    out = model(betas, torch.tensor(orient, dtype=torch.float32), torch.tensor(pose),
                torch.tensor(transl), True)
    return model, out.joints.numpy(), out.vertices.numpy()


def targets_from(joints):
    F = joints.shape[0]
    target = np.zeros((F, 22, 3))
    weight = np.zeros((F, 22))
    for i in INDICES:
        target[:, i] = joints[:, i]
        weight[:, i] = s06.CORRESPONDENCES[i][1]
    return target, weight


def test_shared_shape_and_pose_recovered():
    model, joints, _ = make_truth(betas0=1.5)
    target, weight = targets_from(joints)
    res = fit.fit_sequence(model, target, weight, s06.FULL_BODY_FREE_JOINTS, s06.kabsch,
                           use_floor=False, stage_a=200, stage_b=500)
    assert res["betas"].shape == (10,)                    # one shape for all frames
    assert res["betas"][0] == pytest.approx(1.5, abs=0.35)
    assert np.nanmean(res["joint_error_cm"]) < 2.0


def test_missing_joints_get_zero_weight():
    names = ["left_hip", "right_hip", "left_knee"]
    joints = np.zeros((2, 3, 3))
    joints[:, 0] = [1, 0, 0]
    joints[:, 1] = [-1, 0, 0]
    joints[1, 2] = np.nan                                 # knee missing in frame 1
    target, weight = fit.build_targets(joints, names, s06.CORRESPONDENCES)
    assert weight[0, 4] > 0 and weight[1, 4] == 0
    np.testing.assert_allclose(target[0, 0], [0, 0, 0])   # pelvis = hip midpoint


def test_floor_term_reduces_penetration():
    model, joints, _ = make_truth(floor_shift=-0.9)        # soles end up ~ -3 cm below the floor
    target, weight = targets_from(joints)
    kw = dict(stage_a=150, stage_b=250)
    free = fit.fit_sequence(model, target, weight, s06.FULL_BODY_FREE_JOINTS, s06.kabsch, use_floor=False, **kw)
    floor = fit.fit_sequence(model, target, weight, s06.FULL_BODY_FREE_JOINTS, s06.kabsch, use_floor=True, **kw)
    assert floor["min_vertex_z_m"].min() > free["min_vertex_z_m"].min()


def project_np(joints, K, R, t):
    pc = joints @ R.T + t
    uv = pc[:, :2] / pc[:, 2:3]
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def test_reprojection_term_pulls_noisy_3d_toward_observations():
    model, joints, _ = make_truth(betas0=1.0)
    F = joints.shape[0]
    rng = np.random.default_rng(5)
    noisy = joints + rng.normal(scale=0.04, size=joints.shape)      # bad triangulation, 4 cm noise
    target, weight = targets_from(noisy)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    cams = {"K": np.stack([K, K]), "R": [], "t": []}
    for ang in (0, 60):
        R = Rotation.from_euler("z", ang, degrees=True).as_matrix()
        Rcam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]) @ R   # z-up world -> camera looking along +y
        cams["R"].append(Rcam)
        cams["t"].append(-Rcam @ np.array([0.0, -5.0, 1.0]))
    cams["R"], cams["t"] = np.stack(cams["R"]), np.stack(cams["t"])
    obs = np.full((F, 2, 22, 3), np.nan)
    for f in range(F):
        for c in range(2):
            for i in INDICES:
                if len(s06.CORRESPONDENCES[i][0]) == 1:
                    obs[f, c, i, :2] = project_np(joints[f, i][None], cams["K"][c], cams["R"][c], cams["t"][c])[0]
                    obs[f, c, i, 2] = 1.0
    kw = dict(use_floor=False, stage_a=150, stage_b=300)
    base = fit.fit_sequence(model, target, weight, s06.FULL_BODY_FREE_JOINTS, s06.kabsch, **kw)
    with2d = fit.fit_sequence(model, target, weight, s06.FULL_BODY_FREE_JOINTS, s06.kabsch,
                              obs2d=obs, cameras=cams, reproj_weight=5.0, **kw)

    def err(r):
        return np.linalg.norm(r["joints"][:, INDICES] - joints[:, INDICES], axis=-1).mean()

    assert with2d["reproj_px"] is not None
    assert err(with2d) < err(base)
