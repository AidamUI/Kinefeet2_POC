"""
10_fit_smplx_sequence.py
------------------------
Fits ONE SMPL-X body to every pose of a sequence at once and places each
mesh in the calibrated world frame.

Compared with script 06 (which fits each pose on its own), this version

  * shares the body SHAPE (betas) across all frames - it is the same person,
    so limb lengths cannot change between poses; this also makes the fit far
    less sensitive to one noisy pose;
  * fits each frame's global orientation, translation and body pose
    separately (the person moves and changes posture between frames);
  * adds a floor term: with a calibrated rig the floor is Z = 0, and no
    vertex may sink below it.

Loss per iteration
      3D joint distance to the triangulated joints (weighted, script 06's
      correspondences)  +  2D reprojection into every camera that detected the
      joint (robust Geman-McClure, in metric units at the joint depth)
      +  floor penetration  +  pose prior  +  shape prior.
The 2D term anchors the mesh to what the cameras actually saw, so a joint
that triangulated badly from a few views is pulled back by the others.

NEEDS THE SMPL-X MODEL WEIGHTS, which are licensed and cannot be downloaded
automatically:
  1. Register (free) at https://smpl-x.is.tue.mpg.de/
  2. Download "SMPL-X v1.1" and place the files at
       smpl_models/smplx/SMPLX_NEUTRAL.npz   (and optionally MALE / FEMALE)
  3. pip install smplx torch trimesh

USAGE   python scripts/10_fit_smplx_sequence.py <rig_output_dir> [--version full_body]
                                                 [--gender neutral] [--model-dir smpl_models]

INPUT   <rig_output_dir>/<version>/sequence/joints_world.npz   (script 09)
OUTPUT  <rig_output_dir>/<version>/sequence/smplx_world.npz
            vertices (F, V, 3), faces, joints (F, 22, 3), betas, body_pose, ...
        <rig_output_dir>/<version>/sequence/smplx/<pose>.obj
        <rig_output_dir>/<version>/sequence/smplx_fit.json   (errors in cm)
        script 12 shows the meshes automatically if smplx_world.npz exists.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NUM_BETAS = 10
NUM_BODY_JOINTS = 21
STAGE_A_ITERS = 150
STAGE_B_ITERS = 350
LEARNING_RATE = 0.03
POSE_PRIOR_WEIGHT = 0.02
BETA_PRIOR_WEIGHT = 0.001
FLOOR_WEIGHT = 5.0
REPROJ_WEIGHT = 3.0
REPROJ_SIGMA_M = 0.05   # residuals beyond ~5 cm are down-weighted


def load_script06():
    """Reuse the joint correspondences and Kabsch alignment of script 06."""
    spec = importlib.util.spec_from_file_location("fit_smplx_mesh", os.path.join(HERE, "06_fit_smplx_mesh.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_targets(joints, names, correspondences):
    """(F, J, 3) world joints -> target (F, 22, 3) and weight (F, 22) arrays.

    A SMPL-X joint with no (or a partly missing) observation gets weight 0.
    """
    F = joints.shape[0]
    target = np.zeros((F, 22, 3))
    weight = np.zeros((F, 22))
    for smplx_idx, (src, w) in correspondences.items():
        if not all(n in names for n in src):
            continue
        pts = np.stack([joints[:, names.index(n), :] for n in src])  # (k, F, 3)
        ok = ~np.isnan(pts).any(axis=(0, 2))
        mean = np.where(np.isnan(pts), 0.0, pts).mean(axis=0)
        target[ok, smplx_idx] = mean[ok]
        weight[ok, smplx_idx] = w
    return target, weight


def build_observations_2d(kp2d, names, correspondences):
    """(F, C, J, 3) detections -> (F, C, 22, 3) for SMPL-X joints that map to ONE landmark.

    Synthetic joints (pelvis, neck = midpoints) have no direct detection and
    stay NaN, so they are fitted from 3D only.
    """
    F, C = kp2d.shape[:2]
    obs = np.full((F, C, 22, 3), np.nan)
    for smplx_idx, (src, _) in correspondences.items():
        if len(src) == 1 and src[0] in names:
            obs[:, :, smplx_idx] = kp2d[:, :, names.index(src[0])]
    return obs


def fit_sequence(model, target, weight, free_joints, kabsch, use_floor=True,
                 stage_a=STAGE_A_ITERS, stage_b=STAGE_B_ITERS, lr=LEARNING_RATE, num_betas=NUM_BETAS,
                 obs2d=None, cameras=None, reproj_weight=REPROJ_WEIGHT):
    """Fit shared betas + per-frame pose to targets. ``model`` must be built with batch_size=F.

    obs2d    (F, C, 22, 3) detections (u, v, visibility), NaN where missing
    cameras  dict with K (C,3,3), R (C,3,3), t (C,3) in the same pixel frame as obs2d
    """
    import torch
    from scipy.spatial.transform import Rotation

    F = target.shape[0]
    tgt = torch.tensor(target, dtype=torch.float32)
    wgt = torch.tensor(weight, dtype=torch.float32)

    with torch.no_grad():
        neutral = model(betas=torch.zeros(F, num_betas), global_orient=torch.zeros(F, 3),
                        body_pose=torch.zeros(F, NUM_BODY_JOINTS * 3), transl=torch.zeros(F, 3),
                        return_verts=False).joints[:, :22, :].numpy()

    rot0, tr0 = [], []
    for f in range(F):
        sel = weight[f] > 0
        if sel.sum() < 3:
            raise ValueError(f"frame {f}: fewer than 3 joints available, cannot fit")
        R, t = kabsch(neutral[f][sel], target[f][sel], weight[f][sel])
        rot0.append(Rotation.from_matrix(R).as_rotvec())
        tr0.append(t)
    global_orient = torch.tensor(np.array(rot0), dtype=torch.float32).requires_grad_(True)
    transl = torch.tensor(np.array(tr0), dtype=torch.float32).requires_grad_(True)
    betas = torch.zeros(1, num_betas, requires_grad=True)       # shared by every frame
    body_pose = torch.zeros(F, NUM_BODY_JOINTS * 3, requires_grad=True)

    use_2d = obs2d is not None and cameras is not None and reproj_weight > 0
    if use_2d:
        o_valid = torch.tensor(~np.isnan(obs2d[..., 0]) & (np.nan_to_num(obs2d[..., 2]) > 0.1))
        o_uv = torch.tensor(np.nan_to_num(obs2d[..., :2]), dtype=torch.float32)
        o_w = torch.tensor(np.nan_to_num(obs2d[..., 2]), dtype=torch.float32) * o_valid
        cK = torch.tensor(np.asarray(cameras["K"]), dtype=torch.float32)
        cR = torch.tensor(np.asarray(cameras["R"]), dtype=torch.float32)
        ct = torch.tensor(np.asarray(cameras["t"]), dtype=torch.float32)
        focal = cK[:, 0, 0].mean()

    def project(joints):
        # joints (F,22,3) -> camera frame (F,C,22,3) -> pixels (F,C,22,2) and depth (F,C,22)
        pc = torch.einsum("cij,fkj->fcki", cR, joints) + ct[:, None, :][None]
        z = pc[..., 2].clamp_min(1e-3)
        px = torch.einsum("cij,fckj->fcki", cK, pc / z[..., None])[..., :2]
        return px, z

    def reproj_loss(joints):
        px, z = project(joints)
        res_m = (px - o_uv).norm(dim=-1) * z / focal       # pixel error -> metres at the joint depth
        gm = res_m ** 2 * REPROJ_SIGMA_M ** 2 / (res_m ** 2 + REPROJ_SIGMA_M ** 2)
        return (o_w * gm).sum() / o_w.sum().clamp_min(1e-6)

    mask = torch.zeros(1, NUM_BODY_JOINTS * 3)
    for j in free_joints:
        mask[0, (j - 1) * 3:(j - 1) * 3 + 3] = 1.0

    def run(verts):
        return model(betas=betas.expand(F, -1), global_orient=global_orient,
                     body_pose=body_pose, transl=transl, return_verts=verts)

    def loss_fn():
        out = run(use_floor)
        diff = out.joints[:, :22, :] - tgt
        joint_loss = (wgt * (diff ** 2).sum(-1)).sum() / wgt.sum()
        loss = (joint_loss + POSE_PRIOR_WEIGHT * (body_pose ** 2).mean()
                + BETA_PRIOR_WEIGHT * (betas ** 2).mean())
        if use_floor:
            below = torch.relu(-out.vertices[..., 2])
            loss = loss + FLOOR_WEIGHT * (below ** 2).mean()
        if use_2d:
            loss = loss + reproj_weight * reproj_loss(out.joints[:, :22, :])
        return loss

    opt_a = torch.optim.Adam([global_orient, transl, betas], lr=lr)
    for _ in range(stage_a):
        opt_a.zero_grad()
        loss_fn().backward()
        opt_a.step()
    opt_b = torch.optim.Adam([global_orient, transl, betas, body_pose], lr=lr * 0.7)
    for _ in range(stage_b):
        opt_b.zero_grad()
        loss_fn().backward()
        with torch.no_grad():
            body_pose.grad *= mask
        opt_b.step()
        with torch.no_grad():
            body_pose *= mask

    with torch.no_grad():
        out = run(True)
    pred = out.joints[:, :22, :].numpy()
    err_cm = np.linalg.norm(pred - target, axis=-1) * 100
    err_cm = np.where(weight > 0, err_cm, np.nan)
    verts = out.vertices.numpy()
    reproj_px = None
    if use_2d:
        with torch.no_grad():
            px, _ = project(out.joints[:, :22, :])
            reproj_px = np.where(o_valid.numpy(), (px - o_uv).norm(dim=-1).numpy(), np.nan)
    return {
        "reproj_px": reproj_px,
        "vertices": verts, "faces": np.asarray(model.faces), "joints": pred,
        "betas": betas.detach().numpy()[0], "body_pose": body_pose.detach().numpy(),
        "global_orient": global_orient.detach().numpy(), "transl": transl.detach().numpy(),
        "joint_error_cm": err_cm,
        "min_vertex_z_m": verts[..., 2].min(axis=1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig_output_dir")
    ap.add_argument("--version", default="full_body", choices=["full_body", "waist_down"])
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--reproj-weight", type=float, default=REPROJ_WEIGHT,
                    help="weight of the 2D reprojection term (0 = 3D joints only)")
    ap.add_argument("--model-dir", default=os.path.join(HERE, "..", "smpl_models"))
    args = ap.parse_args()

    s06 = load_script06()
    model_file = os.path.join(args.model_dir, "smplx", f"SMPLX_{args.gender.upper()}.npz")
    if not os.path.exists(model_file):
        s06.SMPLX_MODEL_ROOT = args.model_dir
        s06.GENDER = args.gender
        s06.ensure_model_files()   # prints the download instructions and exits

    seq_dir = os.path.join(args.rig_output_dir, args.version, "sequence")
    d = np.load(os.path.join(seq_dir, "joints_world.npz"), allow_pickle=False)
    names, frames, calibrated = [str(n) for n in d["names"]], [str(f) for f in d["frames"]], bool(d["calibrated"])
    target, weight = build_targets(d["joints"], names, s06.CORRESPONDENCES)
    obs2d = build_observations_2d(d["keypoints2d"], names, s06.CORRESPONDENCES) if "keypoints2d" in d else None
    cameras = {"K": d["K"], "R": d["R"], "t": d["t"]}

    import smplx
    model = smplx.create(model_path=args.model_dir, model_type="smplx", gender=args.gender,
                         num_betas=NUM_BETAS, use_pca=False, batch_size=len(frames))
    if not calibrated:
        print("  [!] cameras are NOT calibrated: floor term disabled and body size is not metric")
    res = fit_sequence(model, target, weight, s06.get_free_joints(args.version), s06.kabsch,
                       use_floor=calibrated, obs2d=obs2d, cameras=cameras,
                       reproj_weight=args.reproj_weight)

    np.savez(os.path.join(seq_dir, "smplx_world.npz"), frames=np.array(frames), **{
        k: v for k, v in res.items() if v is not None})
    mesh_dir = os.path.join(seq_dir, "smplx")
    os.makedirs(mesh_dir, exist_ok=True)
    for f, pose in enumerate(frames):
        s06.export_obj_mesh(os.path.join(mesh_dir, f"{pose}.obj"), res["vertices"][f], res["faces"])
    report = {
        "frames": frames, "gender": args.gender, "calibrated": calibrated,
        "shared_betas": [round(float(b), 4) for b in res["betas"]],
        "mean_joint_error_cm": [round(float(np.nanmean(e)), 2) for e in res["joint_error_cm"]],
        "max_joint_error_cm": [round(float(np.nanmax(e)), 2) for e in res["joint_error_cm"]],
        "lowest_vertex_z_m": [round(float(z), 3) for z in res["min_vertex_z_m"]],
        "mean_reprojection_px": None if res["reproj_px"] is None else
            [round(float(np.nanmean(e)), 2) for e in res["reproj_px"]],
        "loss_terms": "3D joint distance + 2D reprojection + floor penetration + pose prior + shape prior"
                      if res["reproj_px"] is not None else
                      "3D joint distance + floor penetration + pose prior + shape prior (no 2D term)",
    }
    with open(os.path.join(seq_dir, "smplx_fit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"fitted {len(frames)} frames with shared shape -> {seq_dir}")
    print(f"  mean joint error (cm): {report['mean_joint_error_cm']}  "
          f"(<3 great, <8 usable)")
    print(f"  lowest mesh vertex Z (m): {report['lowest_vertex_z_m']}")


if __name__ == "__main__":
    main()
