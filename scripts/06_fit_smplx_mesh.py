"""
06_fit_smplx_mesh.py
-----------------------
Turns the sparse 3D skeleton from script 04 (output/<version>/<pose>/joints_3d.json)
into a full humanoid SMPL-X mesh (10,475 vertices, a real body surface rather
than just joints and lines) by optimizing SMPL-X's shape and pose parameters
so that its joints line up with the triangulated joints as closely as
possible.

This is a simplified version of the standard SMPLify approach:
  1. A rigid (Kabsch/Procrustes) alignment gives a good initial global
     rotation and position, so the optimizer does not get stuck rotated or
     flipped.
  2. Stage A optimizes global position and orientation plus body shape
     (betas), with the body held in a neutral pose.
  3. Stage B unfreezes the joint rotations (body_pose) and refines
     everything together with gradient descent (Adam), minimizing 3D joint
     distance plus a small regularization term that discourages unnaturally
     extreme poses or shapes.

For the waist_down version: SMPL-X models the whole body, but waist-down
captures only contain leg and hip data. For that version, this script only
lets the optimizer move the lower body joints (hips, knees, ankles, feet).
The upper body (spine, shoulders, arms, head) is deliberately held in its
neutral resting pose, since there is no data to inform it. The result is
accurate legs with a generic torso, which is expected rather than a defect.

Before running this script, the actual SMPL-X model files are needed. The
`smplx` Python package is just code; the trained model weights are
distributed separately under their own license and cannot be downloaded
automatically:

  1. Register (free) at https://smpl-x.is.tue.mpg.de/
  2. Download the "SMPL-X v1.1" model package.
  3. Unzip it and arrange the files as:
       models_smplx/smplx/SMPLX_NEUTRAL.npz
       models_smplx/smplx/SMPLX_MALE.npz      (optional)
       models_smplx/smplx/SMPLX_FEMALE.npz    (optional)
     (models_smplx/ lives at the project root, next to config.yaml)
  4. Install the extra dependencies: `pip install smplx torch trimesh`
  5. Run this script.

See GUIDE.md, Mesh Generation section, for the full walkthrough, including
the simpler MediaPipe-native mesh option in script 07 that needs no model
download.

INPUT   output/<version>/<pose>/joints_3d.json     (from script 04)
OUTPUT  output/<version>/<pose>/smplx_mesh.obj      full textured-ready mesh
        output/<version>/<pose>/smplx_mesh.ply
        output/<version>/<pose>/smplx_preview.png
        output/<version>/<pose>/smplx_interactive.html   rotatable in-browser view (needs plotly)
        output/<version>/<pose>/smplx_params.json   fitted betas/pose/orient/transl
"""

import json
import os
import sys

import numpy as np

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")
SMPLX_MODEL_ROOT = os.path.join(os.path.dirname(__file__), "..", "smpl_models")

# --------------------------------------------------------------------------
# CONFIG - edit these to taste
# --------------------------------------------------------------------------
GENDER = "neutral"     # 'neutral' | 'male' | 'female'  (needs the matching .npz file)
NUM_BETAS = 10          # shape PCA components to fit (10 is the SMPL-X default)
DEVICE = "cpu"          # 'cuda' if you have a GPU + GPU-enabled torch installed

STAGE_A_ITERS = 150     # rigid + shape only, body held neutral
STAGE_B_ITERS = 350     # full pose + shape refinement
LEARNING_RATE = 0.03
POSE_PRIOR_WEIGHT = 0.02   # discourages unnaturally extreme joint rotations
BETA_PRIOR_WEIGHT = 0.001  # discourages unnaturally extreme body shapes

# --------------------------------------------------------------------------
# SMPL-X body joint layout (this is fixed by the model, do not edit)
# --------------------------------------------------------------------------
# Joints 0-21 of SMPL-X's 55-joint skeleton are the "body" joints (hands/face
# are 22+). Joint 0 (pelvis) is the kinematic root, controlled by
# `global_orient` + `transl`. Joints 1-21 are controlled by `body_pose`
# (21 x 3 axis-angle values = 63 numbers), in this fixed order:
SMPLX_BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
NUM_BODY_JOINTS = 21  # body_pose covers joints 1..21 (index 0 = global_orient)

# Maps a SMPL-X joint index -> (MediaPipe landmark name(s) to average, weight).
# Two names means "use the midpoint of these as a synthetic target" (there's
# no direct MediaPipe landmark for the pelvis or neck, but the midpoint of the
# hips / shoulders is a good stand-in).
CORRESPONDENCES = {
    0:  (["left_hip", "right_hip"], 1.0),          # pelvis (synthetic)
    1:  (["left_hip"], 1.0),
    2:  (["right_hip"], 1.0),
    4:  (["left_knee"], 1.0),
    5:  (["right_knee"], 1.0),
    7:  (["left_ankle"], 1.0),
    8:  (["right_ankle"], 1.0),
    10: (["left_foot_index"], 0.6),                # approximate - not an exact anatomical match
    11: (["right_foot_index"], 0.6),
    12: (["left_shoulder", "right_shoulder"], 0.8),  # neck (synthetic)
    15: (["nose"], 0.3),                            # rough head proxy, downweighted
    16: (["left_shoulder"], 1.0),
    17: (["right_shoulder"], 1.0),
    18: (["left_elbow"], 1.0),
    19: (["right_elbow"], 1.0),
    20: (["left_wrist"], 1.0),
    21: (["right_wrist"], 1.0),
}

FULL_BODY_FREE_JOINTS = set(range(1, 22))                 # every body_pose joint is optimized
WAIST_DOWN_FREE_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}        # hips, knees, ankles, feet only


def get_free_joints(version):
    return FULL_BODY_FREE_JOINTS if version == "full_body" else WAIST_DOWN_FREE_JOINTS


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_targets(version, pose_name):
    """
    Reads output/<version>/<pose>/joints_3d.json (from script 04) and builds
    the list of (smplx_joint_index, xyz, weight) correspondences that are
    actually available for this pose (waist_down poses simply won't have
    upper-body keys, and are skipped automatically).
    """
    path = os.path.join(OUTPUT_ROOT, version, pose_name, "joints_3d.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    joints = data["joints"]

    targets = []
    for smplx_idx, (names, weight) in CORRESPONDENCES.items():
        pts = []
        ok = True
        for n in names:
            j = joints.get(n)
            if j is None:
                ok = False
                break
            pts.append([j["x"], j["y"], j["z"]])
        if not ok:
            continue
        xyz = np.mean(pts, axis=0)
        targets.append((smplx_idx, xyz, weight))
    return targets


# --------------------------------------------------------------------------
# Rigid alignment (Kabsch / Procrustes) for a good optimizer starting point
# --------------------------------------------------------------------------

def kabsch(P, Q, weights):
    """
    Find rotation R and translation t minimizing sum(w * ||R@P_i + t - Q_i||^2).
    Standard SVD-based rigid alignment (Kabsch algorithm), no scaling.
    """
    w = weights / weights.sum()
    cP = (P * w[:, None]).sum(axis=0)
    cQ = (Q * w[:, None]).sum(axis=0)
    P0, Q0 = P - cP, Q - cQ
    H = (P0 * w[:, None]).T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    return R, t


# --------------------------------------------------------------------------
# Model construction (this is the part that needs the licensed .npz files)
# --------------------------------------------------------------------------

def ensure_model_files():
    smplx_dir = os.path.join(SMPLX_MODEL_ROOT, "smplx")
    expected = os.path.join(smplx_dir, f"SMPLX_{GENDER.upper()}.npz")
    if os.path.exists(expected):
        return
    print("=" * 72)
    print(f"SMPL-X model file not found:\n  {os.path.abspath(expected)}")
    print()
    print("SMPL-X's code is open-source but the trained model weights need a")
    print("free license agreement and can't be auto-downloaded. To fix:")
    print("  1. Register at https://smpl-x.is.tue.mpg.de/")
    print("  2. Download the 'SMPL-X v1.1' model package")
    print("  3. Unzip it so you have:")
    print(f"       {smplx_dir}/SMPLX_NEUTRAL.npz")
    print(f"       {smplx_dir}/SMPLX_MALE.npz    (optional)")
    print(f"       {smplx_dir}/SMPLX_FEMALE.npz  (optional)")
    print("  4. Re-run this script.")
    print("See GUIDE.md, Mesh Generation section, for details.")
    print("=" * 72)
    sys.exit(1)


def build_model():
    import smplx
    model = smplx.create(
        model_path=SMPLX_MODEL_ROOT,
        model_type="smplx",
        gender=GENDER,
        num_betas=NUM_BETAS,
        use_pca=False,          # hand pose is never touched here, this just avoids the PCA hand basis
        batch_size=1,
    )
    return model


# --------------------------------------------------------------------------
# The actual fitting routine. It is pure logic that takes a pre-built
# `model`, so it can be tested with a stand-in model that does not need the
# licensed SMPL-X weights.
# --------------------------------------------------------------------------

def fit_one(model, targets, free_joints):
    import torch
    from scipy.spatial.transform import Rotation

    target_idx = torch.tensor([t[0] for t in targets], dtype=torch.long)
    target_xyz = torch.tensor(np.stack([t[1] for t in targets]), dtype=torch.float32)
    target_w = torch.tensor(np.array([t[2] for t in targets]), dtype=torch.float32)

    # --- Stage 0: rigid (Kabsch) initialization ---
    with torch.no_grad():
        neutral = model(
            betas=torch.zeros(1, NUM_BETAS),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, NUM_BODY_JOINTS * 3),
            transl=torch.zeros(1, 3),
            return_verts=False,
        )
        neutral_joints = neutral.joints[0, :22, :].cpu().numpy()

    P = np.stack([neutral_joints[idx] for idx in target_idx.tolist()])
    Q = target_xyz.numpy()
    W = target_w.numpy()
    R0, t0 = kabsch(P, Q, W)
    rotvec0 = Rotation.from_matrix(R0).as_rotvec()

    global_orient = torch.tensor(rotvec0, dtype=torch.float32).reshape(1, 3).clone().requires_grad_(True)
    transl = torch.tensor(t0, dtype=torch.float32).reshape(1, 3).clone().requires_grad_(True)
    betas = torch.zeros(1, NUM_BETAS, requires_grad=True)
    body_pose = torch.zeros(1, NUM_BODY_JOINTS * 3, requires_grad=True)

    # Gradient mask: joints NOT in `free_joints` are always kept at exactly 0
    # (e.g. the un-observed upper body in a waist_down fit).
    mask = torch.zeros(1, NUM_BODY_JOINTS * 3)
    for j in free_joints:
        mask[0, (j - 1) * 3:(j - 1) * 3 + 3] = 1.0

    def compute_loss():
        out = model(betas=betas, global_orient=global_orient,
                    body_pose=body_pose, transl=transl, return_verts=False)
        pred = out.joints[0, target_idx, :]
        diff = pred - target_xyz
        joint_loss = (target_w * (diff ** 2).sum(-1)).sum() / target_w.sum()
        pose_reg = (body_pose ** 2).mean()
        beta_reg = (betas ** 2).mean()
        loss = joint_loss + POSE_PRIOR_WEIGHT * pose_reg + BETA_PRIOR_WEIGHT * beta_reg
        return loss, pred

    # --- Stage A: rigid pose + shape only, body_pose held at 0 ---
    opt_a = torch.optim.Adam([global_orient, transl, betas], lr=LEARNING_RATE)
    for _ in range(STAGE_A_ITERS):
        opt_a.zero_grad()
        loss, _ = compute_loss()
        loss.backward()
        opt_a.step()

    # --- Stage B: unfreeze (masked) body_pose, refine everything jointly ---
    opt_b = torch.optim.Adam([global_orient, transl, betas, body_pose], lr=LEARNING_RATE * 0.7)
    for _ in range(STAGE_B_ITERS):
        opt_b.zero_grad()
        loss, _ = compute_loss()
        loss.backward()
        with torch.no_grad():
            body_pose.grad *= mask
        opt_b.step()
        with torch.no_grad():
            body_pose *= mask   # hard-enforce frozen joints stay exactly neutral

    with torch.no_grad():
        final_loss, pred = compute_loss()
        out = model(betas=betas, global_orient=global_orient,
                    body_pose=body_pose, transl=transl, return_verts=True)

    errs_cm = (torch.linalg.norm(pred - target_xyz, dim=-1) * 100).numpy()
    diagnostics = [
        {"smplx_joint": SMPLX_BODY_JOINT_NAMES[idx], "error_cm": float(e)}
        for idx, e in zip(target_idx.tolist(), errs_cm)
    ]

    result = {
        "vertices": out.vertices[0].detach().cpu().numpy(),
        "faces": np.asarray(model.faces),
        "joints": out.joints[0, :22, :].detach().cpu().numpy(),
        "params": {
            "gender": GENDER,
            "betas": betas.detach().cpu().numpy().tolist(),
            "global_orient": global_orient.detach().cpu().numpy().tolist(),
            "body_pose": body_pose.detach().cpu().numpy().tolist(),
            "transl": transl.detach().cpu().numpy().tolist(),
        },
        "diagnostics": diagnostics,
        "final_loss": float(final_loss.item()),
        "mean_error_cm": float(np.mean(errs_cm)),
    }
    return result


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export_obj_mesh(path, vertices, faces):
    with open(path, "w") as f:
        f.write("# SMPL-X mesh fitted to triangulated MediaPipe joints\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def export_ply_mesh(path, vertices, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def render_mesh_preview(path, vertices, faces, title, max_faces=20000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) > max_faces:
        # subsample triangles for a faster preview render on very dense meshes
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")

    tris = vertices[faces]  # (F, 3, 3)
    coll = Poly3DCollection(tris, facecolor="lightsteelblue", edgecolor="none", alpha=1.0)
    ax.add_collection3d(coll)

    mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
    mid = (mins + maxs) / 2
    max_range = (maxs - mins).max() / 2.0
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m) [up]")
    ax.view_init(elev=8, azim=-70)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def render_interactive_mesh(path, vertices, faces, title):
    """
    Rotatable, mouse-draggable mesh viewer you can open in any browser - no
    Blender/MeshLab needed. Uses plotly's Mesh3d, which (unlike the
    Scatter3d used for the script 05 skeleton) renders actual shaded
    triangle surfaces. Skipped silently if plotly isn't installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("    (plotly not installed - skipping interactive .html; "
              "`pip install plotly` if you want it)")
        return

    fig = go.Figure(data=[go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightsteelblue",
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
    )])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
    )
    fig.write_html(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ensure_model_files()
    print(f"Loading SMPL-X model (gender={GENDER}, {NUM_BETAS} shape components)...")
    model = build_model()

    for version in ["full_body", "waist_down"]:
        free_joints = get_free_joints(version)
        for pose_name in ["pose1", "pose2", "pose3"]:
            print(f"\n[{version}/{pose_name}]")
            targets = load_targets(version, pose_name)
            if not targets or len(targets) < 3:
                print("  (no joints_3d.json / not enough joints - run script 04 first, skipping)")
                continue

            result = fit_one(model, targets, free_joints)

            out_dir = os.path.join(OUTPUT_ROOT, version, pose_name)
            os.makedirs(out_dir, exist_ok=True)

            export_obj_mesh(os.path.join(out_dir, "smplx_mesh.obj"), result["vertices"], result["faces"])
            export_ply_mesh(os.path.join(out_dir, "smplx_mesh.ply"), result["vertices"], result["faces"])
            title = f"{version} / {pose_name}  (fit error {result['mean_error_cm']:.1f} cm)"
            render_mesh_preview(os.path.join(out_dir, "smplx_preview.png"),
                                 result["vertices"], result["faces"], title=title)
            render_interactive_mesh(os.path.join(out_dir, "smplx_interactive.html"),
                                     result["vertices"], result["faces"], title=title)
            with open(os.path.join(out_dir, "smplx_params.json"), "w") as f:
                json.dump(result["params"], f, indent=2)

            print(f"  mean joint fit error: {result['mean_error_cm']:.2f} cm  "
                  f"(rule of thumb: <3cm is great, <8cm is usable, "
                  f"higher means check joints_3d.json quality for this pose)")
            worst = max(result["diagnostics"], key=lambda d: d["error_cm"])
            print(f"  worst-fit joint: {worst['smplx_joint']} ({worst['error_cm']:.1f} cm)")
            print(f"  saved -> {out_dir}/smplx_mesh.obj (+ .ply, preview.png, interactive.html, params.json)")

    print("\nDone. Open smplx_mesh.obj in Blender/MeshLab for the full mesh, "
          "or smplx_preview.png for a quick look.")


if __name__ == "__main__":
    main()
