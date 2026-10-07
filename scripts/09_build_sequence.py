"""
09_build_sequence.py
--------------------
Stacks the per-pose reconstructions (script 04) into one time-ordered
sequence in the calibrated world frame, and checks that the result is
self-consistent.

Each pose folder (pose1, pose2, pose3...) becomes one "frame". The poses are
separate photo sessions, so the frame index carries no real timing; the same
file format will hold true video frames later (one row per frame).

Because the same person is in every pose, a few quantities must agree
between poses even without a ground-truth measurement:
  * bone lengths (thigh, shank, upper arm...) - a body does not change length
  * left/right symmetry of those bones
  * feet should rest on the floor (world Z = 0 when calibrated)
These are reported in consistency.json as a check on the geometry.

USAGE   python scripts/09_build_sequence.py <rig_output_dir> [--version full_body]
        e.g. python scripts/09_build_sequence.py 4camera/output

INPUT   <rig_output_dir>/<version>/<pose>/joints_3d.json   (script 04)
        <rig_output_dir>/<version>/cameras.json
OUTPUT  <rig_output_dir>/<version>/sequence/joints_world.npz
            joints  (F, J, 3) metres, world frame, NaN where not triangulated
            names   (J,) joint names     frames (F,) pose names
            cam_keys, K, R, t, image_size   camera data for viewers
        <rig_output_dir>/<version>/sequence/consistency.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import landmarks_for_version, LANDMARK_NAMES
from frames import pelvis_centre

CALIBRATED_SOURCE_MARKER = "camera_calibration/export_cameras.py"

# (label, joint_a, joint_b)
BONES = [
    ("thigh_L", "left_hip", "left_knee"), ("thigh_R", "right_hip", "right_knee"),
    ("shank_L", "left_knee", "left_ankle"), ("shank_R", "right_knee", "right_ankle"),
    ("foot_L", "left_heel", "left_foot_index"), ("foot_R", "right_heel", "right_foot_index"),
    ("upper_arm_L", "left_shoulder", "left_elbow"), ("upper_arm_R", "right_shoulder", "right_elbow"),
    ("forearm_L", "left_elbow", "left_wrist"), ("forearm_R", "right_elbow", "right_wrist"),
    ("hip_width", "left_hip", "right_hip"), ("shoulder_width", "left_shoulder", "right_shoulder"),
]
FOOT_JOINTS = ["left_ankle", "right_ankle", "left_heel", "right_heel",
               "left_foot_index", "right_foot_index"]


def load_sequence(rig_dir, version, poses=None):
    version_dir = os.path.join(rig_dir, version)
    if poses is None:
        poses = sorted(os.path.basename(os.path.dirname(p)) for p in
                       glob.glob(os.path.join(version_dir, "pose*", "joints_3d.json")))
    names = [LANDMARK_NAMES[i] for i in landmarks_for_version(version)[0]]
    frames, rows = [], []
    for pose in poses:
        path = os.path.join(version_dir, pose, "joints_3d.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            result = json.load(f)
        row = np.full((len(names), 3), np.nan)
        for j, n in enumerate(names):
            v = result["joints"].get(n)
            if v is not None:
                row[j] = (v["x"], v["y"], v["z"])
        frames.append(pose)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No joints_3d.json found under {version_dir}")
    return names, frames, np.stack(rows)


def load_cameras(rig_dir, version):
    with open(os.path.join(rig_dir, version, "cameras.json")) as f:
        cams = json.load(f)
    keys = sorted(cams)
    return (keys,
            np.array([cams[k]["K"] for k in keys]),
            np.array([cams[k]["R"] for k in keys]),
            np.array([np.array(cams[k]["t"]).reshape(3) for k in keys]),
            np.array([[cams[k]["image_width"], cams[k]["image_height"]] for k in keys]),
            all(cams[k].get("source") == CALIBRATED_SOURCE_MARKER for k in keys))


def bone_lengths(joints, names):
    out = {}
    for label, a, b in BONES:
        if a in names and b in names:
            d = np.linalg.norm(joints[:, names.index(a), :] - joints[:, names.index(b), :], axis=1)
            out[label] = d
    return out


def consistency_report(joints, names, frames, calibrated):
    lengths = bone_lengths(joints, names)
    bones = {}
    for label, d in lengths.items():
        ok = d[~np.isnan(d)]
        if len(ok) == 0:
            continue
        bones[label] = {
            "per_frame_m": [None if np.isnan(x) else round(float(x), 4) for x in d],
            "mean_m": round(float(ok.mean()), 4),
            "std_mm": round(float(ok.std() * 1000), 1) if len(ok) > 1 else None,
            "cv_percent": round(float(ok.std() / ok.mean() * 100), 2) if len(ok) > 1 else None,
        }
    symmetry = {}
    for stem in ("thigh", "shank", "foot", "upper_arm", "forearm"):
        l, r = bones.get(f"{stem}_L"), bones.get(f"{stem}_R")
        if l and r:
            symmetry[stem] = round(abs(l["mean_m"] - r["mean_m"]) / ((l["mean_m"] + r["mean_m"]) / 2) * 100, 2)
    foot_idx = [names.index(n) for n in FOOT_JOINTS if n in names]
    foot_z = joints[:, foot_idx, 2]
    pelvis = pelvis_centre(joints, names)
    return {
        "frames": frames,
        "world_frame": "calibrated (Z up, metres, floor at Z=0)" if calibrated
                       else "ASSUMED camera ring - Z is not tied to the floor; do not trust height values",
        "calibrated": bool(calibrated),
        "bones": bones,
        "left_right_difference_percent": symmetry,
        "lowest_foot_joint_z_m": [None if np.all(np.isnan(r)) else round(float(np.nanmin(r)), 3) for r in foot_z],
        "pelvis_position_m": [[None if np.isnan(x) else round(float(x), 3) for x in p] for p in pelvis],
        "max_bone_cv_percent": max((b["cv_percent"] for b in bones.values() if b["cv_percent"] is not None), default=None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig_output_dir", help="e.g. 4camera/output")
    ap.add_argument("--version", default="full_body", choices=["full_body", "waist_down"])
    ap.add_argument("--poses", nargs="*", default=None, help="pose folders in order (default: all)")
    args = ap.parse_args()

    names, frames, joints = load_sequence(args.rig_output_dir, args.version, args.poses)
    keys, K, R, t, size, calibrated = load_cameras(args.rig_output_dir, args.version)
    out_dir = os.path.join(args.rig_output_dir, args.version, "sequence")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "joints_world.npz"), joints=joints, names=np.array(names),
             frames=np.array(frames), cam_keys=np.array(keys), K=K, R=R, t=t, image_size=size,
             calibrated=calibrated)
    report = consistency_report(joints, names, frames, calibrated)
    with open(os.path.join(out_dir, "consistency.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"{args.version}: {len(frames)} frames x {len(names)} joints -> {out_dir}")
    print(f"  world frame: {report['world_frame']}")
    for k, b in report["bones"].items():
        if b["cv_percent"] is not None:
            print(f"  {k:15s} mean {b['mean_m']*100:5.1f} cm  std {b['std_mm']:5.1f} mm  CV {b['cv_percent']:.1f}%")
    print(f"  lowest foot joint Z per frame (m): {report['lowest_foot_joint_z_m']}")


if __name__ == "__main__":
    main()
