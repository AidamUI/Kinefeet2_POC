"""
11_export_opensim.py
--------------------
Turns the world-frame sequence (script 09) into OpenSim inputs and, when the
`opensim` Python package is installed, runs model scaling and inverse
kinematics to get joint angles (hip, knee, ankle, pelvis...) for each frame.

  1. markers.trc    MediaPipe landmarks renamed to the BlazePose marker set,
                    converted to OpenSim axes (Y up, X forward, Z right).
  2. scale          a generic musculoskeletal model is scaled to this subject
                    from the first frame (assumed to be a standing pose).
  3. inverse kinematics   joint angles that best reproduce the markers.

The model, marker set and scaling / IK setups in opensim_setup/ come from
Pose2Sim (BSD-3-Clause, see opensim_setup/LICENSE_Pose2Sim).

Axis handling
  The calibrated world frame is Z up. OpenSim wants Y up, +X forward and +Z
  to the subject's right, so the markers are rotated (x, y, z) -> (x, z, -y)
  and then about the vertical so that the first frame's right-hip minus
  left-hip vector points along +Z. The same heading rotation is applied to
  every frame, so later heading changes are preserved. Frames are 1 s apart
  (a placeholder: separate photo poses carry no real timing).

Limits worth knowing
  * MediaPipe gives 3 points per foot (ankle, heel, big toe). Ankle flexion
    is meaningful; subtalar (inversion/eversion) and toe motion are weakly
    constrained, so treat those angles as indicative only.
  * Heights are only meaningful for a calibrated rig (floor at Z=0).

USAGE   python scripts/11_export_opensim.py <rig_output_dir> [--version full_body]
                                             [--mass 70] [--trc-only]

INPUT   <rig_output_dir>/<version>/sequence/joints_world.npz   (script 09)
OUTPUT  <rig_output_dir>/<version>/sequence/opensim/
            markers.trc, static.trc
            model_scaled.osim, joint_angles.csv, ik_report.json   (needs opensim)
"""

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from frames import zup_to_yup

SETUP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "opensim_setup"))

# MediaPipe landmark name -> marker name in Pose2Sim's BlazePose marker set
MARKER_MAP = {
    "nose": "Nose",
    "left_shoulder": "LShoulder", "right_shoulder": "RShoulder",
    "left_elbow": "LElbow", "right_elbow": "RElbow",
    "left_wrist": "LWrist", "right_wrist": "RWrist",
    "left_hip": "LHip", "right_hip": "RHip",
    "left_knee": "LKnee", "right_knee": "RKnee",
    "left_ankle": "LAnkle", "right_ankle": "RAnkle",
    "left_heel": "LHeel", "right_heel": "RHeel",
    "left_foot_index": "LBigToe", "right_foot_index": "RBigToe",
    "left_pinky": "LPinky", "right_pinky": "RPinky",
    "left_index": "LIndex", "right_index": "RIndex",
    "left_thumb": "LThumb", "right_thumb": "RThumb",
}

ANGLE_COLUMNS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
    "ankle_angle_r", "subtalar_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
    "ankle_angle_l", "subtalar_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]


def heading_rotation(joints_yup, names):
    """Rotation about Y mapping the first frame's right-minus-left hip vector to +Z."""
    li, ri = names.index("left_hip"), names.index("right_hip")
    for f in range(len(joints_yup)):
        lat = joints_yup[f, ri] - joints_yup[f, li]
        if not np.isnan(lat).any():
            lx, lz = lat[0], lat[2]
            n = np.hypot(lx, lz)
            if n < 1e-9:
                continue
            th = np.arctan2(-lx, lz)
            c, s = np.cos(th), np.sin(th)
            return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.eye(3)


def to_opensim_axes(joints_world, names):
    """(F, J, 3) world (Z up) -> OpenSim ground frame (Y up, X forward, Z right)."""
    yup = zup_to_yup(joints_world)
    return yup @ heading_rotation(yup, names).T


def write_trc(path, joints_os, names, rate=1.0):
    """Write a TRC marker file (metres) for the MARKER_MAP joints present."""
    cols = [(n, MARKER_MAP[n]) for n in names if n in MARKER_MAP]
    idx = [names.index(n) for n, _ in cols]
    n_frames = len(joints_os)
    with open(path, "w", newline="") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:g}\t{rate:g}\t{n_frames}\t{len(cols)}\tm\t{rate:g}\t1\t{n_frames}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(m for _, m in cols) + "\t\t\n")
        f.write("\t\t" + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(cols))) + "\n")
        f.write("\n")
        for k in range(n_frames):
            vals = []
            for j in idx:
                p = joints_os[k, j]
                vals.extend("" if np.isnan(v) else f"{v:.6f}" for v in p)
            f.write(f"{k + 1}\t{k / rate:.6f}\t" + "\t".join(vals) + "\n")
    return [m for _, m in cols]


def _set(root, path, value):
    nodes = root.findall(path)
    if not nodes:
        raise KeyError(f"{path} not found in OpenSim setup file")
    for node in nodes:
        node.text = str(value)


# OpenSim resolves every path inside a setup file relative to that file, and
# mishandles absolute Windows paths, so setups use bare file names and the
# generic model / marker set are copied next to them.
def stage_generic_files(out_dir):
    for name in ("Model_Pose2Sim_simple.osim", "Markers_BlazePose.xml"):
        shutil.copy2(os.path.join(SETUP_DIR, name), os.path.join(out_dir, name))


def configure_scaling(out_dir, trc, t0, t1, mass):
    tree = ET.parse(os.path.join(SETUP_DIR, "Scaling_Setup_Pose2Sim_Blazepose.xml"))
    r = tree.getroot()
    _set(r, ".//ScaleTool/mass", mass)
    _set(r, ".//GenericModelMaker/model_file", "Model_Pose2Sim_simple.osim")
    _set(r, ".//GenericModelMaker/marker_set_file", "Markers_BlazePose.xml")
    _set(r, ".//ModelScaler/marker_file", trc)
    _set(r, ".//ModelScaler/time_range", f"{t0} {t1}")
    _set(r, ".//ModelScaler/output_model_file", "model_scaled.osim")
    _set(r, ".//ModelScaler/output_scale_file", "scale_factors.xml")
    _set(r, ".//MarkerPlacer/marker_file", trc)
    _set(r, ".//MarkerPlacer/output_model_file", "model_scaled_placed.osim")
    path = os.path.join(out_dir, "scaling_setup.xml")
    tree.write(path, xml_declaration=True, encoding="UTF-8")
    return path


def configure_ik(out_dir, trc, t0, t1):
    tree = ET.parse(os.path.join(SETUP_DIR, "IK_Setup_Pose2Sim_Blazepose.xml"))
    r = tree.getroot()
    _set(r, ".//InverseKinematicsTool/results_directory", "./")
    _set(r, ".//InverseKinematicsTool/model_file", "model_scaled.osim")
    _set(r, ".//InverseKinematicsTool/marker_file", trc)
    _set(r, ".//InverseKinematicsTool/time_range", f"{t0} {t1}")
    _set(r, ".//InverseKinematicsTool/output_motion_file", "ik.mot")
    path = os.path.join(out_dir, "ik_setup.xml")
    tree.write(path, xml_declaration=True, encoding="UTF-8")
    return path


def read_mot(path):
    """Return (columns, array) from an OpenSim .mot file."""
    with open(path) as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[start + 1].split()
    data = np.array([[float(x) for x in l.split()] for l in lines[start + 2:] if l.strip()])
    return cols, data


def run_opensim(out_dir, n_frames, mass):
    import opensim
    opensim.Logger.setLevelString("Error")
    stage_generic_files(out_dir)
    cwd = os.getcwd()
    os.chdir(out_dir)  # relative paths in the setup files resolve from here
    try:
        opensim.ScaleTool(configure_scaling(out_dir, "static.trc", 0.0, 1.0, mass)).run()
        opensim.InverseKinematicsTool(configure_ik(out_dir, "markers.trc", 0.0, float(n_frames - 1))).run()
    finally:
        os.chdir(cwd)
    cols, data = read_mot(os.path.join(out_dir, "ik.mot"))
    keep = [c for c in ANGLE_COLUMNS if c in cols]
    with open(os.path.join(out_dir, "joint_angles.csv"), "w") as f:
        f.write("frame,time_s," + ",".join(keep) + "\n")
        for k, row in enumerate(data):
            f.write(f"{k},{row[0]:.3f}," + ",".join(f"{row[cols.index(c)]:.2f}" for c in keep) + "\n")
    return {c: [round(float(v), 2) for v in data[:, cols.index(c)]] for c in keep}, cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig_output_dir")
    ap.add_argument("--version", default="full_body", choices=["full_body", "waist_down"])
    ap.add_argument("--mass", type=float, default=70.0, help="subject mass in kg (does not affect kinematics)")
    ap.add_argument("--trc-only", action="store_true", help="write TRC files and stop")
    args = ap.parse_args()

    seq = os.path.join(args.rig_output_dir, args.version, "sequence")
    d = np.load(os.path.join(seq, "joints_world.npz"), allow_pickle=False)
    names = [str(n) for n in d["names"]]
    joints_os = to_opensim_axes(d["joints"], names)
    out_dir = os.path.abspath(os.path.join(seq, "opensim"))
    os.makedirs(out_dir, exist_ok=True)

    markers = write_trc(os.path.join(out_dir, "markers.trc"), joints_os, names)
    write_trc(os.path.join(out_dir, "static.trc"), np.concatenate([joints_os[:1], joints_os[:1]]), names)
    print(f"wrote markers.trc ({len(joints_os)} frames, {len(markers)} markers) -> {out_dir}")
    if not bool(d["calibrated"]):
        print("  [!] cameras are NOT calibrated: heights and scale are not metric; angles are indicative only")
    if args.trc_only:
        return
    try:
        import opensim  # noqa: F401
    except ImportError:
        print("  opensim package not installed - stopping after TRC export. "
              "Install with: pip install opensim")
        return

    angles, cols = run_opensim(out_dir, len(joints_os), args.mass)
    err_cols, err = read_mot(os.path.join(out_dir, "_ik_marker_errors.sto"))
    rms_cm = [round(float(v) * 100, 2) for v in err[:, err_cols.index("marker_error_RMS")]]
    max_cm = [round(float(v) * 100, 2) for v in err[:, err_cols.index("marker_error_max")]]
    report = {"frames": [str(f) for f in d["frames"]], "joint_angles_deg": angles,
              "marker_error_rms_cm": rms_cm, "marker_error_max_cm": max_cm,
              "note": "1 s per pose is a placeholder. Subtalar and toe angles are weakly constrained "
                      "by three foot markers. Valid only if frame 0 is a standing pose."}
    with open(os.path.join(out_dir, "ik_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  IK marker error RMS (cm) per frame: {rms_cm}  max: {max_cm}")
    for c in ("pelvis_tilt", "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
              "ankle_angle_r", "ankle_angle_l"):
        if c in angles:
            print(f"  {c:16s} {angles[c]}")


if __name__ == "__main__":
    main()
