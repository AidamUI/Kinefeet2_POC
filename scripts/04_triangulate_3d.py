"""
04_triangulate_3d.py
----------------------
The core step: combines the 2D MediaPipe landmarks (script 02) with the
camera projection matrices (script 03) to triangulate a 3D XYZ position
for every joint, for every pose, for both versions.

For each landmark index (e.g. "left_knee"), we gather its 2D pixel
position from every photo where it was detected with decent confidence,
then run multi-view DLT triangulation (see utils.triangulate_point_dlt)
to find the 3D point that best explains all those 2D observations at
once.

INPUT   output/<version>/<pose>/keypoints_2d/*.json   (from script 02)
        output/<version>/cameras.json                  (from script 03)
OUTPUT  output/<version>/<pose>/joints_3d.json
"""

import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from utils import landmarks_for_version, LANDMARK_NAMES, reprojection_error

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_cameras(version):
    path = os.path.join(OUTPUT_ROOT, version, "cameras.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_keypoints_for_pose(version, pose_name):
    """
    Returns a list of per-photo keypoint records, sorted alphabetically by
    filename (which must match the angle order used in 03_setup_camera_rig.py -
    see the README naming convention).
    """
    kp_dir = os.path.join(OUTPUT_ROOT, version, pose_name, "keypoints_2d")
    files = sorted(glob.glob(os.path.join(kp_dir, "*.json")))
    records = []
    for f in files:
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def triangulate_pose(version, pose_name, cameras, min_visibility, min_views):
    records = load_keypoints_for_pose(version, pose_name)
    if not records:
        print(f"  (no 2D keypoints for {version}/{pose_name}, skipping - run script 02 first)")
        return None
    if len(records) != len(cameras):
        print(f"  [!] WARNING: {len(records)} photos but {len(cameras)} camera entries. "
              f"Cameras are assigned to photos by sorted order - double check "
              f"your filenames / config.yaml angles_deg match!")

    cam_list = list(cameras.values())
    landmark_indices, _ = landmarks_for_version(version)

    joints_3d = {}
    diagnostics = {}

    for idx in landmark_indices:
        name = LANDMARK_NAMES[idx]
        P_list, uv_list, w_list, used_cams = [], [], [], []

        n = min(len(records), len(cam_list))
        for i in range(n):
            lm = records[i]["landmarks"][idx]
            if lm["visibility"] >= min_visibility:
                P_list.append(np.array(cam_list[i]["P"]))
                uv_list.append((lm["x_px"], lm["y_px"]))
                w_list.append(lm["visibility"])
                used_cams.append(i)

        if len(P_list) < min_views:
            joints_3d[name] = None
            diagnostics[name] = {"status": "insufficient_views",
                                  "views_available": len(P_list),
                                  "views_required": min_views}
            continue

        from utils import triangulate_point_dlt
        X = triangulate_point_dlt(P_list, uv_list, w_list)

        errs = [reprojection_error(P, X, uv) for P, uv in zip(P_list, uv_list)]
        joints_3d[name] = {"index": idx, "x": X[0], "y": X[1], "z": X[2]}
        diagnostics[name] = {
            "status": "ok",
            "views_used": used_cams,
            "mean_reprojection_error_px": float(np.mean(errs)),
            "max_reprojection_error_px": float(np.max(errs)),
        }

    return {"pose": pose_name, "version": version,
            "joints": joints_3d, "diagnostics": diagnostics}


def main():
    cfg = load_config()
    min_visibility = cfg["triangulation"]["min_visibility"]
    min_views = cfg["triangulation"]["min_views"]

    for version in ["full_body", "waist_down"]:
        cameras = load_cameras(version)
        if cameras is None:
            print(f"[!] No cameras.json for '{version}' - run 03_setup_camera_rig.py first.")
            continue

        for pose_name in ["pose1", "pose2", "pose3"]:
            print(f"\n[{version}/{pose_name}]")
            result = triangulate_pose(version, pose_name, cameras, min_visibility, min_views)
            if result is None:
                continue

            n_ok = sum(1 for v in result["joints"].values() if v is not None)
            n_total = len(result["joints"])
            print(f"  triangulated {n_ok}/{n_total} joints")

            bad = [name for name, v in result["joints"].items() if v is None]
            if bad:
                print(f"  [!] could not triangulate: {bad}  "
                      f"(not enough confident views - check annotated photos "
                      f"from script 02, or lower min_visibility in config.yaml)")

            errs = [d["mean_reprojection_error_px"] for d in result["diagnostics"].values()
                    if d["status"] == "ok"]
            if errs:
                print(f"  mean reprojection error across joints: {np.mean(errs):.1f} px "
                      f"(rule of thumb: <15px is good, >40px means check your rig "
                      f"measurements or that the subject didn't move between shots)")

            out_path = os.path.join(OUTPUT_ROOT, version, pose_name, "joints_3d.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  saved -> {out_path}")

    print("\nDone. Next: run 05_export_and_visualize.py to turn these into "
          "viewable 3D models.")


if __name__ == "__main__":
    main()
