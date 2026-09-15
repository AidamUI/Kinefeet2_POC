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


def effective_projection_matrices(cameras_in_order, records, max_aspect_drift=0.03):
    """P for each camera, rescaled to the resolution that photo was actually
    taken at when that differs from the resolution cameras.json was built for.

    cameras.json's K is tied to one exact frame geometry - fx/fy scale with
    resolution, cx/cy are pixel coordinates in that specific frame. Photos
    re-exported through a different encoder/tool commonly land a few percent
    off that (e.g. 639x475 vs a calibration done at 665x499): same framing,
    slightly different pixel count. Left uncorrected, that mismatch silently
    feeds MediaPipe's pixel coordinates into a K that describes a different
    canvas, which biases every triangulated point by roughly that same
    percentage - not usually catastrophic, but real, and avoidable, since
    cameras.json carries K/R/t (not just the baked P) precisely so this can be
    corrected: a resolution change is just a diagonal scale of K.

    A *cropped* photo cannot be corrected this way - cropping moves the
    principal point by an amount nobody recorded - so this only rescales when
    the aspect ratio still matches (same framing, different pixel count) and
    otherwise falls back to the camera's own P unchanged, exactly as before.
    """
    warned = set()
    matrices = []
    for i, camera in enumerate(cameras_in_order):
        calib_w, calib_h = camera.get("image_width"), camera.get("image_height")
        photo_w, photo_h = None, None
        if i < len(records):
            photo_w = records[i].get("image_width")
            photo_h = records[i].get("image_height")

        if not (calib_w and calib_h and photo_w and photo_h) or (
            (photo_w, photo_h) == (calib_w, calib_h)
        ):
            matrices.append(np.array(camera["P"]))
            continue

        calib_aspect = calib_w / calib_h
        photo_aspect = photo_w / photo_h
        if abs(photo_aspect - calib_aspect) > max_aspect_drift * calib_aspect:
            if i not in warned:
                print(f"    [!] cam {i}: photo is {photo_w}x{photo_h}, calibrated at "
                      f"{calib_w}x{calib_h} - aspect ratio differs too much to be a "
                      f"resize (likely cropped). Using calibration-resolution K "
                      f"uncorrected; this camera's triangulation will be biased.")
                warned.add(i)
            matrices.append(np.array(camera["P"]))
            continue

        sx, sy = photo_w / calib_w, photo_h / calib_h
        K = np.array(camera["K"], dtype=np.float64).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        R = np.array(camera["R"], dtype=np.float64)
        t = np.array(camera["t"], dtype=np.float64).reshape(3, 1)
        matrices.append(K @ np.hstack([R, t]))
        if i not in warned:
            print(f"    cam {i}: rescaling K {calib_w}x{calib_h} (calibrated) -> "
                  f"{photo_w}x{photo_h} (this photo)")
            warned.add(i)
    return matrices


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
    # Per-photo P, corrected for any harmless resolution difference between
    # calibration and capture (see effective_projection_matrices).
    P_by_cam = effective_projection_matrices(cam_list, records)
    landmark_indices, _ = landmarks_for_version(version)

    # Get image dimensions (use first image)
    img_width = records[0].get("image_width", 1)
    img_height = records[0].get("image_height", 1)
    
    # Calculate person scale from 2D keypoints (average across all views)
    # Use shoulder width as reference - it's pose-invariant (unlike torso height)
    person_scales = []
    for record in records:
        landmarks = record["landmarks"]
        # Get left_shoulder (11), right_shoulder (12)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        if (left_shoulder["visibility"] >= min_visibility and
            right_shoulder["visibility"] >= min_visibility):
            
            # Calculate shoulder width in pixels
            shoulder_width = np.sqrt((left_shoulder["x_px"] - right_shoulder["x_px"])**2 +
                                    (left_shoulder["y_px"] - right_shoulder["y_px"])**2)
            person_scales.append(shoulder_width)
    
    # Use average person scale, fallback to image diagonal if not enough data
    if person_scales:
        person_scale = np.mean(person_scales)
    else:
        person_scale = np.sqrt(img_width**2 + img_height**2)
        print(f"  [!] WARNING: Could not calculate person scale, using image diagonal")

    joints_3d = {}
    diagnostics = {}

    for idx in landmark_indices:
        name = LANDMARK_NAMES[idx]
        P_list, uv_list, w_list, used_cams = [], [], [], []

        n = min(len(records), len(cam_list))
        for i in range(n):
            lm = records[i]["landmarks"][idx]
            if lm["visibility"] >= min_visibility:
                P_list.append(P_by_cam[i])
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
        mean_err_px = float(np.mean(errs))
        max_err_px = float(np.max(errs))
        
        joints_3d[name] = {"index": idx, "x": X[0], "y": X[1], "z": X[2]}
        diagnostics[name] = {
            "status": "ok",
            "views_used": used_cams,
            "mean_reprojection_error_px": mean_err_px,
            "max_reprojection_error_px": max_err_px,
            "mean_reprojection_error_normalized": mean_err_px / person_scale,
            "max_reprojection_error_normalized": max_err_px / person_scale,
        }

    return {"pose": pose_name, "version": version,
            "joints": joints_3d, "diagnostics": diagnostics,
            "image_width": img_width, "image_height": img_height,
            "person_scale_px": float(person_scale)}


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

            errs_px = [d["mean_reprojection_error_px"] for d in result["diagnostics"].values()
                       if d["status"] == "ok"]
            errs_norm = [d["mean_reprojection_error_normalized"] for d in result["diagnostics"].values()
                         if d["status"] == "ok"]
            if errs_px:
                img_w = result.get("image_width", 1)
                img_h = result.get("image_height", 1)
                person_scale = result.get("person_scale_px", 1)
                mean_err_px = np.mean(errs_px)
                mean_err_pct = np.mean(errs_norm) * 100
                
                print(f"  mean reprojection error: {mean_err_px:.1f} px ({mean_err_pct:.2f}% of shoulder width)")
                print(f"  image size: {img_w}x{img_h}, person scale: {person_scale:.0f} px (shoulder width)")
                print(f"  rule of thumb: <10% is excellent, 10-25% is good, >25% means check rig measurements")

            out_path = os.path.join(OUTPUT_ROOT, version, pose_name, "joints_3d.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  saved -> {out_path}")

    print("\nDone. Next: run 05_export_and_visualize.py to turn these into "
          "viewable 3D models.")


if __name__ == "__main__":
    main()
