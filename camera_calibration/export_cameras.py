"""
export_cameras.py
-----------------
Stage 3. Turn output/extrinsics.json into the cameras.json files the Kinefeet
pipeline reads, and check the capture frames actually match the calibration.

    python export_cameras.py
    python export_cameras.py --dry-run

Each target in config.yaml names a capture folder and where its cameras.json
goes. The format matches what scripts/04_triangulate_3d.py expects: an ordered
map cam_00, cam_01, ... whose order lines up with the alphabetical order of the
photos in each pose folder, each entry carrying K, R, t and the 3x4 projection
matrix P = K [R|t].

Also checks the capture frames against the calibration and says so, but this is
informational: a plain resolution difference (a different export tool, a codec
rounding dimensions to a multiple of 2 or 16) is corrected automatically, per
photo, by scripts/04_triangulate_3d.py rescaling K to match - see
``effective_projection_matrices`` there. Only a genuine crop is a real problem,
because it moves the principal point by an amount nobody recorded; that is what
gets flagged as CROPPED.
"""

from __future__ import annotations

import argparse
import os


import cv2
import numpy as np

import common

EXTRINSICS_PATH = os.path.join(common.HERE, "output", "extrinsics.json")

# Stamped into every exported camera so the rest of the pipeline can tell a real
# calibration from a synthesised one.
SOURCE_MARKER = "camera_calibration/export_cameras.py"


def pose_folders(folder):
    """Immediate subfolders of a capture folder that contain images."""
    folder = common.resolve(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path) and common.list_images(path):
            out.append((name, path))
    return out


def check_pose(path, expected, max_aspect_drift=0.03):
    """Compare one pose folder's frames against the calibrated frame sizes.

    Returns ``(verdict, detail)``. This is informational, not a gate: pixel
    count differing from the calibration is normal (a different export tool,
    a codec that rounds dimensions to a multiple of 2/16, ...) and
    scripts/04_triangulate_3d.py now corrects for it automatically per photo
    by rescaling K - see ``effective_projection_matrices`` there. What that
    correction *cannot* fix is a genuine crop, which moves the principal
    point by an amount nobody recorded, so that is what this actually flags:
    an aspect ratio that no longer matches, not a pixel count that doesn't.
    """
    images = common.list_images(path)
    sizes = []
    for image_path in images:
        image = cv2.imread(image_path)
        sizes.append(None if image is None else (image.shape[1], image.shape[0]))

    if len(sizes) != len(expected):
        return "MISMATCH", f"{len(sizes)} photo(s) for {len(expected)} cameras"

    worst_drift = 0.0
    exact = True
    for size, want in zip(sizes, expected):
        if size is None:
            return "MISMATCH", "unreadable image"
        if size != want:
            exact = False
        drift = abs(size[0] / size[1] - want[0] / want[1]) / (want[0] / want[1])
        worst_drift = max(worst_drift, drift)

    shapes = ", ".join(f"{w}x{h}" for w, h in sizes)
    if exact:
        return "OK", shapes
    if worst_drift <= max_aspect_drift:
        return "RESCALED", f"{shapes} (different size, same framing - K is auto-corrected)"
    return "CROPPED", f"{shapes} (aspect ratio off by {100 * worst_drift:.1f}% - not a plain resize)"


def projection_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def azimuth_deg(centre):
    """Bearing of a camera around the world +Z axis, for readability only."""
    return float(np.degrees(np.arctan2(centre[1], centre[0])) % 360.0)


def build(extrinsics, size, tolerance=0.02):
    """Build the cameras.json payload at a given capture resolution."""
    cameras = {}
    notes = []
    for i, (name, cam) in enumerate(extrinsics["cameras"].items()):
        K = np.array(cam["K"], dtype=np.float64)
        calib_size = (cam["image_width"], cam["image_height"])
        if size is not None and size != calib_size:
            K, (sx, sy) = common.rescale_intrinsics(K, calib_size, size)
            if abs(sx - sy) > tolerance * max(sx, sy):
                notes.append(
                    f"{name}: calibrated at {calib_size[0]}x{calib_size[1]}, "
                    f"capture is {size[0]}x{size[1]} - different aspect ratio, so "
                    "this is a crop and the rescale cannot be correct."
                )
        out_size = size or calib_size

        R = np.array(cam["R"], dtype=np.float64)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
        centre = np.array(cam["camera_position_world"], dtype=np.float64)
        cameras[f"cam_{i:02d}"] = {
            # 03_setup_camera_rig.py looks for this before overwriting the file
            # with its synthetic camera ring.
            "source": SOURCE_MARKER,
            "name": name,
            "angle_deg": round(azimuth_deg(centre), 2),
            "image_width": out_size[0],
            "image_height": out_size[1],
            "K": K.tolist(),
            "dist_coeffs": cam["dist_coeffs"],
            "R": R.tolist(),
            "t": t.tolist(),
            "camera_position_world": centre.tolist(),
            "P": projection_matrix(K, R, t).tolist(),
        }
    return cameras, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    if not os.path.exists(EXTRINSICS_PATH):
        raise SystemExit(f"Missing {EXTRINSICS_PATH}. Run calibrate_extrinsics.py first.")
    extrinsics = common.load_json(EXTRINSICS_PATH)

    common.banner("EXPORT CAMERAS")
    print(f"  world frame: {extrinsics['world_frame']}, units {extrinsics['unit']}")
    print(f"  extrinsic RMS: {extrinsics['rms_reprojection_error_px']:.3f} px")
    if "triangulation_rms_mm" in extrinsics:
        print(f"  triangulation check: {extrinsics['triangulation_rms_mm']:.2f} mm")
    if extrinsics.get("intrinsics_warnings"):
        print(f"  [!] carried forward from calibrate_extrinsics.py:")
        for line in extrinsics["intrinsics_warnings"]:
            print(f"        {line}")

    targets = cfg.get("export", {}).get("targets") or []
    if not targets:
        raise SystemExit("Nothing in config.yaml -> export.targets.")

    for target in targets:
        out_path = common.resolve(target["path"])
        capture_dir = target.get("capture_dir")
        print(f"\n  [{target.get('name', os.path.basename(os.path.dirname(out_path)))}]")

        # Export at the resolution the cameras were calibrated at - that is the
        # only frame geometry these projection matrices are valid for.
        cameras, notes = build(extrinsics, None)
        for line in notes:
            print(f"    [!] {line}")

        for key, cam in cameras.items():
            centre = cam["camera_position_world"]
            print(f"    {key} <- {cam['name']:8s} at ({centre[0]:+6.2f}, "
                  f"{centre[1]:+6.2f}, {centre[2]:+6.2f}) m, bearing "
                  f"{cam['angle_deg']:6.1f} deg,  "
                  f"{cam['image_width']}x{cam['image_height']}")

        if capture_dir:
            expected = [(c["image_width"], c["image_height"]) for c in cameras.values()]
            poses = pose_folders(capture_dir)
            if not poses:
                print(f"    no capture images under {common.resolve(capture_dir)}")
            else:
                print("    capture frames vs calibrated frames:")
                for name, path in poses:
                    verdict, detail = check_pose(path, expected)
                    print(f"      {name:10s} {verdict:9s} {detail}")
                if any(check_pose(p, expected)[0] == "CROPPED" for _, p in poses):
                    print("    [!] CROPPED poses were cut down after capture, not just "
                          "resized. A crop\n"
                          "        moves the principal point by an unrecorded amount, so "
                          "no K describes\n"
                          "        them - triangulation will be biased for these regardless "
                          "of calibration\n"
                          "        quality. Re-export those frames whole (uncropped); any "
                          "resolution is fine.")

        if args.dry_run:
            print(f"    (dry run, not writing {out_path})")
        else:
            common.save_json(out_path, cameras)

    print("\n  Note: cameras.json carries dist_coeffs, but "
          "scripts/04_triangulate_3d.py\n"
          "  triangulates raw pixel coordinates. Undistort the 2D keypoints "
          "before\n  triangulating to get the benefit of the distortion model.")


if __name__ == "__main__":
    main()
