"""
calibrate_intrinsics.py
-----------------------
Stage 1. Recover each camera's intrinsic matrix K and lens distortion from
photos of the Captury square-grid board, using ``cv2.calibrateCamera``.

    python calibrate_intrinsics.py                  # every camera in config.yaml
    python calibrate_intrinsics.py --camera cam01   # just one
    python calibrate_intrinsics.py --debug          # also write annotated images

Writes ``output/intrinsics/<camera>.json``.

What it does beyond a textbook calibrateCamera call:
  * refuses to mix image sizes (see common.check_image_sizes for why)
  * runs a second pass after dropping views whose reprojection error is a
    gross outlier, which is usually a blurred or badly clipped frame
  * reports the uncertainty of every estimated parameter, so you can see when
    a number is simply not constrained by the images you took
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

import common
import squaregrid as sg

OUT_DIR = os.path.join(common.HERE, "output", "intrinsics")


def _build_flags(cfg):
    """Translate the distortion-model switches in config.yaml into CALIB flags."""
    model = cfg.get("distortion", {})
    flags = 0
    if not model.get("tangential", True):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if not model.get("k3", True):
        flags |= cv2.CALIB_FIX_K3
    if model.get("rational", False):
        flags |= cv2.CALIB_RATIONAL_MODEL
    if model.get("thin_prism", False):
        flags |= cv2.CALIB_THIN_PRISM_MODEL
    if model.get("fix_aspect_ratio", False):
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if model.get("fix_principal_point", False):
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    return flags


def _per_view_rms(obj_pts, img_pts, K, dist, rvecs, tvecs):
    errors = []
    for o, i, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(o, rvec, tvec, K, dist)
        d = projected.reshape(-1, 2) - i.reshape(-1, 2)
        errors.append(float(np.sqrt((d**2).sum() / len(d))))
    return np.array(errors)


def _calibrate(obj_pts, img_pts, size, flags):
    rms, K, dist, rvecs, tvecs, std_int, _, _ = cv2.calibrateCameraExtended(
        obj_pts, img_pts, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-8),
    )
    return rms, K, dist, rvecs, tvecs, std_int.ravel()


def calibrate_camera(name, folder, cfg, debug=False):
    common.banner(f"INTRINSICS  [{name}]")
    print(f"  images from: {common.resolve(folder)}")

    min_cells = int(cfg.get("detection", {}).get("min_squares", 12))
    detections, paths = common.detect_folder(folder, min_cells=min_cells)
    if not detections:
        print("  [!] board not detected in any image - nothing to calibrate.")
        return None

    print(f"\n  board found in {len(detections)}/{len(common.list_images(folder))} images")
    uniform = common.report_image_sizes(detections, paths)

    size, keep, reject = common.check_image_sizes(detections, paths)
    if reject:
        print(f"  dropping {len(reject)} image(s) that do not match {size[0]}x{size[1]}:")
        for i in reject:
            w, h = detections[i].image_size
            print(f"        {os.path.basename(paths[i])}  ({w}x{h})")
    detections = [detections[i] for i in keep]
    paths = [paths[i] for i in keep]

    if len(detections) < 6:
        print(f"  [!] only {len(detections)} usable views (need >= 6). Aborting.")
        return None

    # A printed board cannot be photographed from behind, so a mirrored view is
    # either a flipped camera feed - in which case every frame is mirrored - or
    # a detection error. A minority is therefore always the error case, and
    # including even one corrupts the calibration, since its points are indexed
    # by a labelling that disagrees with every other view.
    flipped = [i for i, d in enumerate(detections) if d.mirrored]
    if flipped:
        if len(flipped) > len(detections) / 2:
            print(
                f"\n  [!] {len(flipped)} of {len(detections)} views are left-right "
                "mirrored.\n      That is a flipped camera feed, not a detection "
                "problem. Turn off the\n      webcam's 'mirror' / 'selfie' preview "
                "setting and re-shoot - a mirrored\n      feed yields a left-handed "
                "camera and wrecks triangulation."
            )
            return None
        print(f"\n  dropping {len(flipped)} mirrored view(s) (mislabelled by the "
              f"detector):")
        for i in flipped:
            print(f"        {os.path.basename(paths[i])}")
        keep_upright = [i for i in range(len(detections)) if i not in flipped]
        detections = [detections[i] for i in keep_upright]
        paths = [paths[i] for i in keep_upright]

    use_corners = bool(cfg.get("detection", {}).get("use_square_corners", False))

    # Planarity screen. Done before calibration, because a bowed board or a
    # smeared frame breaks the pinhole model itself and no amount of robust
    # fitting downstream recovers from having it in the set.
    limit_planar = float(cfg.get("max_planar_residual_px", 1.5))
    residuals = np.array([common.planar_residual_px(d, use_corners) for d in detections])
    flat = [i for i, r in enumerate(residuals) if r <= limit_planar]
    print(f"\n  flatness check (single-view homography residual, limit {limit_planar} px):"
          f" median {np.median(residuals):.2f} px")
    if len(flat) < 6:
        print(f"  [!] only {len(flat)} view(s) pass; keeping the {min(12, len(detections))} flattest instead.")
        flat = list(np.argsort(residuals)[: min(12, len(detections))])
    for i in range(len(detections)):
        if i not in flat:
            print(f"        not flat, dropped: {os.path.basename(paths[i])} "
                  f"({residuals[i]:.2f} px)")
    detections = [detections[i] for i in flat]
    paths = [paths[i] for i in flat]

    pairs = [common.points_from_detection(d, use_corners) for d in detections]
    obj_pts = [p[0] for p in pairs]
    img_pts = [p[1] for p in pairs]
    total = sum(len(p) for p in obj_pts)
    kind = "square corners" if use_corners else "square centres"
    print(f"\n  {total} {kind} across {len(obj_pts)} views, image size {size[0]}x{size[1]}")

    flags = _build_flags(cfg)
    rms, K, dist, rvecs, tvecs, std_int = _calibrate(obj_pts, img_pts, size, flags)
    errors = _per_view_rms(obj_pts, img_pts, K, dist, rvecs, tvecs)
    print(f"  pass 1: RMS = {rms:.4f} px over {len(obj_pts)} views")

    # Drop gross outliers (blur, motion, a half-clipped board) and re-fit. The
    # cut is relative to the median so it adapts to how good the set actually is.
    limit = float(cfg.get("outlier_factor", 3.0)) * float(np.median(errors))
    keep2 = [i for i, e in enumerate(errors) if e <= max(limit, 0.5)]
    if len(keep2) < len(errors) and len(keep2) >= 6:
        for i in range(len(errors)):
            if i not in keep2:
                print(f"        outlier dropped: {os.path.basename(paths[i])} "
                      f"({errors[i]:.2f} px)")
        obj_pts = [obj_pts[i] for i in keep2]
        img_pts = [img_pts[i] for i in keep2]
        paths = [paths[i] for i in keep2]
        detections = [detections[i] for i in keep2]
        rms, K, dist, rvecs, tvecs, std_int = _calibrate(obj_pts, img_pts, size, flags)
        errors = _per_view_rms(obj_pts, img_pts, K, dist, rvecs, tvecs)
        print(f"  pass 2: RMS = {rms:.4f} px over {len(obj_pts)} views")

    dist = dist.ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_x = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(size[1] / (2 * fy)))

    print(f"\n  fx = {fx:9.3f} +/- {std_int[0]:.3f} px")
    print(f"  fy = {fy:9.3f} +/- {std_int[1]:.3f} px")
    print(f"  cx = {cx:9.3f} +/- {std_int[2]:.3f} px   (image centre {size[0]/2:.1f})")
    print(f"  cy = {cy:9.3f} +/- {std_int[3]:.3f} px   (image centre {size[1]/2:.1f})")
    print(f"  horizontal FOV = {fov_x:.2f} deg,  vertical FOV = {fov_y:.2f} deg")
    labels = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
    print("  distortion:")
    for i, value in enumerate(dist):
        sigma = std_int[4 + i] if 4 + i < len(std_int) else float("nan")
        print(f"        {labels[i] if i < len(labels) else f'd{i}':>3} = "
              f"{value: .6f} +/- {sigma:.6f}")
    print(f"\n  worst view: {errors.max():.3f} px, median {np.median(errors):.3f} px")

    if rms > 1.0:
        print("  [!] RMS above 1 px. Usually means blurred frames, a board that is\n"
              "      not rigidly flat, or frames that were cropped/rescaled.")

    result = {
        "camera": name,
        "image_width": int(size[0]),
        "image_height": int(size[1]),
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms_reprojection_error_px": float(rms),
        "fov_x_deg": float(fov_x),
        "fov_y_deg": float(fov_y),
        "n_views": len(obj_pts),
        "n_points": int(sum(len(p) for p in obj_pts)),
        "point_type": "square_corners" if use_corners else "square_centres",
        "uniform_image_size": bool(uniform),
        "std_intrinsics": std_int.tolist(),
        "per_view_error_px": {
            os.path.basename(p): float(e) for p, e in zip(paths, errors)
        },
        "board": {
            "cols": sg.COLS, "rows": sg.ROWS,
            "pitch_mm": sg.PITCH_MM, "square_mm": sg.SQUARE_MM,
        },
    }
    common.save_json(os.path.join(OUT_DIR, f"{name}.json"), result)

    if debug:
        _write_debug(name, detections, paths)
    return result


def _write_debug(name, detections, paths):
    out = os.path.join(common.HERE, "output", "debug", name)
    os.makedirs(out, exist_ok=True)
    for detection, path in zip(detections, paths):
        image = cv2.imread(path)
        if image is None:
            continue
        cv2.imwrite(os.path.join(out, os.path.basename(path)), sg.draw(image, detection))
    print(f"  annotated detections -> {os.path.relpath(out, common.HERE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", default=None, help="calibrate only this camera")
    parser.add_argument("--debug", action="store_true", help="write annotated images")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    sources = cfg["intrinsics"]["sources"]
    if args.camera:
        if args.camera not in sources:
            raise SystemExit(f"Unknown camera '{args.camera}'. Known: {list(sources)}")
        sources = {args.camera: sources[args.camera]}

    results = {}
    for name, folder in sources.items():
        result = calibrate_camera(name, folder, cfg, debug=args.debug)
        if result:
            results[name] = result

    common.banner("SUMMARY")
    if not results:
        raise SystemExit("No camera calibrated.")
    for name, r in results.items():
        print(f"  {name:10s} RMS {r['rms_reprojection_error_px']:.4f} px   "
              f"{r['n_views']} views   fx {r['K'][0][0]:.1f}   "
              f"FOV {r['fov_x_deg']:.1f} deg")
    print(f"\n  -> output/intrinsics/   (next: calibrate_extrinsics.py)")


if __name__ == "__main__":
    main()
