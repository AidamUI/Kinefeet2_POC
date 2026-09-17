"""
calibrate_extrinsics.py
-----------------------
Stage 2. Find where every camera actually is, in one shared metric world frame,
from synchronised views of the square-grid board.

    python calibrate_extrinsics.py
    python calibrate_extrinsics.py --debug

Reads output/intrinsics/*.json, writes output/extrinsics.json.

With ``--debug``, also writes one annotated image per camera per board
placement to ``output/debug/extrinsic/<placement>/<camera>.png``: green
circles are the detected board points, red crosses are those same points
reprojected through the final camera and board poses (after bundle
adjustment), and the header reports that view's RMS/max error.

How it works
    Every camera that sees the board gets the board's pose by solvePnP. If two
    cameras see the *same* board placement, composing one pose with the inverse
    of the other places them relative to each other - so a placement links
    cameras, and a camera links placements, and walking that graph puts
    everything in one frame. All of it is then refined together by bundle
    adjustment: one non-linear solve that minimises the total reprojection error
    over all camera poses and all board placements at once.

    Scale is metric and comes for free, because the board's 90 mm pitch is a
    measured physical quantity. Nothing here is guessed from a tape measure of
    the rig, which is why this is more accurate than an assumed camera ring
    built from typed-in radius and height numbers: any error in those numbers
    would go straight into the triangulated skeleton with nothing to catch it.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from scipy.optimize import least_squares

import common
import squaregrid as sg

OUT_PATH = os.path.join(common.HERE, "output", "extrinsics.json")

# World frame with +Z up out of the floor, for a board lying face-up on it.
# The board's own +Z points into its printed face and so into the floor, and Y
# is flipped alongside it to keep the frame right-handed.
FLIP_Z_UP = np.diag([1.0, -1.0, -1.0])


# --------------------------------------------------------------------------
# Rigid transform helpers.  T = (R, t) maps a point p as  R @ p + t.
# --------------------------------------------------------------------------


def compose(a, b):
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def invert(t):
    return t[0].T, -t[0].T @ t[1]


def to_vec(t):
    return np.concatenate([cv2.Rodrigues(t[0])[0].ravel(), t[1].ravel()])


def from_vec(v):
    return cv2.Rodrigues(v[:3].reshape(3, 1))[0], v[3:6].reshape(3, 1)


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def gather(cfg):
    """Detect the board in every camera image of every placement.

    Returns ``(observations, cameras, positions, sizes, focals)`` where an
    observation is ``(camera_index, position_index, object_points,
    image_points, image_path)``, and ``sizes`` / ``focals`` record each
    camera's frame size and the focal length its own view of the board
    implies.
    """
    extr = cfg["extrinsics"]
    camera_names = list(extr["cameras"])
    min_cells = int(cfg.get("detection", {}).get("min_squares", 12))
    use_corners = bool(cfg.get("detection", {}).get("use_square_corners", False))

    observations = []
    positions = []
    sizes = {}
    focals = {}
    for folder in extr["positions"]:
        images = common.list_images(folder)
        label = os.path.basename(os.path.normpath(common.resolve(folder)))
        print(f"\n  [{label}]  {len(images)} image(s)")
        if len(images) != len(camera_names):
            print(f"    [!] expected {len(camera_names)} images (one per camera), "
                  f"found {len(images)}. Files are matched to cameras in "
                  f"alphabetical order, so this placement is ambiguous - skipping.")
            continue

        p = len(positions)
        found_any = False
        for c, (name, path) in enumerate(zip(camera_names, images)):
            detection = common.detect_board(path, min_cells=min_cells)
            if detection is None:
                print(f"    {name:8s} board NOT found  {os.path.basename(path)}")
                continue
            if detection.mirrored:
                print(f"    {name:8s} [!] MIRRORED view - skipping. A left-right "
                      "flipped feed makes a left-handed camera.")
                continue
            obj, img = common.points_from_detection(detection, use_corners)
            observations.append((c, p, obj.astype(np.float64), img.astype(np.float64), path))
            found_any = True
            sizes[c] = detection.image_size
            focal = common.focal_from_board(detection)
            if focal:
                focals.setdefault(c, []).append(focal)
            print(f"    {name:8s} {detection.debug['n_cells']:2d}/24 squares  "
                  f"{os.path.basename(path)}")
        if found_any:
            positions.append(label)

    return observations, camera_names, positions, sizes, focals


def write_debug_reprojection(observations, intrinsics, T_cam, T_board, camera_names, positions):
    """One annotated image per observation, using the *final* camera and
    board poses (after bundle adjustment, when it ran).

    Green circles are the detected square points, red crosses are those same
    board points reprojected through the fitted camera pose, and the header
    reports this view's RMS/max error - the same visual language as
    calibrate_intrinsics.py's debug output, so the two stages read the same
    way.
    """
    out_root = os.path.join(common.HERE, "output", "debug", "extrinsic")
    for c, p, obj, img, path in observations:
        if c not in T_cam or p not in T_board:
            continue
        image = cv2.imread(path)
        if image is None:
            continue
        K, dist, _, _ = intrinsics[c]
        projected = project(obj, T_cam[c], T_board[p], K, dist)
        errs = np.linalg.norm(projected - img, axis=1)
        label = positions[p]
        name = camera_names[c]
        header = [f"{name}  @  {label}", f"reprojection RMS {np.sqrt((errs**2).mean()):.3f} px   "
                  f"max {errs.max():.3f} px"]
        annotated = common.draw_reprojection(image, img, projected, header)
        out = os.path.join(out_root, label)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, f"{name}.png"), annotated)
    print(f"  annotated reprojection -> {os.path.relpath(out_root, common.HERE)}")


def load_intrinsics(cfg, camera_names):
    """Map each camera to its (K, dist, image_size)."""
    extr = cfg["extrinsics"]
    shared = extr.get("share_intrinsics")
    per_camera = extr.get("per_camera_intrinsics") or {}

    out = {}
    for c, name in enumerate(camera_names):
        source = per_camera.get(name, shared)
        if source is None:
            raise SystemExit(
                f"No intrinsics configured for '{name}'. Set extrinsics."
                "share_intrinsics or extrinsics.per_camera_intrinsics in config.yaml."
            )
        path = os.path.join(common.HERE, "output", "intrinsics", f"{source}.json")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run calibrate_intrinsics.py first.")
        calib = common.load_json(path)
        out[c] = (
            np.array(calib["K"], dtype=np.float64),
            np.array(calib["dist_coeffs"], dtype=np.float64),
            (calib["image_width"], calib["image_height"]),
            source,
        )
    return out


def adapt_intrinsics(intrinsics, camera_names, sizes, focals, estimate_focal=False):
    """Check the intrinsics actually belong to these frames, and adjust if asked.

    Two different failures hide here, and neither shows up in reprojection
    error, because a wrong focal length is absorbed perfectly by putting the
    board further away. They surface only as every distance in the room being
    wrong by a constant factor - which is exactly the kind of bug that makes a
    rig "not work" while every number on screen looks fine.

      * the capture frames are a different *resolution* to the calibration
        frames. Recoverable: scale K by the resolution ratio.
      * the capture frames are a different *field of view* - a crop, a digital
        zoom, a lens or zoom change between sessions. Not recoverable, because
        the calibration simply describes a different optical setup.

    Returns the adapted intrinsics and a list of human-readable complaints.
    """
    print()
    adapted = {}
    complaints = []
    for c, name in enumerate(camera_names):
        K, dist, calib_size, source = intrinsics[c]
        size = sizes.get(c, calib_size)

        note = ""
        if size != calib_size:
            K, (sx, sy) = common.rescale_intrinsics(K, calib_size, size)
            note = f"  rescaled x{sx:.3f}/{sy:.3f} to {size[0]}x{size[1]}"
            if abs(sx - sy) > 0.01 * max(sx, sy):
                complaints.append(
                    f"{name}: calibration frames are {calib_size[0]}x{calib_size[1]} "
                    f"but capture frames are {size[0]}x{size[1]} - a different "
                    "aspect ratio, so these are crops, not resizes, and the "
                    "principal point cannot be recovered."
                )

        fov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
        print(f"  {name:8s} intrinsics '{source}' (measured at "
              f"{calib_size[0]}x{calib_size[1]}){note}")
        print(f"           fx {K[0, 0]:8.1f} px -> {fov:.1f} deg horizontal FOV")

        measured = focals.get(c)
        if measured:
            implied = float(np.median(measured))
            implied_fov = 2 * np.degrees(np.arctan(size[0] / (2 * implied)))
            ratio = implied / K[0, 0]
            print(f"           the board's own perspective in these frames implies "
                  f"fx {implied:.0f} px -> {implied_fov:.1f} deg")
            if abs(np.log(ratio)) > np.log(1.25):
                complaints.append(
                    f"{name}: supplied fx {K[0, 0]:.0f} px disagrees with the "
                    f"{implied:.0f} px the board's perspective implies "
                    f"(x{ratio:.2f}). Every distance this produces will be wrong "
                    f"by about that factor."
                )
                if estimate_focal:
                    K = K.copy()
                    aspect = intrinsics[c][0][1, 1] / intrinsics[c][0][0, 0]
                    K[0, 0] = implied
                    K[1, 1] = implied * aspect
                    # Distortion was fitted over a narrower field than these
                    # frames cover, so extrapolating it outwards does more harm
                    # than leaving it out.
                    dist = np.zeros_like(dist)
                    print(f"           --estimate-focal: using fx {implied:.0f} px, "
                          "distortion set to zero")
        adapted[c] = (K, dist, size, source)

    if complaints:
        print("\n  " + "!" * 66)
        print("  INTRINSICS DO NOT MATCH THESE FRAMES")
        for line in complaints:
            print(f"    - {line}")
        print(
            "\n    Fix it at the source: shoot the board photos with the very same\n"
            "    cameras, settings and output resolution you capture subjects at,\n"
            "    straight from the camera. Screenshots of a video window are\n"
            "    cropped and rescaled by whatever the window happened to be, and\n"
            "    intrinsics measured on them describe no real camera.\n"
            "    Meanwhile --estimate-focal re-derives the focal length from the\n"
            "    board itself so the geometry comes out at roughly the right\n"
            "    scale. Treat that as a stopgap, not a calibration."
        )
        print("  " + "!" * 66)
    return adapted, complaints


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


def initialise(observations, intrinsics, n_cameras, n_positions, world_frame):
    """Seed every camera and board pose by chaining solvePnP results.

    ``T_cam[c]`` maps world -> camera c, ``T_board[p]`` maps board -> world.
    """
    # Board pose as seen by each camera, per (camera, position).
    seen = {}
    for c, p, obj, img, _ in observations:
        K, dist, _, _ = intrinsics[c]
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        # Refine: the initial DLT-style solution is noticeably off when the
        # board is small in frame, which is exactly our case.
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        seen[(c, p)] = (cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1))

    if not seen:
        raise SystemExit("solvePnP failed for every view.")

    flip = FLIP_Z_UP if world_frame == "z_up" else np.eye(3)
    T_board = {0: (flip, np.zeros((3, 1)))}  # board -> world, gauge fixed here
    T_cam = {}

    # Alternate between "cameras that see a placed board" and "placements seen
    # by a placed camera" until nothing new can be reached.
    for _ in range(n_cameras + n_positions + 1):
        grew = False
        for (c, p), T_cam_board in seen.items():
            if c not in T_cam and p in T_board:
                T_cam[c] = compose(T_cam_board, invert(T_board[p]))
                grew = True
            elif c in T_cam and p not in T_board:
                T_board[p] = compose(invert(T_cam[c]), T_cam_board)
                grew = True
        if not grew:
            break

    return T_cam, T_board, seen


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------


def project(obj, T_cam, T_board, K, dist):
    """Project board-frame points through a board pose and a camera pose."""
    pose = compose(T_cam, T_board)
    rvec = cv2.Rodrigues(pose[0])[0]
    projected, _ = cv2.projectPoints(obj, rvec, pose[1], K, dist)
    return projected.reshape(-1, 2)


def bundle_adjust(observations, intrinsics, T_cam, T_board, refine_focal=False):
    """Refine all camera and board poses against every observation at once."""
    cam_ids = sorted(T_cam)
    board_ids = [p for p in sorted(T_board) if p != 0]
    n_cam, n_board = len(cam_ids), len(board_ids)

    x0 = [to_vec(T_cam[c]) for c in cam_ids] + [to_vec(T_board[p]) for p in board_ids]
    x0 = np.concatenate(x0) if x0 else np.zeros(0)
    if refine_focal:
        x0 = np.concatenate([x0, [1.0] * n_cam])

    fixed_board = T_board[0]

    def unpack(x):
        cams = {c: from_vec(x[6 * i : 6 * i + 6]) for i, c in enumerate(cam_ids)}
        base = 6 * n_cam
        boards = {0: fixed_board}
        for j, p in enumerate(board_ids):
            boards[p] = from_vec(x[base + 6 * j : base + 6 * j + 6])
        scales = (
            x[6 * (n_cam + n_board) :] if refine_focal else np.ones(n_cam)
        )
        return cams, boards, {c: scales[i] for i, c in enumerate(cam_ids)}

    def residuals(x):
        cams, boards, scales = unpack(x)
        out = []
        for c, p, obj, img, _ in observations:
            if c not in cams or p not in boards:
                continue
            K, dist, _, _ = intrinsics[c]
            if refine_focal:
                K = K.copy()
                K[0, 0] *= scales[c]
                K[1, 1] *= scales[c]
            out.append((project(obj, cams[c], boards[p], K, dist) - img).ravel())
        return np.concatenate(out) if out else np.zeros(1)

    # soft_l1 keeps one bad square from dragging every pose with it, without
    # needing an explicit outlier-rejection pass.
    result = least_squares(
        residuals, x0, loss="soft_l1", f_scale=2.0, method="trf", max_nfev=400
    )
    cams, boards, scales = unpack(result.x)
    return cams, boards, scales


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def errors_per_camera(observations, intrinsics, T_cam, T_board, camera_names):
    per_camera = {}
    all_errors = []
    for c, p, obj, img, _ in observations:
        if c not in T_cam or p not in T_board:
            continue
        K, dist, _, _ = intrinsics[c]
        d = np.linalg.norm(project(obj, T_cam[c], T_board[p], K, dist) - img, axis=1)
        per_camera.setdefault(camera_names[c], []).extend(d.tolist())
        all_errors.extend(d.tolist())
    return per_camera, np.array(all_errors)


def triangulation_check(observations, intrinsics, T_cam, T_board):
    """Independent accuracy number, in millimetres.

    Reprojection error says the model is self-consistent; it does not say the
    cameras are in the right places relative to each other. So triangulate the
    board's squares from the calibrated cameras and compare the result against
    the board itself, whose geometry is known exactly. Any error in the relative
    poses shows up here as the reconstructed board being the wrong size or
    shape.
    """
    by_position = {}
    for c, p, obj, img, _ in observations:
        if c not in T_cam:
            continue
        by_position.setdefault(p, []).append((c, obj, img))

    residuals = []
    scales = []
    for p, views in by_position.items():
        if len(views) < 2:
            continue
        # Index the points each view saw, by their board coordinates.
        seen = {}
        for c, obj, img in views:
            K, dist, _, _ = intrinsics[c]
            undistorted = cv2.undistortPoints(
                img.reshape(-1, 1, 2), K, dist, P=K
            ).reshape(-1, 2)
            P = K @ np.hstack(T_cam[c])
            for point, pixel in zip(obj, undistorted):
                seen.setdefault(tuple(np.round(point, 6)), []).append((P, pixel))

        model, solved = [], []
        for key, rows in seen.items():
            if len(rows) < 2:
                continue
            A = []
            for P, (u, v) in rows:
                A.append(u * P[2] - P[0])
                A.append(v * P[2] - P[1])
            _, _, vt = np.linalg.svd(np.array(A))
            X = vt[-1]
            if abs(X[3]) < 1e-12:
                continue
            model.append(np.array(key))
            solved.append(X[:3] / X[3])
        if len(model) < 4:
            continue

        model = np.array(model)
        solved = np.array(solved)
        # The triangulated points live in world coordinates and the model in
        # board coordinates, so compare them up to the rigid motion between the
        # two - what we are testing is shape and size, not placement.
        aligned, scale = _rigid_align(model, solved)
        residuals.extend(np.linalg.norm(aligned - solved, axis=1).tolist())
        scales.append(scale)

    return np.array(residuals), np.array(scales)


def _rigid_align(source, target):
    """Kabsch fit of source onto target. Returns the fitted points and the
    similarity scale that *would* have been needed (1.0 means correct size)."""
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    a = source - sc
    b = target - tc
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    R = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    scale = float(s.sum() / max((a**2).sum(), 1e-12))
    return (R @ a.T).T + tc, scale


# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--estimate-focal", action="store_true",
        help="when the supplied intrinsics disagree with these frames, re-derive "
             "the focal length from the board itself (a stopgap, not a fix)",
    )
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    extr = cfg["extrinsics"]

    common.banner("EXTRINSICS")
    observations, camera_names, positions, sizes, focals = gather(cfg)
    if not observations:
        raise SystemExit("\nThe board was not found in any extrinsic image.")

    intrinsics = load_intrinsics(cfg, camera_names)
    n_positions = max(p for _, p, _, _, _ in observations) + 1
    intrinsics, mismatch = adapt_intrinsics(
        intrinsics, camera_names, sizes, focals,
        estimate_focal=args.estimate_focal or bool(extr.get("estimate_focal", False)),
    )

    world_frame = extr.get("world_frame", "z_up")
    T_cam, T_board, _ = initialise(
        observations, intrinsics, len(camera_names), n_positions, world_frame
    )

    missing = [camera_names[c] for c in range(len(camera_names)) if c not in T_cam]
    if missing:
        print(f"\n  [!] not placed (never saw a board shared with a placed camera): "
              f"{', '.join(missing)}")
    if len(T_cam) < 2:
        raise SystemExit("\nFewer than 2 cameras could be placed.")

    _, initial = errors_per_camera(observations, intrinsics, T_cam, T_board, camera_names)
    print(f"\n  initial (per-camera PnP): RMS {np.sqrt((initial**2).mean()):.3f} px")

    if extr.get("bundle_adjust", True):
        T_cam, T_board, scales = bundle_adjust(
            observations, intrinsics, T_cam, T_board,
            refine_focal=bool(extr.get("refine_focal", False)),
        )
        per_camera, final = errors_per_camera(
            observations, intrinsics, T_cam, T_board, camera_names
        )
        print(f"  after bundle adjustment:  RMS {np.sqrt((final**2).mean()):.3f} px")
        if extr.get("refine_focal", False):
            for c in sorted(scales):
                print(f"      {camera_names[c]} focal x{scales[c]:.4f}")
    else:
        per_camera, final = errors_per_camera(
            observations, intrinsics, T_cam, T_board, camera_names
        )

    print("\n  reprojection error by camera:")
    for name, errs in per_camera.items():
        e = np.array(errs)
        print(f"      {name:8s} RMS {np.sqrt((e**2).mean()):6.3f} px   "
              f"max {e.max():6.3f} px   ({len(e)} points)")

    if args.debug:
        write_debug_reprojection(observations, intrinsics, T_cam, T_board, camera_names, positions)

    # ---- geometry report -------------------------------------------------
    print(f"\n  world frame: origin at the board's corner square in "
          f"'{positions[0]}'"
          + (", +Z up out of the floor" if world_frame == "z_up" else ", board frame"))
    print("\n  camera positions (metres):")
    centres = {}
    for c in sorted(T_cam):
        R, t = T_cam[c]
        centre = (-R.T @ t).ravel()
        centres[c] = centre
        forward = R[2]  # camera viewing direction in world coordinates
        print(f"      {camera_names[c]:8s} at ({centre[0]:+7.3f}, {centre[1]:+7.3f}, "
              f"{centre[2]:+7.3f})   looking ({forward[0]:+.2f}, {forward[1]:+.2f}, "
              f"{forward[2]:+.2f})")

    print("\n  baselines (metres):")
    ids = sorted(centres)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            print(f"      {camera_names[a]} - {camera_names[b]}: "
                  f"{np.linalg.norm(centres[a] - centres[b]):.3f}")

    residuals, scale_check = triangulation_check(
        observations, intrinsics, T_cam, T_board
    )
    if len(residuals):
        print(f"\n  triangulation check against the physical board:")
        print(f"      RMS 3D error {1000 * np.sqrt((residuals**2).mean()):6.2f} mm   "
              f"max {1000 * residuals.max():6.2f} mm")
        if len(scale_check):
            err = 100 * (np.mean(scale_check) - 1.0)
            print(f"      reconstructed board size is {err:+.2f}% off its true "
                  f"{sg.PITCH_MM:.0f} mm pitch")

    if len(positions) < 2:
        print("\n  [!] Only one board placement. Every camera pose then rests on a\n"
              "      single view of a board covering a small part of the frame, so\n"
              "      depth along each camera's own axis is weakly constrained and\n"
              "      bundle adjustment has nothing extra to pull against. Shoot 4-8\n"
              "      more placements, moving the board around the capture volume\n"
              "      (and tilting it) between each.")

    payload = {
        "world_frame": world_frame,
        "unit": "metres",
        "positions": positions,
        "rms_reprojection_error_px": float(np.sqrt((final**2).mean())),
        "cameras": {},
    }
    for c in sorted(T_cam):
        R, t = T_cam[c]
        K, dist, size, source = intrinsics[c]
        errs = np.array(per_camera.get(camera_names[c], [0.0]))
        payload["cameras"][camera_names[c]] = {
            "K": K.tolist(),
            "dist_coeffs": dist.tolist(),
            "image_width": size[0],
            "image_height": size[1],
            "intrinsics_source": source,
            "R": R.tolist(),
            "t": t.reshape(3, 1).tolist(),
            "camera_position_world": (-R.T @ t).ravel().tolist(),
            "rms_reprojection_error_px": float(np.sqrt((errs**2).mean())),
        }
    if len(residuals):
        payload["triangulation_rms_mm"] = float(1000 * np.sqrt((residuals**2).mean()))
    if mismatch:
        payload["intrinsics_warnings"] = mismatch
    common.save_json(OUT_PATH, payload)
    print("\n  -> output/extrinsics.json   (next: export_cameras.py)")


if __name__ == "__main__":
    main()
