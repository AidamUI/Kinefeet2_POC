"""
12_view_global.py
-----------------
Interactive 3D viewer for the reconstructed sequence, built on Rerun
(pip install rerun-sdk). Writes a .rrd file and optionally opens it.

What you get
  * A timeline ("pose") to step or play through the poses / frames.
  * 3D tabs - the same motion shown in different reference frames:
        Global            the calibrated world frame (floor grid, cameras)
        Camera 1 ... N    motion with respect to camera i (that camera is the origin)
        Pelvis            motion relative to the mid-hip point
  * Per-camera photo panels with the detected 2D landmarks (green) and the
    3D result projected back (red) - the reprojection check, per frame.
  * A left panel (Rerun's blueprint panel) with a tree of every layer:
    skeleton, SMPL-X mesh (if script 10 was run), cameras, floor, pelvis
    trajectory. Tick or untick to show or hide; drag a layer into a tab to
    add it.

USAGE   python scripts/12_view_global.py <rig_output_dir> [--version full_body]
                                         [--data-dir <rig>/data] [--open | --no-open]
        then:  rerun <rig_output_dir>/<version>/sequence/kinefeet.rrd

INPUT   <rig_output_dir>/<version>/sequence/joints_world.npz   (script 09)
        optional <rig_output_dir>/<version>/sequence/smplx_world.npz  (script 10)
        optional <rig_data_dir>/<version>/<pose>/NN.<ext> photos, and
                 <rig_output_dir>/<version>/<pose>/keypoints_2d/NN.json
OUTPUT  <rig_output_dir>/<version>/sequence/kinefeet.rrd
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import LANDMARK_NAMES, landmarks_for_version
from frames import world_to_camera, camera_center, to_pelvis_frame, camera_to_display

BLUE = (55, 118, 200)
RED = (214, 69, 65)
GREY = (120, 128, 132)
GREEN = (60, 170, 90)


def joint_color(name):
    if name.startswith("left_"):
        return BLUE
    if name.startswith("right_"):
        return RED
    return GREY


def bone_strips(points, names, connections):
    """Line segments (as 2-point strips) for bones with both ends present."""
    idx = {n: i for i, n in enumerate(names)}
    strips, colors = [], []
    for a, b in connections:
        na, nb = LANDMARK_NAMES[a], LANDMARK_NAMES[b]
        if na in idx and nb in idx:
            pa, pb = points[idx[na]], points[idx[nb]]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                strips.append([pa, pb])
                colors.append(joint_color(na) if na.split("_")[0] == nb.split("_")[0] else GREY)
    return strips, colors


def log_skeleton(rr, entity, points, names, connections):
    ok = ~np.isnan(points).any(axis=1)
    rr.log(f"{entity}/joints", rr.Points3D(points[ok], radii=0.012,
                                           colors=[joint_color(n) for n, k in zip(names, ok) if k],
                                           labels=[n for n, k in zip(names, ok) if k],
                                           show_labels=False))
    strips, colors = bone_strips(points, names, connections)
    rr.log(f"{entity}/bones", rr.LineStrips3D(strips, colors=colors, radii=0.006) if strips
           else rr.LineStrips3D([]))


def floor_grid(extent=2.0, step=0.5):
    lines = []
    for v in np.arange(-extent, extent + 1e-9, step):
        lines.append([[v, -extent, 0], [v, extent, 0]])
        lines.append([[-extent, v, 0], [extent, v, 0]])
    return lines


def find_photo(data_dir, version, pose, cam_index):
    stem = f"{cam_index + 1:02d}"
    for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
        p = os.path.join(data_dir, version, pose, f"{stem}.{ext}")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig_output_dir")
    ap.add_argument("--version", default="full_body", choices=["full_body", "waist_down"])
    ap.add_argument("--data-dir", default=None, help="photo folder (default: <rig>/data)")
    ap.add_argument("--open", dest="open_viewer", action="store_true", default=False)
    args = ap.parse_args()

    import rerun as rr
    import rerun.blueprint as rrb

    seq_dir = os.path.join(args.rig_output_dir, args.version, "sequence")
    seq_path = os.path.join(seq_dir, "joints_world.npz")
    if not os.path.exists(seq_path):
        raise SystemExit(f"{seq_path} not found - run scripts/09_build_sequence.py first")
    d = np.load(seq_path, allow_pickle=False)
    joints, names, frames = d["joints"], [str(n) for n in d["names"]], [str(f) for f in d["frames"]]
    cam_keys, K, R, t, size = [str(k) for k in d["cam_keys"]], d["K"], d["R"], d["t"], d["image_size"]
    calibrated = bool(d["calibrated"])
    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.rig_output_dir.rstrip("/\\"))), "data")
    connections = landmarks_for_version(args.version)[1]
    n_cam = len(cam_keys)

    smpl = None
    smpl_path = os.path.join(seq_dir, "smplx_world.npz")
    if os.path.exists(smpl_path):
        smpl = np.load(smpl_path, allow_pickle=False)

    rr.init("kinefeet_global", recording_id=f"kinefeet-{args.version}")
    out = os.path.join(seq_dir, "kinefeet.rrd")
    rr.save(out)

    # --- static scene: floor, cameras ------------------------------------
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    if calibrated:
        rr.log("/world/floor", rr.LineStrips3D(floor_grid(), colors=(200, 200, 200), radii=0.002), static=True)
    for i, key in enumerate(cam_keys):
        R_cw = R[i].T                       # camera -> world rotation
        rr.log(f"/world/cameras/{key}",
               rr.Transform3D(translation=camera_center(R[i], t[i]), mat3x3=R_cw), static=True)
        rr.log(f"/world/cameras/{key}",
               rr.Pinhole(image_from_camera=K[i], resolution=[int(size[i][0]), int(size[i][1])],
                          image_plane_distance=0.4, camera_xyz=rr.ViewCoordinates.RDF), static=True)
    # pelvis trajectory across frames (static polyline)
    pel = 0.5 * (joints[:, names.index("left_hip"), :] + joints[:, names.index("right_hip"), :])
    if len(pel) > 1 and not np.isnan(pel).any():
        rr.log("/world/pelvis_path", rr.LineStrips3D([pel], colors=(230, 160, 40), radii=0.008), static=True)
        rr.log("/world/pelvis_path/points", rr.Points3D(pel, colors=(230, 160, 40), radii=0.02), static=True)

    # --- time-varying: skeleton in every frame ---------------------------
    pelvis_local = to_pelvis_frame(joints, names)
    for f, pose in enumerate(frames):
        rr.set_time("pose", sequence=f)
        rr.log("/world/text", rr.TextLog(f"{pose}"))
        log_skeleton(rr, "/world/skeleton", joints[f], names, connections)
        log_skeleton(rr, "/pelvis/skeleton", pelvis_local[f], names, connections)
        for i, key in enumerate(cam_keys):
            pts_c = camera_to_display(world_to_camera(joints[f], R[i], t[i]))
            log_skeleton(rr, f"/{key}_frame/skeleton", pts_c, names, connections)
            rr.log(f"/{key}_frame/origin", rr.Points3D([[0, 0, 0]], radii=0.03, colors=(40, 40, 40),
                                                       labels=[f"{key} (origin)"]))
        if smpl is not None and f < len(smpl["vertices"]):
            rr.log("/world/smplx", rr.Mesh3D(vertex_positions=smpl["vertices"][f],
                                              triangle_indices=smpl["faces"],
                                              albedo_factor=(190, 175, 160, 255)))

        # per-camera photo with detected (green) and reprojected (red) landmarks
        for i, key in enumerate(cam_keys):
            photo = find_photo(data_dir, args.version, pose, i)
            if photo is None:
                continue
            import cv2
            img = cv2.imread(photo)
            if img is None:
                continue
            h, w = img.shape[:2]
            rr.log(f"/world/cameras/{key}/image", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            sx, sy = w / float(size[i][0]), h / float(size[i][1])
            Ks = K[i].copy()
            Ks[0, :] *= sx
            Ks[1, :] *= sy
            rr.log(f"/world/cameras/{key}", rr.Pinhole(image_from_camera=Ks, resolution=[w, h],
                                                       image_plane_distance=0.4,
                                                       camera_xyz=rr.ViewCoordinates.RDF))
            kp_path = os.path.join(args.rig_output_dir, args.version, pose, "keypoints_2d", f"{i + 1:02d}.json")
            if os.path.exists(kp_path):
                with open(kp_path) as fh:
                    lms = json.load(fh)["landmarks"]
                det = [(lms[LANDMARKS_BY_NAME[n]]["x_px"], lms[LANDMARKS_BY_NAME[n]]["y_px"])
                       for n in names]
                rr.log(f"/world/cameras/{key}/image/detected", rr.Points2D(det, colors=GREEN, radii=3))
            cam_pts = world_to_camera(joints[f], R[i], t[i])
            ok = (~np.isnan(cam_pts).any(axis=1)) & (cam_pts[:, 2] > 1e-6)
            proj = (Ks @ cam_pts[ok].T).T
            proj = proj[:, :2] / proj[:, 2:3]
            rr.log(f"/world/cameras/{key}/image/reprojected", rr.Points2D(proj, colors=RED, radii=3))

    # --- blueprint: left panel + tabs of reference frames ------------------
    tabs = [rrb.Spatial3DView(origin="/world", name="Global (world)")]
    for key in cam_keys:
        tabs.append(rrb.Spatial3DView(origin=f"/{key}_frame", name=f"Relative to {key}"))
    tabs.append(rrb.Spatial3DView(origin="/pelvis", name="Relative to pelvis"))
    cams2d = [rrb.Spatial2DView(origin=f"/world/cameras/{k}/image", name=k) for k in cam_keys]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(rrb.Tabs(*tabs), rrb.Grid(*cams2d), column_shares=[3, 2]),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )
    rr.send_blueprint(blueprint)
    print(f"wrote {out}")
    print(f"  {len(frames)} frames, {n_cam} cameras, calibrated={calibrated}, "
          f"smplx={'yes' if smpl is not None else 'no'}")
    print(f"  open with:  rerun {out}")
    if args.open_viewer:
        import subprocess
        subprocess.Popen([sys.executable, "-m", "rerun", out])


LANDMARKS_BY_NAME = {v: k for k, v in LANDMARK_NAMES.items()}

if __name__ == "__main__":
    main()
