"""
05_export_and_visualize.py
-----------------------------
Turns each output/<version>/<pose>/joints_3d.json into:

  1. A .obj file  - vertices (joints) + line segments (bones), viewable
     in Blender, MeshLab, Windows 3D Viewer, online OBJ viewers, etc.
  2. A .ply file  - same skeleton as a colored point cloud + edges,
     viewable in MeshLab / CloudCompare.
  3. A .png       - a quick static preview rendered with matplotlib so
     you can sanity-check the result without opening a 3D program.
  4. An interactive .html (via plotly, if installed) that you can rotate
     with your mouse in a browser - this is the fastest way to actually
     inspect the reconstruction.

INPUT   output/<version>/<pose>/joints_3d.json
OUTPUT  output/<version>/<pose>/model.obj
        output/<version>/<pose>/model.ply
        output/<version>/<pose>/preview.png
        output/<version>/<pose>/interactive.html   (if plotly is installed)
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

sys.path.insert(0, os.path.dirname(__file__))
from utils import landmarks_for_version, LANDMARK_NAMES

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")


def load_joints(version, pose_name):
    path = os.path.join(OUTPUT_ROOT, version, pose_name, "joints_3d.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_points_and_edges(result, connections):
    """
    Returns:
      names: list of joint names that have a valid 3D position
      pts:   (N,3) array of their coordinates
      edges: list of (i,j) index pairs into `pts`/`names`, only for bones
             where BOTH endpoints were successfully triangulated
    """
    joints = result["joints"]
    valid_names = [n for n, v in joints.items() if v is not None]
    name_to_idx = {n: i for i, n in enumerate(valid_names)}
    pts = np.array([[joints[n]["x"], joints[n]["y"], joints[n]["z"]] for n in valid_names])

    # map landmark-index connections -> name connections -> valid point indices
    idx_to_name = LANDMARK_NAMES
    edges = []
    for a, b in connections:
        na, nb = idx_to_name[a], idx_to_name[b]
        if na in name_to_idx and nb in name_to_idx:
            edges.append((name_to_idx[na], name_to_idx[nb]))

    return valid_names, pts, edges


def export_obj(path, names, pts, edges):
    with open(path, "w") as f:
        f.write("# 3D body skeleton exported from MediaPipe multi-view triangulation\n")
        f.write("# vertices = joints, lines = bones\n")
        for n, p in zip(names, pts):
            f.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}  # {n}\n")
        # OBJ line elements use 1-based indexing
        for i, j in edges:
            f.write(f"l {i + 1} {j + 1}\n")


def export_ply(path, names, pts, edges):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for i, j in edges:
            f.write(f"{i} {j}\n")


def render_preview(path, names, pts, edges, title):
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in edges:
        xs, ys, zs = zip(pts[i], pts[j])
        ax.plot(xs, ys, zs, c="steelblue", linewidth=2)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="crimson", s=25, depthshade=True)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)  [up]")

    # Equal aspect ratio so the body doesn't look stretched/squashed.
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2.0
    mid = pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.view_init(elev=10, azim=-70)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def render_interactive(path, names, pts, edges, title):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("    (plotly not installed - skipping interactive .html; "
              "`pip install plotly` if you want it)")
        return

    edge_x, edge_y, edge_z = [], [], []
    for i, j in edges:
        edge_x += [pts[i, 0], pts[j, 0], None]
        edge_y += [pts[i, 1], pts[j, 1], None]
        edge_z += [pts[i, 2], pts[j, 2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                                line=dict(color="steelblue", width=6), name="bones"))
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+text",
                                marker=dict(size=5, color="crimson"),
                                text=names, textposition="top center", name="joints"))
    fig.update_layout(title=title, scene=dict(aspectmode="data",
                                               xaxis_title="X (m)", yaxis_title="Y (m)",
                                               zaxis_title="Z (m)"))
    fig.write_html(path)


def process(version, pose_name, connections):
    result = load_joints(version, pose_name)
    if result is None:
        print(f"  (no joints_3d.json for {version}/{pose_name}, skipping - run script 04 first)")
        return

    names, pts, edges = get_points_and_edges(result, connections)
    if len(pts) == 0:
        print(f"  [!] zero valid joints for {version}/{pose_name}, nothing to export")
        return

    out_dir = os.path.join(OUTPUT_ROOT, version, pose_name)
    title = f"{version} / {pose_name}  ({len(pts)} joints)"

    export_obj(os.path.join(out_dir, "model.obj"), names, pts, edges)
    export_ply(os.path.join(out_dir, "model.ply"), names, pts, edges)
    render_preview(os.path.join(out_dir, "preview.png"), names, pts, edges, title)
    render_interactive(os.path.join(out_dir, "interactive.html"), names, pts, edges, title)

    print(f"  exported model.obj / model.ply / preview.png -> {out_dir}")


def main():
    for version in ["full_body", "waist_down"]:
        _, connections = landmarks_for_version(version)
        for pose_name in ["pose1", "pose2", "pose3"]:
            print(f"\n[{version}/{pose_name}]")
            process(version, pose_name, connections)

    print("\nDone. Open the preview.png files for a quick look, or interactive.html "
          "in a browser to rotate the model, or model.obj in Blender/MeshLab for the "
          "full experience.")


if __name__ == "__main__":
    main()
