"""
run_pipeline.py
-----------------
Convenience script that runs steps 02, 03, 04, 05 and 08 in order.

Camera calibration is a separate, one-time step: see camera_calibration/README.md.
03_setup_camera_rig.py only builds an approximate camera ring from
config.yaml measurements, and skips itself automatically once
camera_calibration/ has produced a real cameras.json.

Usage:
    python run_pipeline.py [--no-global]

After the steps, scripts/run_global.py (world-frame sequence, SMPL-X mesh, OpenSim
joint angles, Rerun viewer) runs on output/ for each version with results; pass
--no-global to skip it. See GLOBAL_RECONSTRUCTION.md.

Equivalent to running these one at a time:
    python 02_extract_2d_keypoints.py
    python 03_setup_camera_rig.py
    python 04_triangulate_3d.py
    python 05_export_and_visualize.py
    python 08_verify_reprojection.py
"""

import runpy
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(__file__)
STEPS = [
    "02_extract_2d_keypoints.py",
    "03_setup_camera_rig.py",
    "04_triangulate_3d.py",
    "05_export_and_visualize.py",
    "08_verify_reprojection.py",
]


def main():
    for step in STEPS:
        path = os.path.join(SCRIPT_DIR, step)
        print("\n" + "=" * 70)
        print(f"RUNNING {step}")
        print("=" * 70)
        runpy.run_path(path, run_name="__main__")

    if "--no-global" in sys.argv:
        return
    project = os.path.dirname(SCRIPT_DIR)
    output = os.path.join(project, "output")
    for version in ("full_body", "waist_down"):
        if os.path.exists(os.path.join(output, version, "pose1", "joints_3d.json")):
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "run_global.py"), output,
                            "--version", version], cwd=project)


if __name__ == "__main__":
    main()
