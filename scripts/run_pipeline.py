"""
run_pipeline.py
-----------------
Convenience script that runs steps 02-05 in order (step 01, checkerboard
calibration, is optional and manual - see README).

Usage:
    python run_pipeline.py

Equivalent to running these one at a time:
    python 02_extract_2d_keypoints.py
    python 03_setup_camera_rig.py
    python 04_triangulate_3d.py
    python 05_export_and_visualize.py
"""

import runpy
import os

SCRIPT_DIR = os.path.dirname(__file__)
STEPS = [
    "02_extract_2d_keypoints.py",
    "03_setup_camera_rig.py",
    "04_triangulate_3d.py",
    "05_export_and_visualize.py",
]


def main():
    for step in STEPS:
        path = os.path.join(SCRIPT_DIR, step)
        print("\n" + "=" * 70)
        print(f"RUNNING {step}")
        print("=" * 70)
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
