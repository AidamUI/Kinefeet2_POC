"""
run_pipeline_8camera.py
-----------------------
Runs the complete 8-camera pipeline: 8 photos at 0/45/90/135/180/225/270/315
degrees, through the same numbered scripts 4camera/ uses.

    cd 8camera/scripts
    python run_pipeline_8camera.py

Note: uses config_8camera.yaml for configuration.

Camera positions are ASSUMED from config_8camera.yaml (03_setup_camera_rig.py
builds an approximate ring from typed-in radius/height/field-of-view numbers),
not measured - there is no calibrated option for 8camera the way there is for
4camera (see camera_calibration/README.md and 4camera/README.md). Every
distance in the 3D output is only as accurate as those numbers.

Steps run, per version (full_body / waist_down) that has photos:
    1. 02_extract_2d_keypoints.py   MediaPipe 2D keypoints from the 8 photos
    2. 03_setup_camera_rig.py       build the approximate ring from config_8camera.yaml
    3. 04_triangulate_3d.py         DLT triangulation using that ring
    4. 05_export_and_visualize.py   model.obj/.ply, interactive.html
    5. 07_generate_mediapipe_mesh.py body mesh
    6. 08_verify_reprojection.py    reprojection accuracy check

Output: 8camera/output/<version>/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pipeline_common as common

STEPS = [
    "02_extract_2d_keypoints.py",
    "03_setup_camera_rig.py",
    "04_triangulate_3d.py",
    "05_export_and_visualize.py",
    "07_generate_mediapipe_mesh.py",
    "08_verify_reprojection.py",
]


def main():
    print("=" * 70)
    print("8-CAMERA PIPELINE FOR KINEFEET 2.0")
    print("=" * 70)
    print(
        "\n[!] Camera positions are ASSUMED from config_8camera.yaml, not measured.\n"
        "    Every distance in the 3D output is only as accurate as those\n"
        "    typed-in radius/height numbers."
    )
    common.print_photo_naming()

    versions = common.versions_with_data()
    if not versions:
        print(f"\n[!] No photos found under {common.DATA_8CAM_PATH}")
        print("    Add photos to data/full_body/pose1|2|3/ (and/or waist_down/) first.")
        sys.exit(1)

    output_path = os.path.join(common.EIGHT_CAM_ROOT, "output")
    print(f"\nVersions with photos: {', '.join(versions)}")
    print(f"Output will go to: {output_path}\n")

    with common.borrowed_project_paths(output_path):
        common.run_steps(STEPS)

    common.print_results_summary(output_path)


if __name__ == "__main__":
    main()
