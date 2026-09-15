"""
run_pipeline_calibrated.py
---------------------------
Runs the 4-camera pipeline using a REAL camera calibration - the intended way
to run this project.

    cd 4camera/scripts
    python run_pipeline_calibrated.py

Prerequisite: camera_calibration/ has already produced 4camera/output/<version>/
cameras.json for every version you have photos for (see
camera_calibration/README.md):

    cd camera_calibration
    python calibrate_intrinsics.py
    python calibrate_extrinsics.py
    python export_cameras.py

This script does NOT run 03_setup_camera_rig.py - there is nothing for it to
do. cameras.json already holds the real measured camera positions, matched
to the actual capture frame size, with real lens distortion. Building a
synthetic ring on top of that would be a downgrade, not a setup step (see
run_pipeline_uncalibrated.py if that is genuinely what you want).

Steps run, per version (full_body / waist_down) that has photos:
    1. 02_extract_2d_keypoints.py   MediaPipe 2D keypoints from the 4 photos
    2. 04_triangulate_3d.py         DLT triangulation using cameras.json
    3. 05_export_and_visualize.py   model.obj/.ply, interactive.html
    4. 07_generate_mediapipe_mesh.py body mesh
    5. 08_verify_reprojection.py    reprojects the 3D result back into each
                                     photo, the real accuracy check

Output: 4camera/output/<version>/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pipeline_common as common

STEPS = [
    "02_extract_2d_keypoints.py",
    "04_triangulate_3d.py",
    "05_export_and_visualize.py",
    "07_generate_mediapipe_mesh.py",
    "08_verify_reprojection.py",
]


def check_calibration(versions):
    """Refuse to run rather than silently triangulate from a missing/fake rig."""
    missing = []
    for version in versions:
        cameras_json = os.path.join(common.FOUR_CAM_ROOT, "output", version, "cameras.json")
        if not common.is_calibrated(cameras_json):
            missing.append((version, cameras_json))
    return missing


def main():
    print("=" * 70)
    print("4-CAMERA PIPELINE - CALIBRATED")
    print("=" * 70)
    common.print_photo_naming()

    versions = common.versions_with_data()
    if not versions:
        print(f"\n[!] No photos found under {common.DATA_4CAM_PATH}")
        print("    Add photos to data/full_body/pose1|2|3/ (and/or waist_down/) first.")
        sys.exit(1)

    missing = check_calibration(versions)
    if missing:
        print("\n" + "!" * 70)
        print("NOT CALIBRATED")
        print("!" * 70)
        for version, path in missing:
            print(f"  {version}: {path}")
            print(f"    missing, or not produced by camera_calibration/export_cameras.py")
        print(
            "\n  This pipeline requires a real calibration. Run it once:\n\n"
            "    cd ../../camera_calibration\n"
            "    python calibrate_intrinsics.py\n"
            "    python calibrate_extrinsics.py\n"
            "    python export_cameras.py\n\n"
            "  See camera_calibration/README.md. For a quick, approximate preview\n"
            "  without calibrating first, use run_pipeline_uncalibrated.py instead -\n"
            "  its output is clearly kept separate and is not a substitute for this."
        )
        print("!" * 70)
        sys.exit(1)

    print(f"\nVersions with photos and a real calibration: {', '.join(versions)}")
    print(f"Output will go to: {os.path.join(common.FOUR_CAM_ROOT, 'output')}\n")

    with common.borrowed_project_paths(os.path.join(common.FOUR_CAM_ROOT, "output")):
        common.run_steps(STEPS)

    common.print_results_summary(os.path.join(common.FOUR_CAM_ROOT, "output"))


if __name__ == "__main__":
    main()
