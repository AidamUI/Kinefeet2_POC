"""
run_pipeline_uncalibrated.py
------------------------------
Runs the exact same 4-camera pipeline as run_pipeline_calibrated.py, without
requiring a real calibration first. The one difference is the extra
03_setup_camera_rig.py step: instead of reading real measured camera
positions from cameras.json (written by camera_calibration/export_cameras.py),
it builds an approximate camera ring from the radius/height/field-of-view
numbers in config_4camera.yaml.

    cd 4camera/scripts
    python run_pipeline_uncalibrated.py

Use this for a quick preview before you've calibrated, or when you genuinely
can't (no calibration board, one-off shoot). The approximate ring assumes the
cameras sit on a perfect circle at exactly the typed-in radius/height, aimed
exactly at the target point, with a guessed field of view and no lens
distortion - real, but only as accurate as those numbers. For an actual
measured reconstruction, calibrate the rig and use run_pipeline_calibrated.py
instead; see camera_calibration/README.md.

Output goes to 4camera/output_uncalibrated/, never to 4camera/output/, so this
can never overwrite a real calibrated result.

Steps run, per version (full_body / waist_down) that has photos:
    1. 02_extract_2d_keypoints.py   MediaPipe 2D keypoints from the 4 photos
    2. 03_setup_camera_rig.py       build the approximate ring from config_4camera.yaml
    3. 04_triangulate_3d.py         DLT triangulation using that ring
    4. 05_export_and_visualize.py   model.obj/.ply, interactive.html
    5. 07_generate_mediapipe_mesh.py body mesh
    6. 08_verify_reprojection.py    reprojects the 3D result back into each
                                     photo - here this measures agreement with
                                     the assumed ring, not a real calibration
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pipeline_common as common

OUTPUT_PATH = os.path.join(
    common.FOUR_CAM_ROOT, os.environ.get("KINEFEET_OUTPUT", "output_uncalibrated")
)

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
    print("4-CAMERA PIPELINE - UNCALIBRATED (approximate rig)")
    print("=" * 70)
    print(
        "\n[!] Camera positions are ASSUMED from config_4camera.yaml, not measured.\n"
        "    Every distance in the 3D output is only as accurate as those\n"
        "    typed-in radius/height numbers. For real accuracy, calibrate first\n"
        "    and use run_pipeline_calibrated.py instead - see\n"
        "    camera_calibration/README.md."
    )
    common.print_photo_naming()

    versions = common.versions_with_data()
    if not versions:
        print(f"\n[!] No photos found under {common.DATA_4CAM_PATH}")
        print("    Add photos to data/full_body/pose1|2|3/ (and/or waist_down/) first.")
        sys.exit(1)

    calibrated = [
        v for v in versions
        if common.is_calibrated(
            os.path.join(common.FOUR_CAM_ROOT, "output", v, "cameras.json")
        )
    ]
    if calibrated:
        print(
            f"\n[i] {', '.join(calibrated)} already {'has' if len(calibrated) == 1 else 'have'} "
            "a real calibration in 4camera/output/.\n"
            "    Prefer run_pipeline_calibrated.py for those - this script's output "
            "still\n    goes to output_uncalibrated/ and will not touch it."
        )

    print(f"\nVersions with photos: {', '.join(versions)}")
    print(f"Output will go to: {OUTPUT_PATH}\n")

    with common.borrowed_project_paths(OUTPUT_PATH):
        common.run_steps(STEPS)

    common.print_results_summary(OUTPUT_PATH)


if __name__ == "__main__":
    main()
