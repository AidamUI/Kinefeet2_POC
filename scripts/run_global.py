"""
run_global.py
-------------
Runs the global-reconstruction stages on a rig's finished output folder:

    09 build sequence  ->  (10 SMPL-X, if the model weights are present)
                       ->  11 OpenSim export (+ scaling and IK if installed)
                       ->  12 Rerun viewer file

Run the normal pipeline first (e.g. 4camera/scripts/run_pipeline_calibrated.py)
so that <rig_output_dir>/<version>/<pose>/joints_3d.json exists.

USAGE   python scripts/run_global.py 4camera/output [--version full_body] [--open]
                                     [--smplx-gender male|female|neutral]
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    print("\n" + "=" * 70 + f"\n{script}\n" + "=" * 70, flush=True)
    return subprocess.run(cmd, cwd=PROJECT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rig_output_dir")
    ap.add_argument("--version", default="full_body", choices=["full_body", "waist_down"])
    ap.add_argument("--open", action="store_true", help="open the Rerun viewer when done")
    ap.add_argument("--smplx-gender", default="male", choices=["male", "female", "neutral"],
                    help="SMPL-X template (default male); changes only the body surface")
    args = ap.parse_args()
    rig, ver = args.rig_output_dir, ["--version", args.version]

    if run("09_build_sequence.py", rig, *ver):
        sys.exit("sequence build failed - is joints_3d.json present for each pose?")

    weights = os.path.join(PROJECT, "smpl_models", "smplx", f"SMPLX_{args.smplx_gender.upper()}.npz")
    if os.path.exists(weights):
        run("10_fit_smplx_sequence.py", rig, *ver, "--gender", args.smplx_gender)
    else:
        print("\n(skipping 10_fit_smplx_sequence.py: SMPL-X weights not found at "
              f"{os.path.relpath(weights, PROJECT)} - see GLOBAL_RECONSTRUCTION.md)")

    run("11_export_opensim.py", rig, *ver)
    run("12_view_global.py", rig, *ver, *(["--open"] if args.open else []))


if __name__ == "__main__":
    main()
