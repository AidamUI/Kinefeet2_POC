"""
_pipeline_common.py
--------------------
Shared plumbing for the two 4-camera pipeline entry points
(run_pipeline_calibrated.py and run_pipeline_uncalibrated.py).

The numbered scripts in ../../scripts/ (02_extract_2d_keypoints.py etc.) are
written against fixed relative paths - ../data and ../output next to the
script, and ../config.yaml. That is convenient for the single-rig project
layout they were designed for, but 4camera/ keeps its own data, output and
config alongside those of 8camera/ and the original 8-shot rig. The two
pipelines here bridge that gap the same way: temporarily swap the project's
data/, output/ and config.yaml for 4camera's own copies (via a directory
junction on Windows, a symlink elsewhere), run the numbered scripts unchanged,
then swap back - restored even if a step raises.

Not module-level: this file is imported from CWD-relative sibling scripts, so
it protects that with sys.path itself rather than relying on the caller.
"""

from __future__ import annotations

import contextlib
import os
import runpy
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOUR_CAM_ROOT = os.path.dirname(SCRIPT_DIR)          # 4camera/
PROJECT_ROOT = os.path.dirname(FOUR_CAM_ROOT)          # Kinefeet2_POC/
PARENT_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")

CONFIG_4CAM_PATH = os.path.join(SCRIPT_DIR, "config_4camera.yaml")
PROJECT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
PROJECT_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
PROJECT_OUTPUT_PATH_TEMPLATE = os.path.join(PROJECT_ROOT, "output")

DATA_4CAM_PATH = os.path.join(FOUR_CAM_ROOT, "data")

# Must match calibrate_extrinsics.py's stamp on cameras.json (SOURCE_MARKER in
# camera_calibration/export_cameras.py) and the same check in
# scripts/03_setup_camera_rig.py. Duplicated rather than imported because
# camera_calibration/ and 4camera/scripts/ are independent script sets with no
# shared package - see camera_calibration/README.md for why.
CALIBRATED_SOURCE_MARKER = "camera_calibration/export_cameras.py"

if PARENT_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, PARENT_SCRIPT_DIR)


def is_calibrated(cameras_json_path):
    """True if cameras.json at this path came from a real calibration."""
    if not os.path.exists(cameras_json_path):
        return False
    try:
        import json

        with open(cameras_json_path) as f:
            cameras = json.load(f)
        return any(
            isinstance(cam, dict) and cam.get("source") == CALIBRATED_SOURCE_MARKER
            for cam in cameras.values()
        )
    except (ValueError, OSError, AttributeError):
        return False


def versions_with_data():
    """Which of full_body / waist_down have any capture photos under 4camera/data/."""
    found = []
    for version in ("full_body", "waist_down"):
        version_dir = os.path.join(DATA_4CAM_PATH, version)
        if not os.path.isdir(version_dir):
            continue
        has_photos = False
        for pose in os.listdir(version_dir):
            pose_dir = os.path.join(version_dir, pose)
            if not os.path.isdir(pose_dir):
                continue
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(pose_dir)):
                has_photos = True
                break
        if has_photos:
            found.append(version)
    return found


# Dropped inside a borrowed data/output directory (link or fallback copy) so a
# later run can recognise leftover borrowed state that never had a
# ``.pipeline_backup`` to begin with - typically PROJECT_OUTPUT_PATH_TEMPLATE,
# which usually doesn't exist before a run, so there is nothing to back up and
# nothing for ``_heal_stale_backup`` to find, yet a hard kill can still leave
# it sitting there needing to be told apart from a real directory.
_BORROWED_MARKER_NAME = ".kinefeet_borrowed_by_4camera_pipeline"


def _link_or_copy(source, dest):
    """Junction/symlink dest -> source; fall back to a plain copy if that fails."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["mklink", "/J", dest, source], shell=True, check=True, capture_output=True
            )
        else:
            os.symlink(source, dest)
        outcome = "linked"
    except Exception as exc:
        print(f"    [!] could not link {dest} -> {source} ({exc}); copying instead")
        if os.path.isdir(source):
            shutil.copytree(source, dest)
        else:
            os.makedirs(dest, exist_ok=True)
        outcome = "copied"

    if os.path.isdir(dest):
        # A junction transparently exposes the target's own contents, so this
        # marker also silently appears (and is used) inside the *source*
        # directory forever after - harmless (nothing reads it there) but a
        # stray file the source folder didn't ask for, hence only for real
        # directories, not the common junction case.
        if outcome == "copied":
            open(os.path.join(dest, _BORROWED_MARKER_NAME), "w").close()
        else:
            open(dest + _BORROWED_MARKER_NAME, "w").close()  # sibling, not inside
    return outcome


def _is_orphaned_borrow(path):
    """A borrowed link/copy from a run that died before restoring it, and that
    (unlike config.yaml or data/) had nothing backed up in the first place -
    see ``_BORROWED_MARKER_NAME``."""
    return os.path.exists(path + _BORROWED_MARKER_NAME) or (
        os.path.isdir(path) and os.path.exists(os.path.join(path, _BORROWED_MARKER_NAME))
    )


def _unlink_or_remove(path):
    """Remove whatever is at ``path`` - link, directory, or plain file."""
    marker = path + _BORROWED_MARKER_NAME
    if os.path.exists(marker):
        os.remove(marker)

    if not os.path.exists(path) and not os.path.islink(path):
        return
    if os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        if os.name == "nt":
            # A junction and a real directory both look like a dir to
            # os.path.isdir; rmdir removes an empty junction without
            # touching what it points to, but recurses into a real copy.
            subprocess.run(["rmdir", path], shell=True, check=False)
            if os.path.exists(path):
                shutil.rmtree(path)
        else:
            shutil.rmtree(path)
    else:
        os.remove(path)


def _backup_path(path):
    return path + ".pipeline_backup"


def _restore_backups(backups):
    """Put every backed-up path back, unconditionally overwriting whatever
    (borrowed link, borrowed copy, or nothing) currently occupies it."""
    for path, backup in backups.items():
        if not os.path.exists(backup):
            continue
        _unlink_or_remove(path)
        shutil.move(backup, path)


def _heal_stale_backup(path):
    """Recover from a previous run that never reached its own restore step.

    A hard kill (a piped reader like ``| head`` closing early can terminate
    the writer outright on Windows, skipping Python's own exception handling
    and therefore every ``finally``) can leave things exactly as they were
    mid-borrow: the real data/output/config.yaml sitting in
    ``<path>.pipeline_backup`` and a borrowed link or copy still live at
    ``path``. Nothing after that point can tell the two apart from a normal
    first run, so check for it before ever taking a fresh backup - otherwise
    the fresh backup collides with the stale one (a real crash this project
    hit) or, worse, silently backs up the *borrowed* copy, burying the real
    data/config.yaml under it for good.
    """
    backup = _backup_path(path)
    if not os.path.exists(backup):
        return
    print(f"    [!] found {os.path.basename(backup)} left by a run that didn't "
          f"finish - restoring it before continuing")
    _restore_backups({path: backup})


@contextlib.contextmanager
def borrowed_project_paths(output_4cam_path):
    """Point the project's config.yaml / data/ / output/ at 4camera's own copies.

    ``output_4cam_path`` is where results should actually land (either
    4camera/output for the calibrated pipeline, or 4camera/output_uncalibrated
    for the approximate one) - the two must never share a folder, or an
    uncalibrated run could clobber a real calibration's results.

    Crash-safety covers both directions: a step raising inside the ``with``
    block restores everything via the ``finally`` below as before, and in
    case borrowing itself is interrupted (the process killed outright, not a
    normal Python exception - see ``_heal_stale_backup``) whatever backups
    this attempt already took are rolled back before the error propagates,
    and a still-stale one from an *earlier* interrupted run is healed before
    a new attempt even starts.
    """
    paths = (PROJECT_CONFIG_PATH, PROJECT_DATA_PATH, PROJECT_OUTPUT_PATH_TEMPLATE)
    for path in paths:
        _heal_stale_backup(path)
        if os.path.exists(path) and _is_orphaned_borrow(path):
            print(f"    [!] {os.path.basename(path)} is leftover borrowed state "
                  f"from a run that didn't finish (and had nothing to restore "
                  f"it to) - removing it before continuing")
            _unlink_or_remove(path)

    backups = {}
    try:
        for path in paths:
            if os.path.exists(path):
                backup = _backup_path(path)
                if os.path.isfile(path):
                    shutil.copy2(path, backup)
                else:
                    shutil.move(path, backup)
                backups[path] = backup

        shutil.copy2(CONFIG_4CAM_PATH, PROJECT_CONFIG_PATH)
        os.makedirs(output_4cam_path, exist_ok=True)
        _link_or_copy(DATA_4CAM_PATH, PROJECT_DATA_PATH)
        _link_or_copy(output_4cam_path, PROJECT_OUTPUT_PATH_TEMPLATE)
    except BaseException:
        print("    [!] setup failed partway through - rolling back")
        _restore_backups(backups)
        for path in paths:
            if path not in backups:
                _unlink_or_remove(path)  # borrowed, but nothing to restore it to
        raise

    try:
        yield
    finally:
        _restore_backups(backups)
        for path in paths:
            if path not in backups:
                _unlink_or_remove(path)  # borrowed, but nothing to restore it to
        print("    restored project data/, output/ and config.yaml")


def run_steps(steps):
    """Run each numbered pipeline script in ../../scripts/, in order."""
    for step in steps:
        path = os.path.join(PARENT_SCRIPT_DIR, step)
        print("\n" + "=" * 70)
        print(f"RUNNING {step}")
        print("=" * 70)
        runpy.run_path(path, run_name="__main__")


def print_photo_naming():
    print("Expected photo naming (per pose folder):")
    print("  01.jpg = 0 deg (front)")
    print("  02.jpg = 90 deg (right side)")
    print("  03.jpg = 180 deg (back)")
    print("  04.jpg = 270 deg (left side)")


def print_results_summary(output_path):
    print("\n" + "=" * 70)
    print("4-CAMERA PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to {output_path}")
    print("Check the following for each pose:")
    print("  - model.obj / model.ply (3D skeleton)")
    print("  - *_mediapipe_mesh.obj (body mesh)")
    print("  - interactive.html (rotatable viewer)")
    print("  - reprojection_verification/ (accuracy check)")
    print("=" * 70)
