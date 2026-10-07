# Global reconstruction (generate_mesh branch)

Takes the per-pose 3D skeletons from the main pipeline, places them in one
calibrated world frame, checks they are self-consistent, and produces a
viewer, OpenSim joint angles and (optionally) SMPL-X body meshes.

Today's data is a set of separate photo poses, not video. Each pose is one
"frame" with no real timing; the same files hold real video frames later.

## Run it

```bash
# the pipeline wrappers run the global stage at the end (--no-global skips it)
cd 4camera/scripts && python run_pipeline_calibrated.py && cd ../..

# or run the global stage alone on any finished output folder
python scripts/run_global.py 4camera/output            # also: 4camera/output_uncalibrated, 8camera/output
python scripts/run_global.py 8camera/output --version waist_down
python -m rerun 4camera/output/full_body/sequence/kinefeet.rrd
```

Install: `pip install rerun-sdk opensim` (viewer and joint angles). SMPL-X also
needs `smplx torch` and the licensed model files (below).

## Stages

| Script | Output (in `<rig>/<version>/sequence/`) |
|---|---|
| `09_build_sequence.py` | `joints_world.npz` (frames x joints x 3, metres, Z up), `consistency.json` |
| `10_fit_smplx_sequence.py` | `smplx_world.npz`, `smplx/<pose>.obj`, `smplx_fit.json` (needs weights) |
| `11_export_opensim.py` | `opensim/markers.trc`, `model_scaled.osim`, `ik.mot`, `joint_angles.csv`, `ik_report.json` |
| `12_view_global.py` | `kinefeet.rrd` for the Rerun viewer |

`scripts/frames.py` holds the coordinate-frame maths (world, camera i, pelvis,
Y-up) with unit tests.

## Viewer

Left panel (Rerun blueprint panel): tick layers on or off (skeleton, SMPL-X
mesh, cameras, floor, pelvis path). Tabs choose the reference frame:
**Global (world)**, **Relative to cam_00 ... cam_NN** (motion with respect to
that camera) and **Relative to pelvis**. Right side: each camera's photo with
detected landmarks (green) and the 3D result projected back (red). The timeline
at the bottom steps through the poses.

## Checks the sequence stage reports

`consistency.json` compares the same person across poses: bone-length spread,
left/right difference, and the lowest foot joint height. On a calibrated rig the
feet should sit near Z = 0. On an assumed-ring rig Z is not tied to the floor
and the file says so.

## Checking the results

- Quickest: `sequence/smplx_preview.png` (all poses, joints in red) and
  `sequence/smplx_interactive.html` (rotate in a browser, click a pose in the legend
  to hide it).
- Mesh files: `sequence/smplx/pose1.obj` / `.ply` ... open in Blender or MeshLab
  (Z up, metres, all poses share one world frame).
- Everything together: `python -m rerun <rig>/<version>/sequence/kinefeet.rrd`.
- Numbers: `smplx_fit.json` (joint error cm, reprojection px), `consistency.json`,
  `opensim/ik_report.json` (marker error cm, joint angles).

## Needed from you

1. **SMPL-X weights** (already set up on the original machine): register at
   https://smpl-x.is.tue.mpg.de/, download "SMPL-X v1.1", place
   `SMPLX_NEUTRAL.npz` (and optionally MALE/FEMALE) in `smpl_models/smplx/`.
   Git ignores them; the licence forbids redistributing them.
2. **Video** for real motion: synchronized video from static, calibrated cameras
   (a clap or flash visible to all cameras for syncing). Smoothing, gap filling
   and gait curves need this and are not implemented yet.

## Known limits

- MediaPipe gives 3 points per foot, so ankle flexion is usable but subtalar
  (inversion/eversion) and toe angles are weakly constrained.
- Scale is only as good as the calibration. The calibrated 4-camera run gives
  a thigh of about 31 cm and a fitted SMPL-X body 1.5 m tall; that is plausible
  for a petite subject, but compare with the subject's real height and one
  measured limb before trusting absolute sizes.
- The SMPL-X fit initialises each frame on its own; for video add a temporal term.
- `opensim_setup/` is from Pose2Sim (BSD-3-Clause), see `LICENSE_Pose2Sim`.
