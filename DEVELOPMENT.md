# Development guide

For anyone extending Kinefeet 2.0. Users should start with `README.md` and
`GLOBAL_RECONSTRUCTION.md`.

## Pipeline at a glance

```
photos -> 02 2D keypoints -> 03 camera rig (assumed) | camera_calibration/ (measured)
       -> 04 triangulate -> 05 skeleton export -> 07 body mesh -> 08 reprojection check
       -> 09 sequence -> 10 SMPL-X fit -> 11 OpenSim IK -> 12 Rerun viewer
```

Scripts `02`-`08` are the original per-pose pipeline and read `config.yaml`,
`data/` and `output/` through fixed relative paths. `4camera/` and `8camera/`
wrap them by temporarily pointing those paths at the rig's own folders.
Scripts `09`-`12` are newer: they take the rig output folder as an argument and
do not use the path swapping, so they also work on the root `output/`.

Run everything for a rig with its wrapper (it runs the global stage at the end,
`--no-global` skips it), or the global stage alone with
`python scripts/run_global.py <rig>/output [--version waist_down]`.

## Data contracts

**`<rig>/<version>/<pose>/keypoints_2d/NN.json`** (script 02): `image`,
`image_width`, `image_height`, `landmarks[33]` with `x_px`, `y_px`, `visibility`.
`NN` is the camera number, 1-based: `01` is `cam_00`. Cameras are matched by
this number, never by list position (`utils.camera_key_for_image`).

**`<rig>/<version>/cameras.json`**: `cam_00...` with `K`, `R`, `t`, `P`,
`dist_coeffs`, `image_width`, `image_height`, `source`. World frame: metres,
Z up, floor at Z = 0 when `source` is `camera_calibration/export_cameras.py`.

**`<rig>/<version>/sequence/joints_world.npz`** (script 09), the hand-over
format for everything after triangulation:

| key | shape | meaning |
|---|---|---|
| `joints` | (F, J, 3) | metres, world frame, NaN where not triangulated |
| `names`, `frames` | (J,), (F,) | joint names, frame labels (pose names today) |
| `keypoints2d` | (F, C, J, 3) | detected (u, v, visibility), calibration-resolution pixels, undistorted |
| `cam_keys`, `K`, `R`, `t`, `dist`, `image_size` | C cameras | calibration data |
| `calibrated` | bool | false means the world frame is an assumed ring |

`F` is the number of frames. Poses are frames today; video frames can use the
same file with no change to scripts 10-12.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q
```

`tests/` covers camera matching, undistortion, DLT, frame transforms, the
sequence builder, the OpenSim TRC writer, and the SMPL-X fitter. The fitter
tests use `tests/fake_smplx.py`, a forward-kinematics stand-in, so they run
without the licensed weights (they are skipped if `torch` is missing).
`camera_calibration/tests/` is separate. CI runs both (`.github/workflows/tests.yml`).

## Adding video (the main next step)

1. **Capture:** synchronized video from static, calibrated cameras. Put a clap or
   flash in view of every camera so frames can be aligned.
2. **Frames:** write a script that decodes each camera's video to
   `data/<version>/<clip>/<camera>/frame_00000.jpg` at a common frame rate after
   applying the sync offset.
3. **2D detection:** run MediaPipe in VIDEO mode (tracking between frames) and
   save keypoints per frame and camera, in the layout above.
4. **Triangulate per frame:** call `triangulate_pose` from script 04 for every
   frame; build `joints_world.npz` with `F = number of frames` and a real frame
   rate (script 11 hardcodes 1 frame per second, set `rate` in `write_trc`).
5. **Smoothing and gaps:** low-pass filter (Butterworth, 6 Hz is common for
   gait) and interpolate short gaps before OpenSim. Not implemented.
6. **OpenSim:** `11_export_opensim.py` already runs scaling and inverse
   kinematics; use a standing frame as frame 0 for scaling.

## Known issues and roadmap

- The SMPL-X fit initialises every frame independently. For video, add a
  temporal smoothness term and initialise each frame from the previous one.
- MediaPipe has 3 points per foot, so subtalar and toe angles are weak. A denser
  foot model or foot-specific detector is needed for foot-focused measures.
- Absolute scale depends on calibration quality; validate against a known
  length before relying on millimetres.
- Calibration warns about focal-length consistency for some cameras (see
  `camera_calibration/output/`); a dedicated recalibration would help.
- 8-camera and waist-down data use an assumed ring and have no real calibration;
  their errors are large and their world frame is not tied to the floor.
- Script 07 selects the last sorted `joints_3d.json` in its CLI, and script 08
  uses a visibility threshold of 0.5 while triangulation uses 0.1.
- OpenSim `.trc` files assume 1 s between poses; set the real rate for video.
