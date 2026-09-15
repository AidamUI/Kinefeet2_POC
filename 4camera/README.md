# 4-Camera Setup

Self-contained setup for 4 synchronized cameras at 0, 90, 180 and 270
degrees.

## Structure

```
4camera/
├── scripts/
│   ├── run_pipeline_calibrated.py     # requires a real calibration
│   ├── run_pipeline_uncalibrated.py   # approximate preview, no calibration needed
│   ├── _pipeline_common.py            # shared plumbing, not run directly
│   └── config_4camera.yaml
├── data/                 # Input photos (4 per pose)
│   ├── full_body/
│   │   ├── pose1/
│   │   ├── pose2/
│   │   └── pose3/
│   └── waist_down/
│       ├── pose1/
│       ├── pose2/
│       └── pose3/
├── output/               # Calibrated pipeline results
└── output_uncalibrated/  # Uncalibrated pipeline results (kept separate)
```

## Quick start

1. Place 4 photos in `data/full_body/pose1/`:
   - `01.jpg` = 0 degrees (front)
   - `02.jpg` = 90 degrees (right)
   - `03.jpg` = 180 degrees (back)
   - `04.jpg` = 270 degrees (left)

2. Calibrate the rig once. This is what makes the 3D output metric instead
   of a guess. See [`../camera_calibration/README.md`](../camera_calibration/README.md):

   ```bash
   cd ../camera_calibration
   python calibrate_intrinsics.py
   python calibrate_extrinsics.py
   python export_cameras.py
   ```

3. Run the pipeline:

   ```bash
   cd scripts
   python run_pipeline_calibrated.py
   ```

4. Check results in `output/`.

## Two pipelines, one difference

Both scripts run the exact same steps in the exact same order: extract 2D
keypoints, triangulate 3D, export and visualize, generate the mesh, verify by
reprojection. The only difference is where the camera positions come from.

`run_pipeline_calibrated.py` is the one to use for a real 3D reconstruction.
It requires `camera_calibration/` to have already written a real
`cameras.json` for every version you have photos for, and refuses to run
otherwise rather than silently falling back to a guess. It triangulates from
the 4 views using camera positions and lens distortion that were actually
measured. Output goes to `output/`.

`run_pipeline_uncalibrated.py` runs the identical pipeline, but with one
extra step first (`03_setup_camera_rig.py`) that builds an approximate
camera ring from the radius, height and field-of-view numbers in
`config_4camera.yaml`, instead of reading measured values from
`camera_calibration/`. Use this for a quick preview before you have
calibrated, or when you cannot calibrate at all (no board, one-off shoot).
It assumes the cameras sit on a perfect circle at exactly the typed-in
radius and height, aimed exactly at the target point, with a guessed field
of view and no lens distortion. This is still real triangulation, just from
a guess instead of a measurement, so it is noticeably less accurate. Output
goes to `output_uncalibrated/`, which stays permanently separate so this can
never overwrite a real calibrated result.

## Notes

- Both pipelines borrow the parent `scripts/` folder for the actual
  processing steps (MediaPipe extraction, triangulation, mesh generation),
  temporarily pointing its `data/`, `output/` and `config.yaml` at
  4camera's own copies and restoring them afterward.
- Data and output are isolated from the main project and from `8camera/`.
