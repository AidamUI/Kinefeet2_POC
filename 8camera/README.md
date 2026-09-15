# 8-Camera Setup

Self-contained setup for 8 cameras at 0, 45, 90, 135, 180, 225, 270 and 315
degrees.

## Structure

```
8camera/
├── scripts/
│   ├── run_pipeline_8camera.py
│   ├── _pipeline_common.py   # shared plumbing, not run directly
│   └── config_8camera.yaml
├── data/                 # Input photos (8 per pose)
│   ├── full_body/
│   │   ├── pose1/
│   │   ├── pose2/
│   │   └── pose3/
│   └── waist_down/
│       ├── pose1/
│       ├── pose2/
│       └── pose3/
└── output/               # Generated 3D models
```

## Quick Start

1. Place 8 photos in `data/full_body/pose1/`:
   - `01.jpg` = 0 degrees (front)
   - `02.jpg` = 45 degrees
   - `03.jpg` = 90 degrees (right)
   - `04.jpg` = 135 degrees
   - `05.jpg` = 180 degrees (back)
   - `06.jpg` = 225 degrees
   - `07.jpg` = 270 degrees (left)
   - `08.jpg` = 315 degrees

2. Edit `scripts/config_8camera.yaml` with your measurements.

3. Run the pipeline:
   ```bash
   cd scripts
   python run_pipeline_8camera.py
   ```

4. Check results in `output/`.

## Camera positions are assumed, not measured

Unlike `4camera/` (see [`../4camera/README.md`](../4camera/README.md) and
[`../camera_calibration/README.md`](../camera_calibration/README.md)), there
is no calibrated option here yet, because `camera_calibration/` does not
target an 8-camera rig. `03_setup_camera_rig.py` always builds an
approximate ring from the radius, height and field-of-view numbers in
`config_8camera.yaml`. The cameras are assumed to sit on a perfect circle at
exactly that radius and height, aimed exactly at the target point, with a
guessed field of view and no lens distortion. Every distance in the 3D
output is only as accurate as those typed-in numbers.

## Notes

- Uses the parent `scripts/` folder for processing, temporarily pointing its
  `data/`, `output/` and `config.yaml` at 8camera's own copies (through a
  directory junction) and restoring them afterward. This happens even if a
  step fails or the process is interrupted; a later run detects and repairs
  an interrupted swap automatically.
- Data and output are isolated from the main project and from `4camera/`.
