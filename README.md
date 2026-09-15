# Kinefeet 2.0 - Multi-View 3D Foot Reconstruction

Proof of concept: diabetic foot assessment using computer vision and 3D
reconstruction.

## Project Background

Kinefeet is a diabetic foot assessment tool that helps healthcare providers
monitor foot deformities and complications in diabetic patients. Traditional
Kinefeet uses manual measurements and 2D photography, which can be
time-consuming and subjective.

This proof of concept modernizes diabetic foot assessment using multi-view
3D reconstruction. By capturing photos from multiple angles and applying
computer vision, it builds accurate 3D models of a patient's feet and lower
body, enabling:

- Objective measurements: precise 3D distances, angles and volumes
- Progress tracking: comparing 3D models over time to detect changes
- Remote assessment: 3D models can be reviewed without the patient present
- Clinical documentation: permanent 3D records for medical files

## Overview

This system converts ordinary photos taken from multiple angles into 3D
skeletal reconstructions and body meshes. It uses geometric triangulation,
the same principle stereo vision uses to perceive depth, rather than
AI-based depth estimation.

Clinical applications:
- Diabetic foot deformity assessment (arch collapse, toe deformities)
- Lower limb alignment analysis (pronation, supination)
- Wound location and size documentation
- Gait analysis preparation
- Pre- and post-surgical comparison

Key features:
- Multi-view 2D pose detection using MediaPipe
- Geometric 3D triangulation from calibrated cameras
- Full body and waist-down capture modes
- Body mesh generation from skeleton data
- Interactive 3D visualization
- No specialized hardware required; a smartphone camera works

What you get, per pose:
- 3D skeleton models (33 joints, OBJ and PLY format)
- Body meshes (5,000+ vertices)
- Interactive HTML visualizations
- Annotated 2D detection images
- Accuracy metrics (reprojection error per joint and per camera)

## Repository structure

```
Kinefeet2_POC/
├── camera_calibration/   # Calibrates cameras: required for accurate results
├── 4camera/               # Ready-made setup for 4 synchronized cameras
├── 8camera/               # Ready-made setup for 8 synchronized cameras
├── scripts/               # Generic pipeline used by all of the above
├── config.yaml            # Rig measurements for the generic root pipeline
├── requirements.txt        # Python dependencies
├── setup.py                # Installation script
├── data/                   # Input photos for the generic root pipeline
├── output/                 # Generated results for the generic root pipeline
├── models/                 # MediaPipe pose model
└── smpl_models/            # Optional: for SMPL-X mesh generation
```

Most users should start with `4camera/` or `8camera/`, which wrap the
scripts in `scripts/` with a ready-made configuration for that number of
cameras. The root `data/`, `output/` and `config.yaml` belong to the generic
pipeline (`scripts/run_pipeline.py`), which supports any number of cameras
at any angles through configuration, for setups that do not match 4 or 8
cameras.

## Camera calibration

Camera calibration is required, not optional. It is what turns pixel
coordinates into metric 3D positions. Without it, the pipeline falls back to
an assumed camera ring built from typed-in measurements, which is far less
accurate than a real calibration.

Calibration uses the Captury colour square-grid board, not a checkerboard.
See [`camera_calibration/README.md`](camera_calibration/README.md) for the
full explanation and instructions:

```bash
cd camera_calibration
python calibrate_intrinsics.py     # focal length, principal point, distortion
python calibrate_extrinsics.py     # where the cameras are, in one world frame
python export_cameras.py           # writes cameras.json for the pipeline
```

## Quick start (4-camera rig)

1. Install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   python setup.py
   ```

2. Calibrate your cameras once, following
   [`camera_calibration/README.md`](camera_calibration/README.md).

3. Place 4 photos per pose in `4camera/data/full_body/pose1/` (and pose2,
   pose3), named `01.jpg` through `04.jpg` in the order described in
   [`4camera/README.md`](4camera/README.md).

4. Run the pipeline:
   ```bash
   cd 4camera/scripts
   python run_pipeline_calibrated.py
   ```

5. Results appear in `4camera/output/full_body/pose1/` (and pose2, pose3):
   - `model.obj`: the 3D skeleton
   - `interactive.html`: a rotatable 3D view in the browser
   - `*_mediapipe_mesh.obj`: the body mesh
   - `preview.png`: a static render

See [`4camera/README.md`](4camera/README.md) and
[`8camera/README.md`](8camera/README.md) for the ready-made rig setups, or
`GUIDE.md` for the generic root pipeline and a full walkthrough covering
photo capture, configuration and troubleshooting.

## How it works

The pipeline converts photos into 3D models in four stages:

1. Photo capture: photos are taken from several angles around the subject,
   who holds a still pose. Each rig folder documents its expected angles and
   file naming.

2. 2D pose detection: MediaPipe analyzes each photo and detects 33 body
   landmarks, including 6 foot landmarks per foot, giving pixel coordinates
   for each joint in each image.

3. 3D triangulation: using the calibrated camera positions and the 2D
   detections from every view, Direct Linear Transform (DLT) triangulation
   computes a 3D coordinate for each joint. This is the same principle as
   stereo vision.

4. Mesh generation: the skeleton is converted into a body mesh, either from
   geometric primitives (the default, MediaPipe-native approach) or by
   fitting an SMPL-X model.

## Accuracy factors

Result accuracy depends on:
1. Subject stillness: any movement between shots introduces error and is
   usually the largest source of inaccuracy in a walk-around capture.
2. Camera calibration: required for metric accuracy; see
   `camera_calibration/README.md`.
3. Rig geometry: if calibration is skipped, the accuracy of the typed-in
   rig measurements in `config.yaml` becomes the limiting factor.
4. Lighting quality: even lighting improves MediaPipe detection.

## Output files

For each pose:

Skeleton files:
- `model.obj`: 3D skeleton, opens in Blender, MeshLab or Windows 3D Viewer
- `model.ply`: point cloud format
- `joints_3d.json`: raw 3D coordinates with accuracy metrics
- `preview.png`: static 3D render
- `interactive.html`: browser-based 3D viewer

Mesh files:
- `*_mediapipe_mesh.obj` / `.ply`: the body mesh
- `*_mediapipe_mesh_interactive.html`: interactive mesh viewer
- `*_mediapipe_mesh_preview.png`: mesh preview image

Diagnostic files:
- `annotated/*.jpg`: photos with the detected skeleton overlaid
- `keypoints_2d/*.json`: raw 2D detections per photo
- `cameras.json`: camera projection matrices
- `reprojection_verification/`: per-camera accuracy report, produced by
  projecting the 3D result back into each photo

## Checking quality

After running the pipeline:

1. Visual inspection: does `preview.png` look like the pose that was shot?
2. Reprojection error: printed to the console during triangulation and
   written to `reprojection_verification/reprojection_summary.txt`.
   Under about 10% of shoulder width is excellent, under 25% is good and
   usable, and above that suggests checking the rig measurements or
   calibration.
3. Annotated images: check `output/*/pose*/annotated/` for detection
   quality before trusting the 3D result.

## Common issues

"NO POSE DETECTED": the subject is too small or far in the frame, or
lighting is poor. Retake the photo with better framing.

High reprojection error: the subject moved between shots, the rig
measurements do not match reality, or the camera angles are not evenly
spaced. Calibrating the cameras (see `camera_calibration/README.md`)
removes the dependency on typed-in rig measurements entirely.

Distorted 3D model: photo filenames must sort alphabetically in the same
order as the camera angles. Check that `01.jpg` corresponds to the first
angle, `02.jpg` to the second, and so on.

Missing joints: the joint was not visible with enough confidence in enough
photos. Lower `min_visibility` in the configuration, or retake the photo
with better framing.

## Advanced topics

Mesh generation offers two approaches:
1. MediaPipe-native (default): works immediately, good quality.
2. SMPL-X (advanced): requires a model download, higher anatomical fidelity.

See `GUIDE.md` for mesh generation details.

To process a new subject or session with the generic root pipeline, create
new folders under `data/`, update `config.yaml` with the new rig
measurements, and run the pipeline again.

## Technical details

- Language: Python 3.9+
- Key libraries: MediaPipe, OpenCV, NumPy, Trimesh
- Triangulation: Direct Linear Transform (DLT)
- Mesh generation: geometric primitives or SMPL-X fitting
- Visualization: Plotly, Matplotlib

## Limitations

- Requires the subject to hold still during capture; this is the largest
  source of error.
- Produces a sparse skeleton, not a dense point cloud.
- A single camera walked around the subject introduces timing error between
  shots; simultaneous multi-camera capture avoids this.
- Best results need several simultaneous cameras and a real calibration.

## Improving results

- Capture with simultaneous multi-camera rigs rather than walking one
  camera around the subject.
- Use more views (8 cameras gives better results than 4, and more still is
  better for occluded joints).
- Calibrate the cameras.
- Use a tripod for a consistent height and distance.
- Ensure even, diffuse lighting.

## Other use cases

Beyond clinical foot assessment, this reconstruction pipeline is generally
useful for:
- Biomechanical analysis
- Pose comparison studies
- Animation reference
- 3D character creation
- Motion analysis
- Ergonomics research

## Support

For detailed instructions, see `GUIDE.md`.
For the calibration system, see `camera_calibration/README.md`.
For the 4-camera and 8-camera rig setups, see `4camera/README.md` and
`8camera/README.md`.

## License

This project uses MediaPipe (Apache 2.0) and other open-source libraries.
See individual library licenses for details.

## Credits

- MediaPipe: Google
- Triangulation: standard computer vision techniques
- SMPL-X: Max Planck Institute (optional, requires separate download)
