# Kinefeet 2.0 - Multi-View 3D Foot Reconstruction

**Proof of Concept: Next-generation diabetic foot assessment using computer vision and 3D reconstruction**

## Project Background

Kinefeet is a diabetic foot assessment tool designed to help healthcare providers monitor foot deformities and complications in diabetic patients. Traditional Kinefeet uses manual measurements and 2D photography, which can be time-consuming and subjective.

**This is a Proof of Concept (POC)** for Kinefeet 2.0, which aims to modernize diabetic foot assessment using multi-view 3D reconstruction technology. By capturing photos from multiple angles and using computer vision, we can create accurate 3D models of patients' feet and lower body, enabling:

- **Objective measurements** - Precise 3D distances, angles, and volumes
- **Progress tracking** - Compare 3D models over time to detect changes
- **Remote assessment** - 3D models can be analyzed without patient present
- **Clinical documentation** - Permanent 3D records for medical files

## Overview

This system converts ordinary photos taken from multiple angles into accurate 3D skeletal reconstructions and realistic body meshes. The approach uses geometric triangulation - the same principle your eyes use to perceive depth - rather than AI-based depth estimation.

**Clinical Applications:**
- Diabetic foot deformity assessment (arch collapse, toe deformities)
- Lower limb alignment analysis (pronation, supination)
- Wound location and size documentation
- Gait analysis preparation
- Pre/post-surgical comparison

**Key Features:**
- Multi-view 2D pose detection using MediaPipe
- Geometric 3D triangulation from calibrated cameras
- Full body and waist-down capture modes
- Realistic mesh generation from skeleton data
- Interactive 3D visualization
- No specialized hardware required (smartphone camera works)

**What You Get:**
- 3D skeleton models (33 joints, OBJ/PLY format)
- Realistic body meshes (5,000+ vertices)
- Interactive HTML visualizations
- Annotated 2D detection images
- Detailed accuracy metrics

## Use Cases

### Full Body Mode
Captures entire body from head to toe. Useful for:
- Overall posture and alignment assessment
- Gait preparation (understanding full body mechanics)
- Comprehensive diabetic complication screening
- Research and documentation

### Waist-Down Mode
Focuses on lower body and feet. Ideal for:
- **Diabetic foot assessment** (primary use case)
- Detailed foot deformity analysis
- Lower limb alignment studies
- Faster processing (fewer landmarks to track)

**Protocol:** Capture 3 different poses for each mode to assess:
- Pose 1: Natural standing position
- Pose 2: Weight-bearing variation
- Pose 3: Alternative stance or specific clinical position

## Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd mp3d

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
python setup.py
```

### 2. Capture Photos

Take 8 photos around your subject at evenly-spaced angles (every 45°):
- Use the same camera/phone for all shots
- Keep camera at consistent height and distance
- Subject must hold pose completely still
- Good, even lighting (avoid harsh shadows)

Place photos in:
```
data/full_body/pose1/01.jpg through 08.jpg
data/full_body/pose2/01.jpg through 08.jpg
data/full_body/pose3/01.jpg through 08.jpg
```

### 3. Configure Measurements

Edit `config.yaml` with your actual measurements:
```yaml
rig:
  full_body:
    radius_m: 2.5          # Distance from subject to camera
    camera_height_m: 1.1   # Height of camera from ground
    target_height_m: 0.9   # Height on body you aimed at
```

### 4. Run Pipeline

```bash
cd scripts
python run_pipeline.py
```

Results appear in `output/full_body/pose1/` (and pose2, pose3):
- `model.obj` - 3D skeleton
- `interactive.html` - Rotatable 3D view
- `*_mediapipe_mesh.obj` - Realistic body mesh
- `preview.png` - Static render

## How It Works

### The Kinefeet 2.0 Pipeline

This system uses a 4-step process to convert photos into 3D models:

1. **Photo Capture** - Take 8 photos around the subject at 45° intervals
   - Subject holds completely still in standing position
   - Same camera, same height, same distance for all shots
   - Captures full body OR waist-down (for focused foot assessment)

2. **2D Pose Detection** - MediaPipe analyzes each photo
   - Detects 33 body landmarks (including 6 foot landmarks per foot)
   - Provides pixel coordinates for each joint in each image
   - Works with ordinary smartphone photos

3. **3D Triangulation** - Geometric reconstruction
   - Uses camera positions and 2D detections from all 8 views
   - Calculates exact 3D coordinates for each joint
   - Same principle as stereo vision (how your eyes see depth)

4. **Mesh Generation** - Creates realistic 3D body surface
   - Converts skeleton to smooth, anatomically correct mesh
   - ~5,000 vertices for detailed surface representation
   - Suitable for clinical measurements and visualization

### Why Multi-View?

Single photos can't capture depth. By using 8 cameras positioned around the subject, we can:
- Triangulate accurate 3D positions (like how your two eyes perceive depth)
- Capture the complete foot/body from all angles
- Achieve millimeter-level accuracy for clinical measurements
- Work with ordinary cameras (no depth sensors needed)

### Accuracy Factors

Your results depend on:
1. **Subject stillness** - Any movement between shots causes errors (biggest factor)
2. **Measurement accuracy** - Precise rig geometry improves results
3. **Camera calibration** - Required; see camera_calibration/
4. **Lighting quality** - Even lighting helps MediaPipe detection

## Project Structure

```
mp3d/
├── README.md              # This file
├── GUIDE.md               # Detailed usage guide
├── DEPENDENCIES.md        # Technical dependencies reference
├── config.yaml            # Your rig measurements
├── requirements.txt       # Python dependencies
├── setup.py               # Installation script
├── data/                  # Your input photos
│   ├── calibration_images/
│   ├── full_body/
│   │   ├── pose1/
│   │   ├── pose2/
│   │   └── pose3/
│   └── waist_down/
│       ├── pose1/
│       ├── pose2/
│       └── pose3/
├── scripts/
│   ├── 02_extract_2d_keypoints.py
│   ├── 03_setup_camera_rig.py
│   ├── 04_triangulate_3d.py
│   ├── 05_export_and_visualize.py
│   ├── 07_generate_mediapipe_mesh.py
│   ├── run_pipeline.py
│   └── utils.py
├── output/                # Generated results
└── smpl_models/           # Optional: for advanced mesh generation
```

## Output Files

For each pose, you'll get:

**Skeleton Files:**
- `model.obj` - 3D skeleton (open in Blender, MeshLab, Windows 3D Viewer)
- `model.ply` - Point cloud format
- `joints_3d.json` - Raw 3D coordinates with accuracy metrics
- `preview.png` - Static 3D render
- `interactive.html` - Browser-based 3D viewer

**Mesh Files:**
- `*_mediapipe_mesh.obj` - Realistic body mesh (~5,000 vertices)
- `*_mediapipe_mesh.ply` - Mesh in PLY format
- `*_mediapipe_mesh_interactive.html` - Interactive mesh viewer
- `*_mediapipe_mesh_preview.png` - Mesh preview image

**Diagnostic Files:**
- `annotated/*.jpg` - Photos with detected skeleton overlay
- `keypoints_2d/*.json` - Raw 2D detections per photo
- `cameras.json` - Camera projection matrices

## Viewing Results

### Skeleton Models
- **Windows**: Right-click `model.obj` → Open with → 3D Viewer
- **Blender**: File → Import → Wavefront (.obj)
- **Online**: Upload to https://3dviewer.net/

### Body Meshes
- **Windows**: Right-click `*_mediapipe_mesh.obj` → Open with → 3D Viewer
- **Blender**: File → Import → Wavefront (.obj)
- **Browser**: Open `*_mediapipe_mesh_interactive.html`

## Checking Quality

After running the pipeline, check:

1. **Visual Inspection** - Does `preview.png` look like the pose?
2. **Reprojection Error** - Printed in console during triangulation
   - < 15px: Excellent
   - 15-40px: Good, usable
   - > 40px: Check troubleshooting section
3. **Annotated Images** - Check `output/*/pose*/annotated/` for detection quality

## Common Issues

**"NO POSE DETECTED"**
- Subject too small/far in frame
- Poor lighting or low contrast
- Retake photo with better framing

**High reprojection errors (>40px)**
- Subject moved between shots
- Rig measurements in config.yaml don't match reality
- Camera angles not evenly spaced
- Run camera calibration (see camera_calibration/README.md)

**Distorted 3D model**
- Photo filenames don't sort in same order as angles
- Check that 01.jpg = 0°, 02.jpg = 45°, etc.

**Missing joints**
- Joint not visible in enough photos
- Lower `min_visibility` in config.yaml
- Retake with better framing

## Advanced Features

### Camera Calibration
Required, not optional - it is what makes the 3D output metric. Uses the Captury
colour square-grid board, not a checkerboard:
```bash
cd camera_calibration
python calibrate_intrinsics.py     # focal length, principal point, distortion
python calibrate_extrinsics.py     # where the cameras are, in one world frame
python export_cameras.py           # -> <rig>/output/<version>/cameras.json
```
See [camera_calibration/README.md](camera_calibration/README.md).

### Mesh Generation
Two approaches available:
1. **MediaPipe-native** (default) - Works immediately, good quality
2. **SMPL-X** (advanced) - Requires model download, highest quality

See GUIDE.md for mesh generation details.

### Multiple Subjects
To process different subjects or sessions:
1. Create new folders in `data/`
2. Update `config.yaml` with new rig measurements
3. Run pipeline for each session

## Technical Details

- **Language**: Python 3.9+
- **Key Libraries**: MediaPipe, OpenCV, NumPy, Trimesh
- **Triangulation**: Direct Linear Transform (DLT)
- **Mesh Generation**: Geometric primitives or SMPL-X fitting
- **Visualization**: Plotly, Matplotlib

## Limitations

- Requires subject to hold completely still (biggest limitation)
- Produces sparse skeleton, not dense point cloud
- Single-camera "walk around" capture introduces timing errors
- Best results need 8+ simultaneous cameras (or very still subject)

## Improvements

To get better results:
- Use simultaneous multi-camera capture (eliminates movement error)
- Increase number of views (12-16 cameras better than 8)
- Perform camera calibration
- Use tripod for consistent height/distance
- Ensure even, diffuse lighting

## Use Cases

- Biomechanical analysis
- Pose comparison studies
- Animation reference
- 3D character creation
- Motion analysis
- Ergonomics research

## Support

For detailed instructions, see `GUIDE.md`
For technical dependencies, see `DEPENDENCIES.md`

## License

This project uses MediaPipe (Apache 2.0) and other open-source libraries.
See individual library licenses for details.

## Credits

- MediaPipe: Google
- Triangulation algorithms: Standard computer vision techniques
- SMPL-X: Max Planck Institute (optional, requires separate download)
