# Complete Usage Guide

This guide provides step-by-step instructions for using the multi-view 3D pose reconstruction pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Camera Calibration](#camera-calibration)
3. [Photo Capture](#photo-capture)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Understanding Output](#understanding-output)
7. [Mesh Generation](#mesh-generation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Installation

### System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Windows, macOS, or Linux

### Setup Steps

1. **Download the project**
   ```bash
   cd mp3d
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate it:
   # Windows:
   venv\Scripts\activate
   
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   python setup.py
   ```

   This installs all required packages. The first run will download MediaPipe's pose model (~30MB).

### Verify Installation

```bash
python scripts/02_extract_2d_keypoints.py --help
```

If this shows help text, installation succeeded.

---

## Camera Calibration

Camera calibration is optional but highly recommended for accuracy. Do this once per camera/lens/zoom combination.

### Why Calibrate?

Calibration determines your camera's exact focal length and lens distortion, which significantly improves 3D reconstruction accuracy.

### Steps

1. **Print checkerboard pattern**
   - Search "opencv checkerboard 9x6" and print on A4/Letter paper
   - Tape to flat cardboard to keep it rigid
   - Measure one square's size in mm (typically 25mm)

2. **Update square size** (if not 25mm)
   
   Edit `scripts/01_calibrate_camera.py`:
   ```python
   SQUARE_SIZE_MM = 25  # Change to your measured size
   ```

3. **Take calibration photos**
   - Use the EXACT camera/phone/zoom you'll use for body photos
   - Take 12-20 photos of the checkerboard
   - Vary angles, distances, and tilts
   - Keep entire board in frame
   - Save to `data/calibration_images/`

4. **Run calibration**
   ```bash
   python scripts/01_calibrate_camera.py
   ```

5. **Check results**
   - Reprojection error should be < 1.5 pixels
   - Creates `output/camera_intrinsics.json`
   - Pipeline automatically uses this file if present

---

## Photo Capture

This is the most critical step. Poor capture = poor results.

### Equipment Needed

- Camera (smartphone works fine)
- Tripod or consistent hand-held height
- Tape measure
- Masking tape or chalk for floor marks
- Subject willing to hold still

### Rig Setup

1. **Choose location**
   - Even, diffuse lighting (outdoor shade or well-lit room)
   - Avoid harsh shadows on one side
   - Enough space for 2-3 meter radius circle

2. **Mark center point**
   - Subject stands here
   - Mark with tape

3. **Mark camera positions**
   - Measure radius from center (2-3m for full body, 1.5-2m for waist-down)
   - Mark 8 positions at 45° intervals
   - Use string as compass or eyeball "octagon corners + midpoints"

4. **Set camera height**
   - Full body: chest/navel height (~1.0-1.2m)
   - Waist-down: mid-thigh height (~0.5-0.7m)
   - Keep EXACT same height for all 8 shots

5. **Record measurements**
   - Write down: radius, camera height, target height on body
   - You'll enter these in config.yaml

### Shooting Process

For each pose:

1. **Subject assumes pose**
   - Hold completely still
   - Brace against wall/chair if needed
   - Choose sustainable poses (30-60 seconds)

2. **Capture 8 photos**
   - Move camera to each marked position
   - Always aim at same point on body
   - Work quickly to minimize subject fatigue
   - Take photos in order: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

3. **Name files correctly**
   - `01.jpg` = 0° (front)
   - `02.jpg` = 45°
   - `03.jpg` = 90° (right side)
   - `04.jpg` = 135°
   - `05.jpg` = 180° (back)
   - `06.jpg` = 225°
   - `07.jpg` = 270° (left side)
   - `08.jpg` = 315°
   
   **CRITICAL**: Files must sort alphabetically in angle order!

4. **Save to correct folder**
   - Full body: `data/full_body/pose1/` (or pose2, pose3)
   - Waist-down: `data/waist_down/pose1/` (or pose2, pose3)

### Framing Guidelines

**Full Body:**
- Entire body visible with small margin
- Don't crop head or feet
- Subject fills ~70% of frame height

**Waist-Down:**
- Hips to feet clearly visible
- Don't crop ankles/feet
- Subject fills ~80% of frame height

**Both:**
- Portrait or landscape OK (be consistent within pose set)
- Same zoom level for all 8 shots
- Subject centered in frame

### Pro Tips

- **Best approach**: Use 8 cameras simultaneously (eliminates movement error)
- **Good approach**: Single camera on tripod, work quickly
- **Avoid**: Handheld without tripod (height varies too much)
- **Lighting**: Overcast day outdoors is ideal
- **Poses**: Arms away from body, legs slightly apart (easier detection)

---

## Configuration

Edit `config.yaml` with your actual measurements.

### Full Body Configuration

```yaml
rig:
  full_body:
    angles_deg: [0, 45, 90, 135, 180, 225, 270, 315]
    radius_m: 2.5          # YOUR measured distance
    camera_height_m: 1.1   # YOUR measured camera height
    target_height_m: 0.9   # Height on body you aimed at
```

### Waist-Down Configuration

```yaml
  waist_down:
    angles_deg: [0, 45, 90, 135, 180, 225, 270, 315]
    radius_m: 1.5          # Usually smaller than full body
    camera_height_m: 0.55  # Usually lower than full body
    target_height_m: 0.45  # Mid-thigh area
```

### Detection Thresholds

```yaml
detection:
  min_visibility: 0.5     # Lower = accept less confident detections
  min_views: 3            # Minimum cameras that must see a joint
```

**When to adjust:**
- High `min_visibility` (0.7-0.8): Strict, fewer false positives
- Low `min_visibility` (0.3-0.4): Lenient, more complete skeletons
- High `min_views` (4-5): More reliable but may miss occluded joints
- Low `min_views` (2-3): More complete but less reliable

### Fewer Than 8 Angles

If you only took 6 photos at 0°, 60°, 120°, 180°, 240°, 300°:

```yaml
angles_deg: [0, 60, 120, 180, 240, 300]
```

Files must still be named in order: `01.jpg` through `06.jpg`

---

## Running the Pipeline

### Complete Pipeline

Easiest way - runs all steps:

```bash
cd scripts
python run_pipeline.py
```

### Individual Steps

Run each script separately (useful for debugging):

```bash
cd scripts

# Step 1: Extract 2D keypoints from photos
python 02_extract_2d_keypoints.py

# Step 2: Setup camera matrices from config
python 03_setup_camera_rig.py

# Step 3: Triangulate 3D positions
python 04_triangulate_3d.py

# Step 4: Export models and visualizations
python 05_export_and_visualize.py

# Step 5: Generate body meshes
python 07_generate_mediapipe_mesh.py
```

### What Each Script Does

**02_extract_2d_keypoints.py**
- Runs MediaPipe on every photo
- Detects 33 body landmarks per photo
- Saves to `output/*/pose*/keypoints_2d/*.json`
- Creates annotated images in `output/*/pose*/annotated/`
- **Check these annotated images before proceeding!**

**03_setup_camera_rig.py**
- Reads your measurements from config.yaml
- Calculates camera projection matrices
- Saves to `output/*/cameras.json`
- Uses calibration file if available

**04_triangulate_3d.py**
- For each joint, gathers 2D positions from all photos
- Triangulates 3D position using DLT algorithm
- Computes reprojection error (accuracy metric)
- Saves to `output/*/pose*/joints_3d.json`
- **Watch console output for reprojection errors**

**05_export_and_visualize.py**
- Converts 3D joints to OBJ/PLY formats
- Creates preview images
- Generates interactive HTML (if plotly installed)

**07_generate_mediapipe_mesh.py**
- Builds realistic body mesh from skeleton
- Creates head, torso, limbs, hands, feet
- Exports mesh files and interactive viewer

---

## Understanding Output

### Directory Structure

```
output/
├── full_body/
│   ├── cameras.json
│   ├── pose1/
│   │   ├── joints_3d.json
│   │   ├── model.obj
│   │   ├── model.ply
│   │   ├── preview.png
│   │   ├── interactive.html
│   │   ├── *_mediapipe_mesh.obj
│   │   ├── *_mediapipe_mesh.ply
│   │   ├── *_mediapipe_mesh_interactive.html
│   │   ├── *_mediapipe_mesh_preview.png
│   │   ├── annotated/
│   │   │   └── 01.jpg through 08.jpg
│   │   └── keypoints_2d/
│   │       └── 01.json through 08.json
│   ├── pose2/
│   └── pose3/
└── waist_down/
    └── (same structure)
```

### File Descriptions

**joints_3d.json**
- Raw 3D coordinates for all 33 landmarks
- Includes accuracy diagnostics per joint
- Format:
  ```json
  {
    "joints": {
      "left_knee": {
        "index": 25,
        "x": -0.138,
        "y": 0.428,
        "z": -1.811
      }
    },
    "diagnostics": {
      "left_knee": {
        "mean_reprojection_error_px": 11.25,
        "max_reprojection_error_px": 22.76,
        "views_used": [0, 1, 4, 5, 7]
      }
    }
  }
  ```

**model.obj / model.ply**
- 3D skeleton with joints and bones
- Open in Blender, MeshLab, Windows 3D Viewer
- OBJ is more universal, PLY better for point clouds

**preview.png**
- Static 3D render of skeleton
- Quick visual check without 3D software

**interactive.html**
- Rotatable 3D view in browser
- Requires plotly (installed by default)
- Just double-click to open

**Mesh files (*_mediapipe_mesh.*)**
- Realistic body surface mesh
- ~5,000 vertices, ~10,000 faces
- Watertight, suitable for 3D printing
- Open same way as skeleton files

**annotated/*.jpg**
- Your original photos with detected skeleton overlay
- **Always check these first!**
- Red dots = detected joints
- Lines = detected bones
- If detection looks wrong here, 3D will be wrong too

**keypoints_2d/*.json**
- Raw MediaPipe output per photo
- 33 landmarks with x, y, visibility scores
- Usually don't need to look at these

**cameras.json**
- Camera projection matrices
- One 3x4 matrix per angle
- Technical reference, usually don't need to view

### Quality Metrics

**Reprojection Error** (printed during triangulation):
- Measures how well 3D point projects back to 2D observations
- Lower = better
- **< 15px**: Excellent, trustworthy
- **15-40px**: Good, usable for most purposes
- **> 40px**: Poor, check troubleshooting

**Views Used** (in joints_3d.json):
- How many cameras saw this joint
- More views = more reliable
- Minimum is `min_views` from config (default 3)

**Visibility Score** (in keypoints_2d/*.json):
- MediaPipe's confidence (0-1)
- Higher = more confident detection
- Filtered by `min_visibility` from config

---

## Mesh Generation

Two approaches available:

### MediaPipe-Native Mesh (Default)

**Advantages:**
- Works immediately, no setup
- Good quality (~5,000 vertices)
- Fast generation (~5 seconds per pose)
- Anatomically correct proportions

**How it works:**
- Builds body from geometric primitives
- Head: ellipsoid
- Torso: box with depth
- Limbs: tapered cylinders
- Hands/feet: detailed geometry
- All parts smoothly connected

**Usage:**
```bash
python scripts/07_generate_mediapipe_mesh.py
```

**Output:**
- `*_mediapipe_mesh.obj` - Mesh file
- `*_mediapipe_mesh.ply` - Point cloud format
- `*_mediapipe_mesh_interactive.html` - Browser viewer
- `*_mediapipe_mesh_preview.png` - Static image

### SMPL-X Mesh (Advanced)

**Advantages:**
- Highest quality anatomical model
- Industry-standard format
- ~10,000 vertices
- Proper body topology

**Disadvantages:**
- Requires model download and registration
- More complex setup
- Slower generation (~30 seconds per pose)
- May have fitting issues with MediaPipe joints

**Setup:**
1. Register at https://smpl-x.is.tue.mpg.de/
2. Download SMPL-X models
3. Place in `smpl_models/smplx/`:
   - `SMPLX_NEUTRAL.npz`
   - `SMPLX_MALE.npz`
   - `SMPLX_FEMALE.npz`
4. Install: `pip install smplx`

**Usage:**
```bash
python scripts/06_fit_smplx_mesh.py
```

**Note:** SMPL-X fitting may show high loss values. This is expected due to joint structure mismatch between MediaPipe (33 landmarks) and SMPL-X (different joint system). The MediaPipe-native mesh is recommended for most users.

---

## Troubleshooting

### Detection Issues

**"NO POSE DETECTED" for a photo**

Causes:
- Subject too small in frame
- Poor lighting/contrast
- Unusual pose MediaPipe doesn't recognize
- Blurry image

Solutions:
- Retake with subject filling more of frame
- Improve lighting (even, diffuse)
- Try more standard pose
- Use tripod to avoid blur

**Missing joints in 3D model**

Causes:
- Joint not visible in enough photos
- Low visibility scores
- Occluded by clothing/body

Solutions:
- Check annotated images - is joint detected?
- Lower `min_visibility` in config.yaml
- Lower `min_views` in config.yaml
- Retake with better framing/pose

**Joints detected in wrong places**

Causes:
- Tight/patterned clothing confuses MediaPipe
- Poor lighting creates false edges
- Unusual body position

Solutions:
- Wear form-fitting, solid-color clothing
- Improve lighting
- Try more standard pose
- Check if consistent across all 8 photos

### Reconstruction Issues

**High reprojection errors (>40px)**

Causes:
- Subject moved between shots
- Rig measurements wrong in config.yaml
- Camera angles not evenly spaced
- Using approximate FOV (no calibration)

Solutions:
1. Check config.yaml measurements match reality
2. Verify photo filenames sort in angle order
3. Run camera calibration
4. Retake photos with subject holding still
5. Use tripod for consistent camera position

**3D model looks distorted/twisted**

Causes:
- Photo-to-angle mismatch
- Filenames don't sort in capture order

Solutions:
1. Verify `01.jpg` = 0°, `02.jpg` = 45°, etc.
2. Check `angles_deg` in config.yaml matches your actual angles
3. Rename files if needed

**Limbs in wrong positions**

Causes:
- Same as distorted model
- Or: bad detection in one or more photos

Solutions:
1. Check annotated images for that pose
2. Find which photo(s) have bad detection
3. Retake those specific photos
4. Re-run from step 02

**Model too small/large**

Causes:
- Wrong `radius_m` in config.yaml
- Units confusion (meters vs feet)

Solutions:
- Verify measurements are in meters
- Re-measure rig radius
- Update config.yaml
- Re-run from step 03

### Mesh Issues

**Mesh looks wrong but skeleton is good**

Causes:
- Mesh generation issue
- Skeleton has some inaccuracies

Solutions:
- Try MediaPipe-native mesh (script 07)
- Check skeleton reprojection errors
- Improve skeleton quality first

**Mesh has holes or gaps**

Causes:
- Missing joints in skeleton
- Mesh generation algorithm issue

Solutions:
- Fix skeleton completeness first
- Try different mesh generation approach
- Use mesh repair tools (MeshLab, Blender)

### Software Issues

**ModuleNotFoundError**

Cause: Virtual environment not activated or dependencies not installed

Solution:
```bash
# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall
python setup.py
```

**Out of memory**

Cause: Large images or many poses

Solutions:
- Resize images to 1920x1080 or smaller
- Process one pose at a time
- Close other applications

**Slow performance**

Causes:
- Large images
- No GPU acceleration
- Many poses

Solutions:
- Resize images
- Process poses individually
- Be patient (MediaPipe is CPU-intensive)

---

## Advanced Topics

### Using More Cameras

To use 12 cameras at 30° intervals:

```yaml
angles_deg: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
```

Name files `01.jpg` through `12.jpg` in angle order.

More cameras = better accuracy, especially for occluded joints.

### Simultaneous Capture

Best accuracy comes from capturing all angles simultaneously:

**Setup:**
- 8 (or more) cameras on tripods
- All at same height and distance
- Synchronized shutters (timer apps work)

**Advantages:**
- Eliminates subject movement error
- Much more reliable results
- Can use faster poses

**Disadvantages:**
- Requires multiple cameras
- More complex setup

### Custom Poses

You can capture any pose, but consider:

**Good poses:**
- Arms away from body
- Legs slightly apart
- Sustainable for 60 seconds
- All joints visible from multiple angles

**Difficult poses:**
- Arms crossed (occludes torso)
- Hands behind back (not visible)
- Extreme flexibility (MediaPipe may struggle)
- Ground contact (floor occludes joints)

### Batch Processing

To process multiple subjects:

1. Create folders: `data/subject1/`, `data/subject2/`, etc.
2. Update config.yaml for each
3. Run pipeline for each
4. Results go to `output/subject1/`, etc.

### Exporting to Other Formats

**From OBJ to FBX** (for game engines):
```bash
# In Blender:
# File → Import → Wavefront (.obj)
# File → Export → FBX (.fbx)
```

**From OBJ to STL** (for 3D printing):
```bash
# In MeshLab:
# File → Import Mesh → model.obj
# File → Export Mesh As → STL
```

### Animation

To animate the skeleton:

1. Import OBJ into Blender
2. Add armature (bones)
3. Parent mesh to armature
4. Animate bones

Or use multiple poses as keyframes:
1. Import pose1, pose2, pose3
2. Use as animation frames
3. Interpolate between them

### Measurement Accuracy

For research-grade accuracy:

1. **Calibrate camera** (essential)
2. **Use calibrated rig** (measure with precision tools)
3. **Simultaneous capture** (8+ cameras)
4. **Controlled lighting** (photo studio)
5. **Multiple captures** (average results)

Expected accuracy with good setup:
- Joint positions: ±5-10mm
- Limb lengths: ±1-2%
- Angles: ±2-3°

### Integration with Other Tools

**Export to motion capture software:**
- Use OBJ/PLY files
- Or write custom exporter for BVH/C3D formats

**Use in biomechanics analysis:**
- Extract joint angles from joints_3d.json
- Calculate velocities, accelerations
- Analyze gait, posture, movement

**Create training data:**
- Generate many poses
- Use for ML model training
- Augment with synthetic variations

---

## Tips for Best Results

1. **Lighting is critical** - Even, diffuse light beats harsh directional
2. **Subject stillness matters most** - Use braces, short capture time
3. **Calibrate your camera** - 15 minutes once, better results forever
4. **Check annotated images** - Catch problems before 3D reconstruction
5. **Measure carefully** - Rig geometry directly affects accuracy
6. **Use tripod** - Consistent height/distance crucial
7. **Solid colors** - Avoid busy patterns on clothing
8. **Multiple attempts** - First try rarely perfect, iterate
9. **Start simple** - Test with one pose before doing all six
10. **Read error messages** - They usually tell you exactly what's wrong

---

## Getting Help

If you're stuck:

1. Check annotated images in `output/*/pose*/annotated/`
2. Look at reprojection errors in console output
3. Verify config.yaml matches your actual setup
4. Try with just one pose first
5. Review this guide's troubleshooting section
6. Check DEPENDENCIES.md for technical details

Common mistakes:
- Forgetting to activate virtual environment
- Wrong measurements in config.yaml
- Photos not named in angle order
- Subject moved between shots
- Skipping camera calibration

Remember: This is a geometric system, not AI magic. Accuracy depends on your input quality!