# Technical Dependencies Reference

This document lists all dependencies used in the project, their purposes, and technical details.

## Core Dependencies

### MediaPipe (1.0.0+)
**Purpose**: 2D pose landmark detection
**License**: Apache 2.0
**Installation**: `pip install mediapipe`

**What it does:**
- Detects 33 body landmarks in images
- Provides x, y coordinates and visibility scores
- Uses machine learning models trained on diverse datasets
- Works on CPU (no GPU required)

**Technical details:**
- Model: BlazePose Heavy (most accurate variant)
- Input: RGB images (any resolution, auto-scaled)
- Output: 33 landmarks with confidence scores
- Processing time: ~100-300ms per image on modern CPU

**Landmarks detected:**
- Face: nose, eyes, ears, mouth (11 points)
- Upper body: shoulders, elbows, wrists, hands (12 points)
- Lower body: hips, knees, ankles, feet (10 points)

### OpenCV (4.5.0+)
**Purpose**: Image processing and camera calibration
**License**: Apache 2.0
**Installation**: `pip install opencv-python`

**What it does:**
- Loads and processes images
- Camera calibration (checkerboard detection)
- Image transformations and annotations
- Drawing detected skeletons on images

**Used for:**
- Reading JPG/PNG files
- Checkerboard corner detection
- Computing camera intrinsics
- Creating annotated output images

### NumPy (1.20.0+)
**Purpose**: Numerical computations
**License**: BSD
**Installation**: `pip install numpy`

**What it does:**
- Matrix operations for triangulation
- Array manipulations for 3D coordinates
- Linear algebra (DLT algorithm)
- Efficient numerical processing

**Key operations:**
- Camera projection matrices (3x4)
- Triangulation via SVD
- Coordinate transformations
- Reprojection error calculations

### PyYAML (5.4.0+)
**Purpose**: Configuration file parsing
**License**: MIT
**Installation**: `pip install pyyaml`

**What it does:**
- Reads config.yaml
- Parses rig measurements
- Loads detection thresholds

## Visualization Dependencies

### Matplotlib (3.3.0+)
**Purpose**: 2D plotting and preview images
**License**: PSF (Python Software Foundation)
**Installation**: `pip install matplotlib`

**What it does:**
- Creates preview.png static renders
- 3D scatter plots of skeletons
- Mesh preview images
- Diagnostic plots

**Used for:**
- Quick visual checks without 3D software
- Publication-ready figures
- Debugging visualizations

### Plotly (5.0.0+)
**Purpose**: Interactive 3D visualizations
**License**: MIT
**Installation**: `pip install plotly`

**What it does:**
- Creates interactive.html files
- Rotatable 3D views in browser
- No additional software needed
- Mesh and skeleton visualization

**Features:**
- Mouse rotation/zoom
- Hover information
- Export to PNG
- Responsive design

## Mesh Generation Dependencies

### Trimesh (3.9.0+)
**Purpose**: 3D mesh processing
**License**: MIT
**Installation**: `pip install trimesh`

**What it does:**
- Creates mesh geometry (vertices, faces)
- Mesh operations (union, intersection)
- Export to OBJ/PLY formats
- Mesh validation and repair

**Used for:**
- Building body meshes from primitives
- Combining mesh parts
- Exporting final models
- Mesh quality checks

### PyTorch (2.0.0+)
**Purpose**: Optimization for SMPL-X fitting
**License**: BSD
**Installation**: `pip install torch`

**What it does:**
- Gradient-based optimization
- SMPL-X model parameter fitting
- Automatic differentiation
- GPU acceleration (optional)

**Note**: Only needed for SMPL-X mesh generation. MediaPipe-native mesh doesn't require PyTorch.

### SciPy (1.7.0+)
**Purpose**: Scientific computing utilities
**License**: BSD
**Installation**: `pip install scipy`

**What it does:**
- Spatial operations (ConvexHull)
- Interpolation for smooth curves
- Optimization algorithms
- Statistical functions

**Used for:**
- Mesh generation algorithms
- Curve fitting
- Spatial data structures

## Optional Dependencies

### SMPL-X
**Purpose**: Parametric human body model
**License**: Custom (requires registration)
**Installation**: `pip install smplx` + model files

**What it does:**
- Provides realistic human body topology
- Parametric shape and pose control
- Industry-standard format
- ~10,000 vertex mesh

**Setup:**
1. Register at https://smpl-x.is.tue.mpg.de/
2. Download model files
3. Place in `smpl_models/smplx/`

**Note**: Optional. MediaPipe-native mesh works without this.

## Development Dependencies

### Python (3.9+)
**Required version**: 3.9 or higher
**Recommended**: 3.10 or 3.11

**Why this version:**
- Type hints support
- Performance improvements
- Library compatibility
- Modern syntax features

### Virtual Environment
**Tool**: venv (built-in)
**Purpose**: Isolated dependency management

**Why use it:**
- Prevents version conflicts
- Clean uninstall
- Project isolation
- Reproducible environments

## System Requirements

### Minimum
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 2 GB free
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04

### Recommended
- **CPU**: Quad-core 3.0 GHz
- **RAM**: 8 GB
- **Storage**: 5 GB free (for models and output)
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+

### GPU
**Not required** - All operations run on CPU
**Optional** - PyTorch can use GPU for SMPL-X fitting (minor speedup)

## Installation Methods

### Standard Installation
```bash
pip install -r requirements.txt
```

### Automated Setup
```bash
python setup.py
```

### Manual Installation
```bash
pip install mediapipe>=1.0.0
pip install opencv-python>=4.5.0
pip install numpy>=1.20.0
pip install pyyaml>=5.4.0
pip install matplotlib>=3.3.0
pip install plotly>=5.0.0
pip install trimesh>=3.9.0
pip install scipy>=1.7.0

# Optional for SMPL-X
pip install torch>=2.0.0
pip install smplx
```

## Dependency Tree

```
mp3d/
├── Core Pipeline
│   ├── mediapipe (pose detection)
│   ├── opencv-python (image processing)
│   ├── numpy (numerical computing)
│   └── pyyaml (configuration)
│
├── Visualization
│   ├── matplotlib (static plots)
│   └── plotly (interactive 3D)
│
├── Mesh Generation
│   ├── trimesh (mesh processing)
│   ├── scipy (spatial operations)
│   └── torch (optional, for SMPL-X)
│
└── Optional
    └── smplx (parametric body model)
```

## Version Compatibility

### Tested Combinations

**Python 3.9:**
- All dependencies work
- Stable, recommended

**Python 3.10:**
- All dependencies work
- Better performance
- Recommended

**Python 3.11:**
- All dependencies work
- Best performance
- Recommended

**Python 3.12:**
- Most dependencies work
- Some packages may need updates
- Generally compatible

### Known Issues

**MediaPipe + Python 3.12:**
- May require latest MediaPipe version
- Check for updates if issues occur

**PyTorch + macOS ARM:**
- Use `pip install torch` (auto-detects ARM)
- GPU acceleration not available on M1/M2

**OpenCV + Linux:**
- May need system packages: `libgl1-mesa-glx`
- Install: `sudo apt-get install libgl1-mesa-glx`

## Disk Space Usage

**Installation:**
- Base dependencies: ~500 MB
- MediaPipe model: ~30 MB
- PyTorch (if used): ~200 MB
- SMPL-X models (if used): ~100 MB
- **Total**: ~800 MB

**Runtime:**
- Input photos: ~50 MB per pose set (8 photos)
- Output files: ~20 MB per pose
- Temporary files: ~10 MB
- **Total per project**: ~100-200 MB

## Network Requirements

**Initial setup:**
- Download dependencies: ~500 MB
- MediaPipe model download: ~30 MB (one-time)
- **Total**: ~530 MB

**Runtime:**
- No internet required after setup
- All processing is local

## Performance Characteristics

### Processing Time (per pose, 8 photos)

**On modern laptop (i7, 16GB RAM):**
- 2D detection: ~2-3 seconds
- Triangulation: <1 second
- Visualization: ~1 second
- Mesh generation: ~5 seconds
- **Total**: ~10 seconds

**On older hardware (i5, 8GB RAM):**
- 2D detection: ~5-8 seconds
- Triangulation: ~1 second
- Visualization: ~2 seconds
- Mesh generation: ~10 seconds
- **Total**: ~20 seconds

### Memory Usage

**Peak memory:**
- Base: ~500 MB
- Per image: ~50 MB
- Mesh generation: ~200 MB
- **Total**: ~1-2 GB

**Typical usage:**
- Idle: ~300 MB
- Processing: ~800 MB
- Peak: ~1.5 GB

## Troubleshooting Dependencies

### Import Errors

**"No module named 'mediapipe'"**
```bash
pip install mediapipe
```

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"No module named 'yaml'"**
```bash
pip install pyyaml
```

### Version Conflicts

**"Package X requires Y<2.0, but you have Y 2.1"**
```bash
# Create fresh environment
python -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Platform-Specific Issues

**Windows: "Microsoft Visual C++ required"**
- Install Visual C++ Redistributable
- Or use pre-built wheels: `pip install --only-binary :all: package_name`

**macOS: "Command not found: python"**
- Use `python3` instead of `python`
- Or create alias: `alias python=python3`

**Linux: "libGL.so.1: cannot open shared object file"**
```bash
sudo apt-get install libgl1-mesa-glx
```

## Updating Dependencies

### Check for updates
```bash
pip list --outdated
```

### Update all
```bash
pip install --upgrade -r requirements.txt
```

### Update specific package
```bash
pip install --upgrade mediapipe
```

### Pin versions (for reproducibility)
```bash
pip freeze > requirements_locked.txt
```

## Alternative Packages

### If MediaPipe doesn't work:
- **OpenPose**: More complex setup, requires GPU
- **AlphaPose**: Good alternative, similar accuracy
- **MMPose**: Research-oriented, flexible

### If Plotly doesn't work:
- **Mayavi**: Desktop 3D visualization
- **PyVista**: Scientific visualization
- **Three.js**: Web-based (requires JavaScript)

### If Trimesh doesn't work:
- **Open3D**: More features, larger install
- **PyMesh**: Research-oriented
- **MeshLab**: External tool, not Python

## License Summary

All dependencies use permissive licenses:
- **Apache 2.0**: MediaPipe, OpenCV
- **BSD**: NumPy, SciPy, PyTorch, Matplotlib
- **MIT**: PyYAML, Plotly, Trimesh

**Commercial use**: Allowed for all dependencies
**Modification**: Allowed for all dependencies
**Distribution**: Allowed with attribution

**Exception**: SMPL-X requires separate registration and has custom license terms.

## Citation Information

If using this project in research, consider citing:

**MediaPipe:**
```
@article{mediapipe2020,
  title={MediaPipe: A Framework for Building Perception Pipelines},
  author={Lugaresi, Camillo and others},
  journal={arXiv preprint arXiv:1906.08172},
  year={2019}
}
```

**SMPL-X (if used):**
```
@inproceedings{SMPL-X:2019,
  title={Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author={Pavlakos, Georgios and others},
  booktitle={CVPR},
  year={2019}
}
```

## Support and Updates

**Dependency issues:**
1. Check this document first
2. Verify Python version (3.9+)
3. Try fresh virtual environment
4. Check package documentation

**Keeping up to date:**
- MediaPipe: Updates frequently, usually safe to upgrade
- OpenCV: Stable, upgrade for bug fixes
- NumPy: Stable, upgrade for performance
- PyTorch: Major versions may break compatibility
- Others: Generally safe to upgrade

**Long-term maintenance:**
- Pin versions for production use
- Test updates in separate environment
- Keep requirements.txt updated
- Document any custom modifications