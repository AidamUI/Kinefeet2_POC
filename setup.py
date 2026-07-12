#!/usr/bin/env python3
"""
Unified setup script for mp3d project.
Installs all required dependencies and verifies installation.
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"→ {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Verify Python version is 3.9 or higher."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("\n✗ Python 3.9 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        print("  Please upgrade Python and try again")
        return False
    
    print("✓ Python version is compatible")
    return True


def install_dependencies():
    """Install all required packages."""
    print_header("Installing Dependencies")
    
    # Core dependencies
    core_packages = [
        ("mediapipe>=1.0.0", "MediaPipe (pose detection)"),
        ("opencv-python>=4.5.0", "OpenCV (image processing)"),
        ("numpy>=1.20.0", "NumPy (numerical computing)"),
        ("pyyaml>=5.4.0", "PyYAML (configuration)"),
    ]
    
    # Visualization dependencies
    viz_packages = [
        ("matplotlib>=3.3.0", "Matplotlib (plotting)"),
        ("plotly>=5.0.0", "Plotly (interactive 3D)"),
    ]
    
    # Mesh generation dependencies
    mesh_packages = [
        ("trimesh>=3.9.0", "Trimesh (mesh processing)"),
        ("scipy>=1.7.0", "SciPy (scientific computing)"),
        ("torch>=2.0.0", "PyTorch (optimization)"),
    ]
    
    all_packages = core_packages + viz_packages + mesh_packages
    
    failed = []
    
    for package, description in all_packages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", package],
            f"Installing {description}"
        ):
            failed.append(description)
    
    if failed:
        print(f"\n✗ Failed to install: {', '.join(failed)}")
        return False
    
    print("\n✓ All dependencies installed successfully")
    return True


def verify_installation():
    """Verify that all packages can be imported."""
    print_header("Verifying Installation")
    
    packages = [
        ("mediapipe", "MediaPipe"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("trimesh", "Trimesh"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
    ]
    
    failed = []
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name} imported successfully")
        except ImportError as e:
            print(f"  ✗ {name} import failed: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n✗ Failed to import: {', '.join(failed)}")
        print("  Try running setup again or install manually")
        return False
    
    print("\n✓ All packages verified")
    return True


def check_mediapipe_model():
    """Check if MediaPipe model will download on first use."""
    print_header("MediaPipe Model")
    
    print("MediaPipe will download its pose detection model (~30MB)")
    print("on first use. This requires an internet connection.")
    print("\nThe model will be cached for future use.")
    print("✓ No action needed now")


def create_directories():
    """Create necessary directories if they don't exist."""
    print_header("Setting Up Directories")
    
    base_dir = Path(__file__).parent
    
    directories = [
        "data/calibration_images",
        "data/full_body/pose1",
        "data/full_body/pose2",
        "data/full_body/pose3",
        "data/waist_down/pose1",
        "data/waist_down/pose2",
        "data/waist_down/pose3",
        "output",
        "smpl_models/smplx",
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_path}")
        else:
            print(f"  → {dir_path} already exists")
    
    print("\n✓ Directory structure ready")


def print_next_steps():
    """Print instructions for next steps."""
    print_header("Setup Complete!")
    
    print("Next steps:")
    print("\n1. Take photos:")
    print("   - 8 photos per pose at 45° intervals")
    print("   - Place in data/full_body/pose1/ (and pose2, pose3)")
    print("   - Name as 01.jpg through 08.jpg")
    
    print("\n2. Configure measurements:")
    print("   - Edit config.yaml with your rig measurements")
    print("   - Set radius_m, camera_height_m, target_height_m")
    
    print("\n3. Run the pipeline:")
    print("   cd scripts")
    print("   python run_pipeline.py")
    
    print("\n4. View results:")
    print("   - Check output/full_body/pose1/")
    print("   - Open interactive.html in browser")
    print("   - Open model.obj in 3D viewer")
    
    print("\nFor detailed instructions, see GUIDE.md")
    print("For troubleshooting, see GUIDE.md → Troubleshooting section")
    
    print("\n" + "=" * 70)


def main():
    """Main setup function."""
    print("\n" + "=" * 70)
    print("  MP3D - Multi-View 3D Pose Reconstruction")
    print("  Setup Script")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Setup failed during verification")
        sys.exit(1)
    
    # Check MediaPipe model
    check_mediapipe_model()
    
    # Create directories
    create_directories()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()

# Made with Bob
