#!/usr/bin/env python3
"""
Setup script for Kinefeet 2.0.

Installs all required dependencies, verifies the installation, and creates
the data and output folders the pipeline expects.
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """Run a command and report whether it succeeded."""
    print(f"{description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  done: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  failed: {description}")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Verify Python version is 3.9 or higher."""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("\nPython 3.9 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        print("  Upgrade Python and try again")
        return False

    print("Python version is compatible")
    return True


def install_dependencies():
    """Install all required packages."""
    print_header("Installing Dependencies")

    core_packages = [
        ("mediapipe>=1.0.0", "MediaPipe (pose detection)"),
        ("opencv-python>=4.5.0", "OpenCV (image processing)"),
        ("numpy>=1.20.0", "NumPy (numerical computing)"),
        ("pyyaml>=5.4.0", "PyYAML (configuration)"),
    ]

    viz_packages = [
        ("matplotlib>=3.3.0", "Matplotlib (plotting)"),
        ("plotly>=5.0.0", "Plotly (interactive 3D)"),
    ]

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
        print(f"\nFailed to install: {', '.join(failed)}")
        return False

    print("\nAll dependencies installed successfully")
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
            print(f"  {name} imported successfully")
        except ImportError as e:
            print(f"  {name} import failed: {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        print("  Try running setup again or install the listed packages manually")
        return False

    print("\nAll packages verified")
    return True


def check_mediapipe_model():
    """Note that MediaPipe's pose model downloads automatically."""
    print_header("MediaPipe Model")

    print("MediaPipe downloads its pose detection model (about 30MB)")
    print("on first use, which requires an internet connection.")
    print("\nThe model is cached afterward for offline use.")
    print("No action needed now.")


def create_directories():
    """Create the data and output folders the pipeline expects."""
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
            print(f"  created {dir_path}")
        else:
            print(f"  {dir_path} already exists")

    print("\nDirectory structure ready")


def print_next_steps():
    """Print instructions for what to do after setup finishes."""
    print_header("Setup Complete")

    print("This root-level setup targets scripts/run_pipeline.py, a generic")
    print("single-rig pipeline configured entirely through config.yaml.")
    print("Most users want one of the ready-made rig folders instead:")
    print("  4camera/  - 4 synchronized cameras, see 4camera/README.md")
    print("  8camera/  - 8 synchronized cameras, see 8camera/README.md")

    print("\nTo use the generic root pipeline:")
    print("\n1. Take photos:")
    print("   - 8 photos per pose at 45 degree intervals")
    print("   - Place in data/full_body/pose1/ (and pose2, pose3)")
    print("   - Name them 01.jpg through 08.jpg")

    print("\n2. Configure measurements:")
    print("   - Edit config.yaml with your rig measurements")
    print("   - Set radius_m, camera_height_m, target_height_m")

    print("\n3. Calibrate the cameras (required for accurate results):")
    print("   - See camera_calibration/README.md")

    print("\n4. Run the pipeline:")
    print("   cd scripts")
    print("   python run_pipeline.py")

    print("\n5. View results:")
    print("   - Check output/full_body/pose1/")
    print("   - Open interactive.html in a browser")
    print("   - Open model.obj in a 3D viewer")

    print("\nFor detailed instructions, see GUIDE.md")
    print("For troubleshooting, see the Troubleshooting section in GUIDE.md")

    print("\n" + "=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  Kinefeet 2.0 - Setup")
    print("=" * 70)

    if not check_python_version():
        sys.exit(1)

    if not install_dependencies():
        print("\nSetup failed during dependency installation")
        sys.exit(1)

    if not verify_installation():
        print("\nSetup failed during verification")
        sys.exit(1)

    check_mediapipe_model()
    create_directories()
    print_next_steps()


if __name__ == "__main__":
    main()
