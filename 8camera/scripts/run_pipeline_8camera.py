"""
run_pipeline_8camera.py
-----------------------
Convenience script that runs the complete 8-camera pipeline.

This script runs all steps in order for the 8-camera setup:
    1. Extract 2D keypoints from 8 photos
    2. Setup camera rig for 8 cameras at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    3. Triangulate 3D positions from 8 views
    4. Export and visualize 3D models
    5. Generate body meshes
    6. Verify reprojection accuracy

Usage:
    cd 8camera/scripts
    python run_pipeline_8camera.py

Note: This uses config_8camera.yaml for configuration.
"""

import os
import sys
import shutil
import subprocess

# Add parent scripts directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EIGHT_CAM_ROOT = os.path.dirname(SCRIPT_DIR)  # 8camera folder
PROJECT_ROOT = os.path.dirname(EIGHT_CAM_ROOT)  # Kinefeet2_POC folder
PARENT_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, PARENT_SCRIPT_DIR)

# Config paths
CONFIG_8CAM_PATH = os.path.join(SCRIPT_DIR, "config_8camera.yaml")
ORIGINAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

# Data and output paths for 8camera
DATA_8CAM_PATH = os.path.join(EIGHT_CAM_ROOT, "data")
OUTPUT_8CAM_PATH = os.path.join(EIGHT_CAM_ROOT, "output")
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
ORIGINAL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output")

# Steps to run (using original scripts from parent directory)
STEPS = [
    "02_extract_2d_keypoints.py",
    "03_setup_camera_rig.py",
    "04_triangulate_3d.py",
    "05_export_and_visualize.py",
    "07_generate_mediapipe_mesh.py",
    "08_verify_reprojection.py",
]


def main():
    print("=" * 70)
    print("8-CAMERA PIPELINE FOR KINEFEET 2.0")
    print("=" * 70)
    print(f"\nUsing configuration: {CONFIG_8CAM_PATH}")
    print("Camera angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°")
    print("\nExpected photo naming:")
    print("  01.jpg = 0° (front)")
    print("  02.jpg = 45°")
    print("  03.jpg = 90° (right side)")
    print("  04.jpg = 135°")
    print("  05.jpg = 180° (back)")
    print("  06.jpg = 225°")
    print("  07.jpg = 270° (left side)")
    print("  08.jpg = 315°")
    print("\n" + "=" * 70)
    
    # Backup and swap config, data, and output paths
    config_backup = None
    data_backup = None
    output_backup = None
    
    if os.path.exists(ORIGINAL_CONFIG_PATH):
        config_backup = ORIGINAL_CONFIG_PATH + ".backup"
        shutil.copy2(ORIGINAL_CONFIG_PATH, config_backup)
    
    if os.path.exists(ORIGINAL_DATA_PATH):
        data_backup = ORIGINAL_DATA_PATH + ".backup"
        shutil.move(ORIGINAL_DATA_PATH, data_backup)
    
    if os.path.exists(ORIGINAL_OUTPUT_PATH):
        output_backup = ORIGINAL_OUTPUT_PATH + ".backup"
        shutil.move(ORIGINAL_OUTPUT_PATH, output_backup)
    
    # Set up 8camera paths
    shutil.copy2(CONFIG_8CAM_PATH, ORIGINAL_CONFIG_PATH)
    
    # Use junction points on Windows (doesn't require admin privileges)
    # or symbolic links on Unix
    try:
        if os.name == 'nt':  # Windows - use junction
            subprocess.run(['mklink', '/J', ORIGINAL_DATA_PATH, DATA_8CAM_PATH], 
                         shell=True, check=True, capture_output=True)
            subprocess.run(['mklink', '/J', ORIGINAL_OUTPUT_PATH, OUTPUT_8CAM_PATH], 
                         shell=True, check=True, capture_output=True)
        else:  # Unix/Linux/Mac - use symlink
            os.symlink(DATA_8CAM_PATH, ORIGINAL_DATA_PATH)
            os.symlink(OUTPUT_8CAM_PATH, ORIGINAL_OUTPUT_PATH)
    except Exception as e:
        # Fallback: copy directories instead of linking
        print(f"Warning: Could not create links ({e}), copying directories instead...")
        shutil.copytree(DATA_8CAM_PATH, ORIGINAL_DATA_PATH)
        os.makedirs(ORIGINAL_OUTPUT_PATH, exist_ok=True)
    
    print(f"\nUsing 8camera data from: {DATA_8CAM_PATH}")
    print(f"Output will go to: {OUTPUT_8CAM_PATH}\n")
    
    try:
        for step in STEPS:
            path = os.path.join(PARENT_SCRIPT_DIR, step)
            
            print("\n" + "=" * 70)
            print(f"RUNNING {step}")
            print("=" * 70)
            
            # Run the script
            import runpy
            runpy.run_path(path, run_name="__main__")
    
    finally:
        # Restore original paths
        try:
            if os.path.exists(ORIGINAL_DATA_PATH):
                if os.path.islink(ORIGINAL_DATA_PATH) or os.path.isdir(ORIGINAL_DATA_PATH):
                    if os.name == 'nt':
                        # Remove junction on Windows
                        subprocess.run(['rmdir', ORIGINAL_DATA_PATH], shell=True, check=False)
                    else:
                        os.remove(ORIGINAL_DATA_PATH)
            
            if os.path.exists(ORIGINAL_OUTPUT_PATH):
                if os.path.islink(ORIGINAL_OUTPUT_PATH) or os.path.isdir(ORIGINAL_OUTPUT_PATH):
                    if os.name == 'nt':
                        # Remove junction on Windows
                        subprocess.run(['rmdir', ORIGINAL_OUTPUT_PATH], shell=True, check=False)
                    else:
                        os.remove(ORIGINAL_OUTPUT_PATH)
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        
        if config_backup and os.path.exists(config_backup):
            shutil.move(config_backup, ORIGINAL_CONFIG_PATH)
        
        if data_backup and os.path.exists(data_backup):
            shutil.move(data_backup, ORIGINAL_DATA_PATH)
        
        if output_backup and os.path.exists(output_backup):
            shutil.move(output_backup, ORIGINAL_OUTPUT_PATH)
        
        print(f"\n\nRestored original paths")
    
    print("\n" + "=" * 70)
    print("8-CAMERA PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nResults saved to output/ directory")
    print("Check the following for each pose:")
    print("  - model.obj / model.ply (3D skeleton)")
    print("  - *_mediapipe_mesh.obj (body mesh)")
    print("  - interactive.html (rotatable viewer)")
    print("  - reprojection_verification/ (accuracy check)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

# Made with Bob
