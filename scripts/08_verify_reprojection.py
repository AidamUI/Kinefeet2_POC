#!/usr/bin/env python3
"""
Verify 3D reconstruction quality by back-projecting 3D points onto original images.
This creates annotated images showing:
- Original 2D detections (green circles)
- Back-projected 3D points (red crosses)
- Reprojection error lines (yellow)
- Error values in pixels
"""

import os
import numpy as np
import cv2
import json
import yaml
from pathlib import Path
import argparse


def load_cameras(cameras_file):
    """Load all camera parameters from cameras.json."""
    with open(cameras_file, 'r') as f:
        return json.load(f)


def load_2d_keypoints(keypoints_file):
    """Load 2D keypoints from JSON."""
    with open(keypoints_file, 'r') as f:
        data = json.load(f)
    
    # Convert to dict by landmark name
    keypoints_dict = {}
    for lm in data['landmarks']:
        keypoints_dict[lm['name']] = {
            'x': lm['x_px'],
            'y': lm['y_px'],
            'visibility': lm['visibility']
        }
    return keypoints_dict


def load_3d_joints(joints_file):
    """Load 3D joints from JSON."""
    with open(joints_file, 'r') as f:
        data = json.load(f)
    
    # Extract 3D coordinates
    joints_3d = {}
    for name, joint_data in data['joints'].items():
        joints_3d[name] = np.array([joint_data['x'], joint_data['y'], joint_data['z']])
    
    return joints_3d, data.get('diagnostics', {})


def effective_projection_matrix(cam_data, photo_width, photo_height, max_aspect_drift=0.03):
    """This camera's P, rescaled to the resolution the photo actually is.

    cameras.json's K is tied to one exact frame geometry - fx/fy scale with
    resolution, cx/cy are pixel coordinates in that specific frame. Comparing
    its raw P against 2D keypoints measured in a different photo resolution
    (a different export tool, a codec rounding dimensions to a multiple of 2 or
    16) makes this verification measure that resolution mismatch instead of the
    reconstruction's real accuracy - every reported error inflated by roughly
    the resolution ratio, which reads as "camera calibration isn't working"
    when the 3D reconstruction itself (see scripts/04_triangulate_3d.py, which
    already corrects for exactly this) may be fine.

    Mirrors 04_triangulate_3d.py's effective_projection_matrices - duplicated
    rather than imported because module names starting with a digit aren't
    importable with a plain ``import`` statement.
    """
    calib_w, calib_h = cam_data.get("image_width"), cam_data.get("image_height")
    if not (calib_w and calib_h) or (photo_width, photo_height) == (calib_w, calib_h):
        return np.array(cam_data["P"])

    calib_aspect = calib_w / calib_h
    photo_aspect = photo_width / photo_height
    if abs(photo_aspect - calib_aspect) > max_aspect_drift * calib_aspect:
        print(f"    [!] photo is {photo_width}x{photo_height}, calibrated at "
              f"{calib_w}x{calib_h} - aspect ratio differs too much to be a "
              f"resize (likely cropped); using calibration-resolution P "
              f"uncorrected, so these numbers will be biased")
        return np.array(cam_data["P"])

    sx, sy = photo_width / calib_w, photo_height / calib_h
    K = np.array(cam_data["K"], dtype=np.float64).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    R = np.array(cam_data["R"], dtype=np.float64)
    t = np.array(cam_data["t"], dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R, t])


def project_3d_to_2d(point_3d, P):
    """Project a 3D point to 2D using projection matrix P."""
    # Convert to homogeneous coordinates
    point_3d_h = np.append(point_3d, 1.0)
    
    # Project
    point_2d_h = P @ point_3d_h
    
    # Convert from homogeneous
    point_2d = point_2d_h[:2] / point_2d_h[2]
    
    return point_2d


def calculate_reprojection_error(point_2d_detected, point_2d_projected):
    """Calculate Euclidean distance between detected and projected points."""
    return np.linalg.norm(point_2d_detected - point_2d_projected)


def draw_reprojection_comparison(image, keypoints_2d, joints_3d, P, 
                                 landmark_names, show_errors=True):
    """
    Draw comparison between original 2D detections and back-projected 3D points.
    """
    img_annotated = image.copy()
    errors = {}
    
    for name in landmark_names:
        if name not in joints_3d or name not in keypoints_2d:
            continue
        
        # Get 2D detection
        kp_2d = keypoints_2d[name]
        point_2d_detected = np.array([kp_2d['x'], kp_2d['y']])
        
        # Skip if not visible
        if kp_2d.get('visibility', 1.0) < 0.5:
            continue
        
        # Get 3D point and project it
        point_3d = joints_3d[name]
        point_2d_projected = project_3d_to_2d(point_3d, P)
        
        # Calculate error
        error = calculate_reprojection_error(point_2d_detected, point_2d_projected)
        errors[name] = error
        
        # Draw detected point (green circle)
        cv2.circle(img_annotated, 
                  tuple(point_2d_detected.astype(int)), 
                  8, (0, 255, 0), 2)
        
        # Draw projected point (red cross)
        pt_proj = tuple(point_2d_projected.astype(int))
        cv2.drawMarker(img_annotated, pt_proj, (0, 0, 255), 
                      cv2.MARKER_CROSS, 12, 2)
        
        # Draw error line (yellow)
        cv2.line(img_annotated,
                tuple(point_2d_detected.astype(int)),
                pt_proj,
                (0, 255, 255), 1)
        
        # Show error value
        if show_errors:
            text = f"{error:.1f}px"
            text_pos = tuple((point_2d_detected + point_2d_projected).astype(int) // 2)
            cv2.putText(img_annotated, text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_annotated, errors


def create_error_summary_overlay(image, errors, cam_name, person_scale=None):
    """Create summary statistics overlay on image."""
    img_with_stats = image.copy()
    
    # Calculate statistics
    error_values = list(errors.values())
    if not error_values:
        return img_with_stats
    
    mean_error = np.mean(error_values)
    max_error = np.max(error_values)
    min_error = np.min(error_values)
    
    # Calculate normalized error using person scale if available
    img_h, img_w = image.shape[:2]
    if person_scale and person_scale > 0:
        normalization_value = person_scale
        norm_label = "shoulder width"
    else:
        normalization_value = np.sqrt(img_w**2 + img_h**2)
        norm_label = "image diagonal"
    mean_error_pct = (mean_error / normalization_value) * 100
    
    # Create semi-transparent overlay (taller to fit normalized error)
    overlay = img_with_stats.copy()
    cv2.rectangle(overlay, (10, 10), (450, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_with_stats, 0.3, 0, img_with_stats)
    
    # Add text
    y_offset = 35
    cv2.putText(img_with_stats, f"Camera: {cam_name}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_offset += 30
    if person_scale:
        cv2.putText(img_with_stats, f"Image: {img_w}x{img_h}, Person: {person_scale:.0f}px", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(img_with_stats, f"Image: {img_w}x{img_h}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    y_offset += 30
    cv2.putText(img_with_stats, "Reprojection Error:", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img_with_stats, f"Mean: {mean_error:.2f} px ({mean_error_pct:.2f}% of {norm_label})", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img_with_stats, f"Max:  {max_error:.2f} px", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img_with_stats, f"Min:  {min_error:.2f} px", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Add quality assessment (adjusted thresholds for shoulder-width normalization)
    y_offset += 30
    if mean_error_pct < 10.0:
        quality = "Excellent"
        color = (0, 255, 0)
    elif mean_error_pct < 25.0:
        quality = "Good"
        color = (0, 255, 255)
    else:
        quality = "Check rig"
        color = (0, 0, 255)
    cv2.putText(img_with_stats, f"Quality: {quality}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    y_offset += 30
    cv2.circle(img_with_stats, (30, y_offset), 6, (0, 255, 0), 2)
    cv2.putText(img_with_stats, "= Detected 2D", (45, y_offset + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    y_offset += 20
    cv2.drawMarker(img_with_stats, (30, y_offset), (0, 0, 255),
                  cv2.MARKER_CROSS, 10, 2)
    cv2.putText(img_with_stats, "= Projected 3D", (45, y_offset + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img_with_stats


def process_pose(data_dir, pose_name, version, output_dir):
    """Process a single pose and create reprojection visualizations."""
    print(f"\nProcessing {version}/{pose_name}...")
    
    # Paths
    input_pose_dir = output_dir / version / pose_name
    data_pose_dir = data_dir / version / pose_name
    output_pose_dir = output_dir / version / pose_name / "reprojection_verification"
    output_pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 3D joints from OUTPUT directory
    joints_file = input_pose_dir / "joints_3d.json"
    if not joints_file.exists():
        print(f"  [ERROR] joints_3d.json not found at {joints_file}")
        print(f"  Make sure you've run triangulation first (script 04)")
        return
    
    with open(joints_file, 'r') as f:
        joints_data = json.load(f)
    
    joints_3d, diagnostics = load_3d_joints(joints_file)
    print(f"  Loaded {len(joints_3d)} 3D joints")
    
    # Get person scale from joints_3d.json
    person_scale = joints_data.get('person_scale_px', None)
    if person_scale:
        print(f"  Person scale: {person_scale:.0f} px")
    
    # Get image dimensions from first available image
    img_width, img_height = None, None
    
    # Load cameras from OUTPUT directory
    cameras_file = output_dir / version / "cameras.json"
    if not cameras_file.exists():
        print(f"  [ERROR] cameras.json not found at {cameras_file}")
        print(f"  Make sure you've run camera setup (script 03)")
        return
    
    cameras = load_cameras(cameras_file)
    print(f"  Loaded {len(cameras)} cameras")
    
    # Get landmark names
    landmark_names = list(joints_3d.keys())
    
    # Process each camera view
    keypoints_dir = input_pose_dir / "keypoints_2d"
    if not keypoints_dir.exists():
        print(f"  [ERROR] keypoints_2d directory not found")
        return
    
    keypoints_files = sorted(keypoints_dir.glob("*.json"))
    
    all_errors = {}
    
    for kp_file in keypoints_files:
        cam_name = kp_file.stem  # e.g., "01", "02", etc.
        print(f"  Processing camera {cam_name}...")
        
        # Get camera parameters - map "01" to "cam_00", "02" to "cam_01", etc.
        cam_key = f"cam_{int(cam_name) - 1:02d}"
        if cam_key not in cameras:
            print(f"    [SKIP] No camera parameters for {cam_key}")
            continue
        
        cam_data = cameras[cam_key]

        # Load 2D keypoints
        keypoints_2d = load_2d_keypoints(kp_file)

        # Load original image from DATA directory
        image_patterns = [
            data_pose_dir / f"{cam_name}.png",
            data_pose_dir / f"{cam_name}.jpg",
            data_pose_dir / f"{cam_name}.jpeg",
        ]

        image_file = None
        for pattern in image_patterns:
            if pattern.exists():
                image_file = pattern
                break

        if image_file is None:
            print(f"    [SKIP] No image found for {cam_name}")
            continue

        image = cv2.imread(str(image_file))
        if image is None:
            print(f"    [ERROR] Could not load image {image_file}")
            continue

        # Store image dimensions from first image
        photo_height, photo_width = image.shape[:2]
        if img_width is None:
            img_height, img_width = photo_height, photo_width

        # Rescaled to THIS photo's actual resolution, not the one cameras.json
        # was calibrated at - see effective_projection_matrix.
        P = effective_projection_matrix(cam_data, photo_width, photo_height)
        
        # Create reprojection visualization
        img_annotated, errors = draw_reprojection_comparison(
            image, keypoints_2d, joints_3d, P, 
            landmark_names, show_errors=True
        )
        
        # Add statistics overlay
        img_with_stats = create_error_summary_overlay(img_annotated, errors, cam_name, person_scale)
        
        # Save
        output_file = output_pose_dir / f"cam{cam_name}_reprojection.jpg"
        cv2.imwrite(str(output_file), img_with_stats)
        
        # Store errors
        all_errors[f"cam{cam_name}"] = errors
        
        # Print summary
        if errors:
            mean_err = np.mean(list(errors.values()))
            max_err = np.max(list(errors.values()))
            print(f"    Mean error: {mean_err:.2f}px, Max: {max_err:.2f}px")
    
    # Create summary report
    create_summary_report(all_errors, diagnostics, output_pose_dir, pose_name, img_width, img_height, person_scale)
    
    print(f"  [SUCCESS] Saved to {output_pose_dir}")


def create_summary_report(all_errors, diagnostics, output_dir, pose_name, img_width, img_height, person_scale=None):
    """Create a text summary report of reprojection errors."""
    report_file = output_dir / "reprojection_summary.txt"
    
    # Use person scale for normalization if available, otherwise use image diagonal
    if person_scale and person_scale > 0:
        normalization_value = person_scale
        norm_label = "shoulder width"
    else:
        normalization_value = np.sqrt(img_width**2 + img_height**2) if img_width and img_height else 1.0
        norm_label = "image diagonal"
    
    with open(report_file, 'w') as f:
        f.write(f"Reprojection Error Summary: {pose_name}\n")
        f.write("=" * 60 + "\n\n")
        
        if img_width and img_height:
            f.write(f"Image Resolution: {img_width}x{img_height}\n")
            if person_scale:
                f.write(f"Person Scale: {person_scale:.1f} px (shoulder width)\n")
                f.write(f"Normalization: Using shoulder width (pose-invariant)\n\n")
            else:
                f.write(f"Image Diagonal: {normalization_value:.1f} px\n")
                f.write(f"Normalization: Using image diagonal (person scale unavailable)\n\n")
        
        # Per-camera summary
        f.write("Per-Camera Statistics:\n")
        f.write("-" * 60 + "\n")
        
        for cam_name in sorted(all_errors.keys()):
            errors = all_errors[cam_name]
            if not errors:
                continue
            
            error_values = list(errors.values())
            mean_err = np.mean(error_values)
            mean_err_pct = (mean_err / normalization_value) * 100
            
            f.write(f"\n{cam_name}:\n")
            f.write(f"  Mean error: {mean_err:.2f} px ({mean_err_pct:.2f}% of {norm_label})\n")
            f.write(f"  Max error:  {np.max(error_values):.2f} px\n")
            f.write(f"  Min error:  {np.min(error_values):.2f} px\n")
            f.write(f"  Std dev:    {np.std(error_values):.2f} px\n")
        
        # Per-landmark summary
        f.write("\n\nPer-Landmark Statistics:\n")
        f.write("-" * 60 + "\n")
        
        # Collect errors by landmark
        landmark_errors = {}
        for cam_errors in all_errors.values():
            for landmark, error in cam_errors.items():
                if landmark not in landmark_errors:
                    landmark_errors[landmark] = []
                landmark_errors[landmark].append(error)
        
        # Sort by mean error (worst first)
        sorted_landmarks = sorted(landmark_errors.items(), 
                                 key=lambda x: np.mean(x[1]), 
                                 reverse=True)
        
        for landmark, errors in sorted_landmarks:
            f.write(f"\n{landmark}:\n")
            f.write(f"  Mean: {np.mean(errors):.2f} px\n")
            f.write(f"  Max:  {np.max(errors):.2f} px\n")
            f.write(f"  Views: {len(errors)}\n")
            
            # Compare with diagnostics if available
            if diagnostics and landmark in diagnostics:
                diag = diagnostics[landmark]
                diag_mean = diag.get('mean_reprojection_error_px', 'N/A')
                f.write(f"  Diagnostic mean: {diag_mean}\n")
        
        # Overall summary
        f.write("\n\nOverall Summary:\n")
        f.write("-" * 60 + "\n")
        
        all_error_values = []
        for cam_errors in all_errors.values():
            all_error_values.extend(cam_errors.values())
        
        if all_error_values:
            mean_overall = np.mean(all_error_values)
            mean_overall_pct = (mean_overall / normalization_value) * 100
            
            f.write(f"Total measurements: {len(all_error_values)}\n")
            f.write(f"Mean error: {mean_overall:.2f} px ({mean_overall_pct:.2f}% of {norm_label})\n")
            f.write(f"Max error:  {np.max(all_error_values):.2f} px\n")
            f.write(f"Min error:  {np.min(all_error_values):.2f} px\n")
            f.write(f"Std dev:    {np.std(all_error_values):.2f} px\n")
            
            # Quality assessment (adjusted thresholds for shoulder-width normalization)
            f.write(f"\nQuality Assessment:\n")
            if mean_overall_pct < 10.0:
                f.write(f"  EXCELLENT - Error is {mean_overall_pct:.2f}% of {norm_label}\n")
            elif mean_overall_pct < 25.0:
                f.write(f"  GOOD - Error is {mean_overall_pct:.2f}% of {norm_label}\n")
            else:
                f.write(f"  NEEDS IMPROVEMENT - Error is {mean_overall_pct:.2f}% of {norm_label}\n")
                f.write(f"  Consider checking camera rig measurements or subject movement\n")


def process_version(version, data_dir, output_dir, pose_arg):
    """Process all (or one) pose(s) for a single version (full_body / waist_down)."""
    output_version_dir = output_dir / version
    if not output_version_dir.exists():
        print(f"[ERROR] Directory not found: {output_version_dir}")
        return

    if pose_arg:
        poses = [pose_arg]
    else:
        poses = sorted([d.name for d in output_version_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pose')])

    if not poses:
        print(f"[ERROR] No poses found in {output_version_dir}")
        return

    print(f"\nProcessing {len(poses)} pose(s) from {version}...")

    for pose_name in poses:
        try:
            process_pose(data_dir, pose_name, version, output_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {pose_name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    # Relative to THIS FILE, not the process's current working directory. Every
    # other numbered script resolves its paths the same way (via __file__);
    # this one used to default to the plain strings '../data' / '../output',
    # which resolve against whatever the CWD happened to be at launch - the
    # project root if run directly, but something else entirely when launched
    # through 4camera's or 8camera's pipeline wrappers, which is exactly when
    # getting it wrong matters, since those wrappers point data/ and output/ at
    # a *borrowed* per-rig copy that only exists at a fixed absolute path.
    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

    parser = argparse.ArgumentParser(
        description="Verify 3D reconstruction by back-projecting to 2D images"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=default_data_dir,
        help='Data directory containing pose folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_output_dir,
        help='Output directory for verification images'
    )
    parser.add_argument(
        '--pose',
        type=str,
        help='Specific pose to process (e.g., pose1). If not specified, processes all poses.'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        choices=['full_body', 'waist_down'],
        help='Version to process. If not specified, processes BOTH full_body and waist_down.'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("3D Reconstruction Verification via Back-Reprojection")
    print("=" * 60)

    versions = [args.version] if args.version else ['full_body', 'waist_down']

    for version in versions:
        process_version(version, data_dir, output_dir, args.pose)

    print("\n" + "=" * 60)
    print("[SUCCESS] Verification complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()