"""
extract_split_screen_frames.py
-------------------------------
Extract and crop frames from split-screen calibration video.

Usage:
    python extract_split_screen_frames.py video.mp4 \
        --layout 2x2 \
        --camera-order "0deg,90deg,270deg,180deg" \
        --num-frames 15 \
        --output calibration_data/extrinsic

This script:
1. Extracts evenly-spaced frames from video
2. Crops each frame into individual camera views
3. Saves in correct folder structure for calibration
"""

import cv2 as cv
import os
import argparse
import numpy as np
from typing import List, Tuple


def extract_frames_from_video(video_path: str, num_frames: int) -> List[np.ndarray]:
    """
    Extract evenly-spaced frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of frames as numpy arrays
    """
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f'Could not open video: {video_path}')
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f'Video info:')
    print(f'  Total frames: {total_frames}')
    print(f'  FPS: {fps:.2f}')
    print(f'  Duration: {duration:.2f} seconds')
    print(f'  Extracting {num_frames} frames...')
    
    # Calculate frame indices to extract (evenly spaced)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
            timestamp = idx / fps
            print(f'  Frame {len(frames)}/{num_frames} @ {timestamp:.2f}s')
        else:
            print(f'  [!] Could not read frame {idx}')
    
    cap.release()
    return frames


def crop_split_screen(frame: np.ndarray, layout: str, camera_order: List[str]) -> dict:
    """
    Crop split-screen frame into individual camera views.
    
    Args:
        frame: Full split-screen frame
        layout: Layout format (e.g., "2x2", "1x4", "4x1")
        camera_order: Order of cameras in layout (e.g., ["0deg", "90deg", "270deg", "180deg"])
    
    Returns:
        Dictionary mapping camera names to cropped images
    """
    height, width = frame.shape[:2]
    
    # Parse layout
    if 'x' in layout:
        rows, cols = map(int, layout.split('x'))
    else:
        raise ValueError(f'Invalid layout format: {layout}. Use format like "2x2" or "1x4"')
    
    if len(camera_order) != rows * cols:
        raise ValueError(f'Camera order length ({len(camera_order)}) must match layout ({rows}x{cols} = {rows*cols})')
    
    # Calculate crop dimensions
    crop_height = height // rows
    crop_width = width // cols
    
    print(f'  Crop size: {crop_width} x {crop_height}')
    
    # Crop each camera view
    crops = {}
    for idx, camera_name in enumerate(camera_order):
        row = idx // cols
        col = idx % cols
        
        y1 = row * crop_height
        y2 = (row + 1) * crop_height
        x1 = col * crop_width
        x2 = (col + 1) * crop_width
        
        crop = frame[y1:y2, x1:x2]
        crops[f'camera_{camera_name}'] = crop
    
    return crops


def save_cropped_frames(frames: List[np.ndarray], layout: str, camera_order: List[str], 
                       output_dir: str):
    """
    Save cropped frames in calibration folder structure.
    
    Args:
        frames: List of split-screen frames
        layout: Layout format
        camera_order: Order of cameras
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx, frame in enumerate(frames, start=1):
        position_dir = os.path.join(output_dir, f'position_{frame_idx:02d}')
        os.makedirs(position_dir, exist_ok=True)
        
        print(f'\nProcessing frame {frame_idx}/{len(frames)}...')
        
        # Crop frame
        crops = crop_split_screen(frame, layout, camera_order)
        
        # Save each camera view
        for camera_name, crop in crops.items():
            output_path = os.path.join(position_dir, f'{camera_name}.jpg')
            cv.imwrite(output_path, crop)
            print(f'  Saved: {output_path}')
    
    print(f'\n✓ Extracted {len(frames)} positions to {output_dir}')


def main():
    """Main extraction entry point."""
    parser = argparse.ArgumentParser(
        description='Extract and crop frames from split-screen calibration video')
    parser.add_argument('video', help='Path to split-screen video file')
    parser.add_argument('--layout', default='2x2', 
                       help='Split-screen layout (e.g., "2x2", "1x4", "4x1")')
    parser.add_argument('--camera-order', required=True,
                       help='Comma-separated camera order (e.g., "0deg,90deg,270deg,180deg")')
    parser.add_argument('--num-frames', type=int, default=15,
                       help='Number of frames to extract (default: 15)')
    parser.add_argument('--output', default='calibration_data/extrinsic',
                       help='Output directory (default: calibration_data/extrinsic)')
    parser.add_argument('--preview', action='store_true',
                       help='Show preview of first frame crops')
    args = parser.parse_args()
    
    # Parse camera order
    camera_order = [c.strip() for c in args.camera_order.split(',')]
    
    print(f'\n{"="*60}')
    print(f'Split-Screen Frame Extraction')
    print(f'{"="*60}')
    print(f'Video: {args.video}')
    print(f'Layout: {args.layout}')
    print(f'Camera order: {camera_order}')
    print(f'Frames to extract: {args.num_frames}')
    print(f'Output: {args.output}')
    print(f'{"="*60}\n')
    
    # Extract frames
    frames = extract_frames_from_video(args.video, args.num_frames)
    
    if len(frames) == 0:
        print('[!] No frames extracted!')
        return 1
    
    # Preview first frame if requested
    if args.preview and len(frames) > 0:
        print('\nGenerating preview of first frame...')
        crops = crop_split_screen(frames[0], args.layout, camera_order)
        
        for camera_name, crop in crops.items():
            cv.imshow(f'Preview: {camera_name}', crop)
        
        print('Press any key to continue or ESC to cancel...')
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        
        if key == 27:  # ESC
            print('Cancelled by user')
            return 0
    
    # Save cropped frames
    save_cropped_frames(frames, args.layout, camera_order, args.output)
    
    print(f'\n{"="*60}')
    print(f'✓ Extraction complete!')
    print(f'{"="*60}')
    print(f'\nNext steps:')
    print(f'1. Verify images in {args.output}/')
    print(f'2. Run: python calibrate_stereo_rig.py settings_stereo_4camera.yaml intrinsic.pkl')
    
    return 0


if __name__ == "__main__":
    exit(main())
