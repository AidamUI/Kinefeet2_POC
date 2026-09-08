# 8-Camera Setup

Alternative pipeline for 8 cameras at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.

## Usage

```bash
cd 8camera/scripts
python run_pipeline_8camera.py
```

## Configuration

Edit `config_8camera.yaml` with your measurements:
- `angles_deg: [0, 45, 90, 135, 180, 225, 270, 315]` - Fixed for 8 cameras
- `radius_m` - Distance from subject to cameras
- `camera_height_m` - Camera height from floor
- `target_height_m` - Aim point on subject's body
- `min_views: 3` - Minimum cameras needed per joint

## Photo Naming

- `01.jpg` = 0° (front)
- `02.jpg` = 45°
- `03.jpg` = 90° (right)
- `04.jpg` = 135°
- `05.jpg` = 180° (back)
- `06.jpg` = 225°
- `07.jpg` = 270° (left)
- `08.jpg` = 315°

Place in `../data/full_body/pose1/` or `../data/waist_down/pose1/` etc.

## Output

Results will be saved to `../output/`