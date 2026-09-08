# 4-Camera Setup

Alternative pipeline for 4 synchronized cameras at 0°, 90°, 180°, 270°.

## Usage

```bash
cd 4camera/scripts
python run_pipeline_4camera.py
```

## Configuration

Edit `config_4camera.yaml` with your measurements:
- `angles_deg: [0, 90, 180, 270]` - Fixed for 4 cameras
- `radius_m` - Distance from subject to cameras
- `camera_height_m` - Camera height from floor
- `target_height_m` - Aim point on subject's body
- `min_views: 2` - Minimum cameras needed per joint

## Photo Naming

- `01.jpg` = 0° (front)
- `02.jpg` = 90° (right)
- `03.jpg` = 180° (back)
- `04.jpg` = 270° (left)

Place in `../data/full_body/pose1/` or `../data/waist_down/pose1/` etc.

## Output

Results will be saved to `../output/`