# 8-Camera Setup

Self-contained setup for 8 cameras at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.

## Structure

```
8camera/
├── scripts/              # Pipeline scripts
│   ├── run_pipeline_8camera.py
│   ├── config_8camera.yaml
│   └── README.md
├── data/                 # Input photos (8 per pose)
│   ├── full_body/
│   │   ├── pose1/
│   │   ├── pose2/
│   │   └── pose3/
│   └── waist_down/
│       ├── pose1/
│       ├── pose2/
│       └── pose3/
└── output/               # Generated 3D models
```

## Quick Start

1. Place 8 photos in `data/full_body/pose1/`:
   - `01.jpg` = 0° (front)
   - `02.jpg` = 45°
   - `03.jpg` = 90° (right)
   - `04.jpg` = 135°
   - `05.jpg` = 180° (back)
   - `06.jpg` = 225°
   - `07.jpg` = 270° (left)
   - `08.jpg` = 315°

2. Edit `scripts/config_8camera.yaml` with your measurements

3. Run pipeline:
   ```bash
   cd scripts
   python run_pipeline_8camera.py
   ```

4. Check results in `output/`

## Notes

- Uses parent `scripts/` folder for processing
- Isolated data and output from main project
- Standard 8-camera configuration for best accuracy