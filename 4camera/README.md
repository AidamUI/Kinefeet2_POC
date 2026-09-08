# 4-Camera Setup

Self-contained setup for 4 synchronized cameras at 0°, 90°, 180°, 270°.

## Structure

```
4camera/
├── scripts/              # Pipeline scripts
│   ├── run_pipeline_4camera.py
│   ├── config_4camera.yaml
│   └── README.md
├── data/                 # Input photos (4 per pose)
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

1. Place 4 photos in `data/full_body/pose1/`:
   - `01.jpg` = 0° (front)
   - `02.jpg` = 90° (right)
   - `03.jpg` = 180° (back)
   - `04.jpg` = 270° (left)

2. Edit `scripts/config_4camera.yaml` with your measurements

3. Run pipeline:
   ```bash
   cd scripts
   python run_pipeline_4camera.py
   ```

4. Check results in `output/`

## Notes

- Uses parent `scripts/` folder for processing
- Isolated data and output from main project
- Optimized for 4 synchronized cameras