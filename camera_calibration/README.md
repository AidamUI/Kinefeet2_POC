# Camera calibration

Calibrates the Kinefeet rig with OpenCV, from photos of the Captury colour
square-grid board (`squaregrid_color_60x40cm`).

```
python calibrate_intrinsics.py      # K + lens distortion, per camera
python calibrate_extrinsics.py      # where each camera is, in one world frame
python export_cameras.py            # writes cameras.json for the pipeline
```

Add `--debug` to either calibration stage to write annotated images showing
exactly which squares were found and how they were labelled. Look at them the
first time you run against new data; it takes ten seconds and catches most
problems.

## The target

It is **not a chessboard.** It is a grid of *isolated* squares, so
`findChessboardCorners`, `findCirclesGrid` and the ArUco detectors cannot find
it — they have no concept of this pattern. The geometry, read out of the shipped
PDF, is fixed in `squaregrid.py` and has nothing to configure:

| | |
|---|---|
| grid | 6 columns × 4 rows of squares |
| pitch | 90 mm, both axes |
| square | 50 mm, so a 40 mm gap |
| sheet | 600 × 400 mm held landscape |
| markers | red, green and blue squares near the middle |

The three coloured squares are what make the board usable across several
cameras. A plain 6 × 4 grid looks identical rotated 180°, and identical again
mirrored; without the colours, two cameras can label the same physical square
differently and the extrinsics come out silently wrong. The detector resolves
the labelling from the colours and then verifies it against the board's own 24
squares, which is the one colour measurement in the pipeline that no background
clutter can reach.

## What the stages do

**`calibrate_intrinsics.py`** runs `cv2.calibrateCamera` over every view where
the board was found. Before calibrating it fits a single homography to each view
on its own: the board is flat, so whatever that leaves behind is a bowed board,
a blurred frame or rolling-shutter tearing, and those views are dropped rather
than allowed to bias the result. It refuses to mix image sizes, reports the
uncertainty of every parameter it estimates, and re-runs once after discarding
reprojection outliers.

**`calibrate_extrinsics.py`** gets the board's pose in each camera with
`solvePnP`, chains those poses into one shared frame, and refines everything
together by bundle adjustment. Scale is metric for free, from the board's 90 mm
pitch — no tape measure of the rig is involved. It finishes by triangulating the
board from the calibrated cameras and comparing against the physical board,
which is an accuracy figure in millimetres that reprojection error cannot give
you.

It also checks the intrinsics actually belong to the frames it is given, by
comparing the supplied focal length against the one the board's own perspective
implies. This catches the failure where calibration photos and capture photos
come from different crops or zoom levels: a wrong focal length is absorbed
perfectly by placing the board further away, so reprojection error stays low and
only the distances in the room come out wrong.

**`export_cameras.py`** writes `cameras.json` in the form
`scripts/04_triangulate_3d.py` expects, and checks each pose folder's frames
against the calibrated frame size — informationally. A plain resolution
difference (a different export tool, a codec that rounds dimensions to a
multiple of 2 or 16) is not a problem: `04_triangulate_3d.py` rescales K to
match each photo's actual resolution automatically before triangulating, so it
is only flagged `CROPPED` — and only then does it actually matter — when the
*aspect ratio* no longer matches, because that is the one thing rescaling
cannot fix: a crop moves the principal point by an amount nobody recorded.

## Shooting the photos

Everything below matters more than anything in `config.yaml`.

**Use frames straight out of the camera** for the calibration board photos, and
avoid cropping capture photos to the subject. `K` describes one exact frame
geometry: `fx`/`fy` scale with resolution and `cx`/`cy` are pixel coordinates in
that specific frame. A plain resize is fine and corrected automatically (see
above); a crop is not, because nobody recorded where the frame was cut.

**Same zoom and lens throughout**, board photos and subject captures alike —
changing either mid-session describes a different camera than the one that was
calibrated, and no resolution fix-up can recover from that. Resolution itself
is allowed to drift.

**Intrinsics:** 15–25 photos per camera, one camera at a time, board filling a
good part of the frame. Vary the angle hard — tilt it 30–45° in different
directions, not just square-on. Push the board into the corners of the frame in
several shots: distortion is only observable where the board actually reaches,
and views that only cover the middle leave the radial terms unconstrained. Keep
the board rigidly flat; a hand-held foam board flexes, and the flatness screen
will throw those frames out.

**Extrinsics:** put the board where every camera can see it, and capture all
cameras at the same instant. Then move it and do it again — 4–8 placements
spread around the capture volume, tilted differently each time. One placement
technically works but leaves each camera's pose resting on a single view of a
small, distant board, with nothing for bundle adjustment to pull against.

## Tests

Every threshold in `squaregrid.py` was chosen while looking at one set of 40
photos, which is exactly the way to end up with something that works on those 40
photos and nothing else. So the tests render the board with a camera whose K and
pose are known exactly, and check what comes back:

```
python tests/test_detector.py --views 100   # detection, labelling, sub-pixel accuracy
python tests/test_calibration.py            # recovered K and rig vs the rendered truth
```

Nothing in `tests/synthetic.py` was tuned against the real photos, so agreement
between the two is evidence and disagreement is a bug. The synthetic sweep is
what found the frame-clipping bug — a square cut off by the image edge is still
a convex quadrilateral and passes every shape test, but its centre is displaced
by half of whatever was cut away.

`test_calibration.py` exists because reprojection error cannot catch a
calibration that is wrong self-consistently: a focal length that is too long is
absorbed exactly by putting everything further away, and the error shows up only
as every distance in the room being off by a constant factor.

## Output

```
output/intrinsics/<camera>.json    K, distortion, per-view errors, uncertainties
output/extrinsics.json             R, t and camera positions in the world frame
output/debug/                      annotated detections, with --debug
```

World frame (`world_frame: z_up`): origin at the corner square of the first
board placement, +Z up out of the floor, units metres.

## Files

| | |
|---|---|
| `squaregrid.py` | board model and detector |
| `common.py` | config, image discovery, detection cache, the sanity checks |
| `calibrate_intrinsics.py` | stage 1 |
| `calibrate_extrinsics.py` | stage 2 |
| `export_cameras.py` | stage 3 |
| `config.yaml` | the only knobs |
| `tests/synthetic.py` | renders the board with a known camera |
| `tests/test_detector.py` | detection and labelling against ground truth |
| `tests/test_calibration.py` | recovered K and rig against ground truth |
| `tools/extract_split_screen_frames.py` | split a 2×2 split-screen recording into per-camera frames |

Detections are cached under `output/.detection_cache/`, keyed by file content,
so re-running a stage is fast. Delete the folder to force a re-detect.
