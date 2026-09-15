"""
test_detector.py
----------------
Sweeps the detector over synthetic boards with known ground truth.

    python tests/test_detector.py [--views 200]

Reports, per distance band:
  found        the board was detected at all
  labelled     every square it returned is the square it says it is
  accuracy     how far the returned centres are from where they really project
  mirrored     detections with a flipped labelling (must be zero - a mirrored
               labelling makes a left-handed camera and ruins triangulation)

A square is "correctly labelled" when its centre is within a quarter of a cell
of the truth; anything worse means the grid was indexed wrong, which is a
different and much more serious failure than a noisy centre.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import squaregrid as sg  # noqa: E402
import synthetic as syn  # noqa: E402

BANDS = [
    ("close   (0.6-1.2 m)", 0.6, 1.2, 0),
    ("mid     (1.5-3 m)", 1.5, 3.0, 12),
    ("far     (4-7 m)", 4.0, 7.0, 30),
    ("distant (7-10 m)", 7.0, 10.0, 40),
]


def run(views_per_band, size=(1280, 720), focal=900.0, seed=0):
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    rows = []
    for band, (label, near, far, clutter) in enumerate(BANDS):
        # Seed from the band index, not hash(label): string hashing is
        # randomised per process, which would make runs incomparable.
        rng = np.random.default_rng([seed, band])
        found = labelled = mirrored = 0
        errors, cells, pitches, failures = [], [], [], []

        for i in range(views_per_band):
            distance = rng.uniform(near, far)
            rvec, tvec = syn.random_pose(rng, distance)
            image, truth = syn.render(
                K, rvec, tvec, size, unit_mm=syn.UNIT_M,
                clutter=clutter, seed=int(rng.integers(1 << 30)),
            )
            pitch = np.linalg.norm(truth[5] - truth[0]) / 5.0
            pitches.append(pitch)

            detection = sg.detect(image, unit_mm=syn.UNIT_M)
            if detection is None:
                continue
            found += 1
            if detection.mirrored:
                mirrored += 1

            seen = ~np.isnan(detection.centres[:, 0])
            delta = np.linalg.norm(detection.centres[seen] - truth[seen], axis=1)
            cells.append(int(seen.sum()))
            if len(delta) and delta.max() < 0.25 * pitch:
                labelled += 1
                errors.extend(delta.tolist())
            else:
                failures.append((i, distance, pitch, float(delta.max())))

        rows.append(
            {
                "band": label,
                "n": views_per_band,
                "found": found,
                "labelled": labelled,
                "mirrored": mirrored,
                "cells": float(np.mean(cells)) if cells else 0.0,
                "pitch": float(np.mean(pitches)),
                "median_px": float(np.median(errors)) if errors else float("nan"),
                "p95_px": float(np.percentile(errors, 95)) if errors else float("nan"),
                "failures": failures,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=int, default=60, help="views per band")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"\n  {args.views} synthetic views per band, ground truth known\n")
    print(f"  {'band':22s} {'pitch':>7s} {'found':>10s} {'labelled':>10s} "
          f"{'mirrored':>9s} {'cells':>6s} {'err med':>8s} {'p95':>7s}")
    rows = run(args.views, seed=args.seed)
    worst = []
    for r in rows:
        print(f"  {r['band']:22s} {r['pitch']:6.1f}p {r['found']:4d}/{r['n']:<5d} "
              f"{r['labelled']:4d}/{r['n']:<5d} {r['mirrored']:9d} {r['cells']:6.1f} "
              f"{r['median_px']:7.3f}p {r['p95_px']:6.3f}p")
        if r["mirrored"]:
            worst.append(f"{r['band']}: {r['mirrored']} mirrored")
        for view, distance, pitch, err in r["failures"]:
            worst.append(
                f"{r['band']}: view {view} at {distance:.2f} m "
                f"(pitch {pitch:.1f} px) off by {err:.1f} px"
            )

    print()
    if worst:
        print("  FAILURES:")
        for line in worst:
            print(f"    {line}")
    else:
        print("  No mirrored or mislabelled detections.")


if __name__ == "__main__":
    main()
