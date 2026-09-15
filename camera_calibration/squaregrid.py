"""
squaregrid.py
-------------
Detector and geometric model for the Captury colour square-grid calibration
target (``squaregrid_color_60x40cm``).

THE TARGET IS NOT A CHESSBOARD. It is a grid of *isolated* filled squares on a
white field, so ``cv2.findChessboardCorners`` / ``findCirclesGrid`` /
``aruco`` can never detect it. Geometry, read straight out of the shipped PDF:

    page ................ 400 mm x 600 mm  (held landscape -> 600 x 400)
    grid ................ 6 columns x 4 rows of squares
    pitch ............... 90 mm  (centre to centre, both axes)
    square edge ......... 50 mm  (so a 40 mm white gap between squares)
    colour markers ...... red, green and blue squares near the middle

The three coloured squares make the board fully orientation-resolving: a plain
grid of 6x4 identical squares is ambiguous under a 180 degree rotation (and
under mirroring), which would silently scramble point correspondences between
cameras and wreck extrinsic calibration. The colours pin down exactly one
labelling.

Canonical board frame used everywhere below (``cell`` = ``(col, row)``):

    col index c grows along +X, row index r grows along +Y, board lies in Z=0
    centre of cell (c, r) is at (c * pitch, r * pitch, 0)

    green  at cell (2, 2)      <- anchor / origin of the colour triad
    red    at cell (2, 1)      <- green + (0, -1)   i.e. -Y neighbour
    blue   at cell (3, 2)      <- green + (+1, 0)   i.e. +X neighbour

    c=0   1   2   3   4   5
    .   .   .   .   .   .     r=0
    .   .   R   .   .   .     r=1
    .   .   G   B   .   .     r=2
    .   .   .   .   .   .     r=3

Sub-pixel accuracy
    The projection of a square's centre is *not* the centroid of its projected
    outline (perspective shifts the centroid), but it *is* the intersection of
    the projected diagonals, because a projective map preserves incidence. So
    centres are recovered exactly, by intersecting diagonals of the refined
    quad rather than by taking blob centroids the way a circle-grid detector
    would. Corners themselves are refined by fitting lines to the four edges
    from sub-pixel gradient samples, which is far more stable on an isolated
    convex square than ``cornerSubPix`` (that one is built for saddle points).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Board model
# --------------------------------------------------------------------------

COLS = 6
ROWS = 4
PITCH_MM = 90.0
SQUARE_MM = 50.0

# cell (col, row) of each coloured square
CELL_RED = (2, 1)
CELL_GREEN = (2, 2)
CELL_BLUE = (3, 2)

# Corner order used for every square: relative to the cell centre, in units of
# half a square edge, ordered (-X-Y), (+X-Y), (+X+Y), (-X+Y).
_CORNER_OFFSETS = np.array(
    [[-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0]], dtype=np.float64
)


def cells():
    """All 24 cells as ``(col, row)``, row-major (r outer, c inner)."""
    return [(c, r) for r in range(ROWS) for c in range(COLS)]


def object_centres(unit_mm: float = PITCH_MM):
    """(24, 3) centre of every square in board coordinates, row-major.

    ``unit_mm`` is the physical pitch expressed in the output unit, so the
    default returns millimetres and ``unit_mm=0.09`` returns metres.
    """
    pts = np.array([[c, r, 0.0] for (c, r) in cells()], dtype=np.float64)
    return pts * unit_mm


def object_corners(unit_mm: float = PITCH_MM):
    """(24, 4, 3) the four corners of every square, matching ``CORNER`` order."""
    half = 0.5 * SQUARE_MM / PITCH_MM * unit_mm
    out = np.zeros((len(cells()), 4, 3), dtype=np.float64)
    for i, (c, r) in enumerate(cells()):
        centre = np.array([c, r], dtype=np.float64) * unit_mm
        out[i, :, :2] = centre + _CORNER_OFFSETS * half
    return out


@dataclass
class Detection:
    """One successful board detection in one image."""

    centres: np.ndarray  # (24, 2) float64, row-major over cells()
    corners: np.ndarray  # (24, 4, 2) float64
    image_size: tuple  # (width, height)
    homography: np.ndarray  # (3, 3) board-mm -> pixels
    mirrored: bool = False  # True if the image appears left-right flipped
    refined: int = 0  # how many quads got sub-pixel edge refinement
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Low level geometry helpers
# --------------------------------------------------------------------------


def _order_ccw(pts):
    """Order 4 points counter-clockwise in image coordinates."""
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    return pts[np.argsort(ang)]


def _line_intersection(p1, p2, p3, p4):
    """Intersection of line (p1,p2) with line (p3,p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return np.array(
        [
            (a * (x3 - x4) - (x1 - x2) * b) / den,
            (a * (y3 - y4) - (y1 - y2) * b) / den,
        ]
    )


def _diagonal_centre(quad):
    """Projected centre of the square = intersection of the quad's diagonals."""
    hit = _line_intersection(quad[0], quad[2], quad[1], quad[3])
    return quad.mean(axis=0) if hit is None else hit


def _sample_bilinear(img, pts):
    """Bilinear sample of a float32 single-channel image at (N, 2) xy points."""
    h, w = img.shape
    x = np.clip(pts[:, 0], 0, w - 1.001)
    y = np.clip(pts[:, 1], 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = x - x0
    fy = y - y0
    v00 = img[y0, x0]
    v01 = img[y0, x0 + 1]
    v10 = img[y0 + 1, x0]
    v11 = img[y0 + 1, x0 + 1]
    return (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )


def _fit_line(points):
    """Total-least-squares line fit. Returns two points on the line."""
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean)
    direction = vt[0]
    return mean, mean + direction


def _refine_quad_edges(gray32, quad, n_samples=13):
    """Sub-pixel refinement of a square's 4 corners by fitting its 4 edges.

    For every edge we walk along it and, at each step, read a short intensity
    profile across the edge and take the centroid of |gradient| as the
    sub-pixel edge crossing. A line through those crossings is far more
    accurate than any single corner measurement, and the corners then fall out
    as intersections of adjacent edge lines. Returns ``None`` when the square
    is too small for the profiles to be meaningful.
    """
    sides = np.array(
        [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    )
    side = float(sides.min())
    if side < 12.0:
        return None

    # Half-width of the profile across the edge. The white gap either side of a
    # square is 40 mm against a 50 mm edge, so staying within a quarter of the
    # edge length can never reach the neighbouring square.
    half = max(2.0, min(0.25 * side, 12.0))
    offs = np.arange(-half, half + 1e-9, 0.25)

    lines = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        d = b - a
        length = np.linalg.norm(d)
        if length < 1e-9:
            return None
        along = d / length
        normal = np.array([-along[1], along[0]])

        crossings = []
        # Inset from the corners: corner rounding from printing and blur makes
        # the very ends of an edge unreliable.
        for t in np.linspace(0.22, 0.78, n_samples) * length:
            base = a + along * t
            profile = _sample_bilinear(gray32, base + np.outer(offs, normal))
            grad = np.abs(np.gradient(profile))
            peak = int(np.argmax(grad))
            lo = max(0, peak - 6)
            hi = min(len(grad), peak + 7)
            window = grad[lo:hi]
            total = window.sum()
            if total < 1e-6:
                continue
            crossings.append(base + normal * float((window * offs[lo:hi]).sum() / total))

        if len(crossings) < 4:
            return None
        lines.append(_fit_line(np.array(crossings)))

    refined = np.zeros((4, 2))
    for i in range(4):
        prev = lines[(i - 1) % 4]
        cur = lines[i]
        hit = _line_intersection(prev[0], prev[1], cur[0], cur[1])
        if hit is None or np.linalg.norm(hit - quad[i]) > 0.5 * side:
            return None
        refined[i] = hit
    return refined


# --------------------------------------------------------------------------
# Candidate square detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    """One detection attempt's tolerances.

    ``detect`` walks these from strict to permissive. A close-up of the board
    is found by the first, and the later ones exist for the hard case: a board
    lying on the floor seen from across the room, where each square projects to
    something like 13 x 4 pixels and a strict "roughly square" test throws the
    whole board away.
    """

    name: str
    min_area: float
    max_side_ratio: float  # a square seen edge-on is a very long thin quad
    min_side_px: float
    min_solidity: float
    approx_eps: tuple
    upsample: int  # detect on an enlarged copy, then scale results back


# How badly a grid hypothesis may fit its own homography, as a fraction of the
# cell pitch. Relative, not absolute: the same board fills 200 px per cell in a
# close-up and 25 px across a room, and a pixel means something entirely
# different in each. A true board lands inside a few percent; clutter that
# happens to sit in a grid-ish arrangement does not.
_MAX_GRID_RESIDUAL_FRACTION = 0.08

# Fit good enough to stop looking at more permissive detector settings.
_GOOD_GRID_RESIDUAL_FRACTION = 0.04

# A candidate quad this close to the frame edge is assumed to be clipped by it.
_BORDER_MARGIN_PX = 1.5

# Least pigment a square must carry to count as a verified colour marker, in
# fraction-of-full-scale (see _colour_scores). Real markers score 0.03-0.3 even
# washed out across a room; black squares sit near zero.
_MIN_MARKER_PIGMENT = 0.02

PRESETS = (
    Preset("close", 25.0, 2.5, 3.0, 0.88, (0.035,), 1),
    Preset("oblique", 12.0, 6.0, 1.5, 0.80, (0.02, 0.04), 2),
    Preset("distant", 10.0, 12.0, 1.0, 0.72, (0.015, 0.03, 0.05), 4),
)


def _candidate_quads(gray, preset):
    """All convex quadrilateral dark blobs, de-duplicated by centre."""
    h, w = gray.shape
    max_area = 0.25 * h * w

    binaries = []
    for frac in (0.05, 0.15):
        block = max(11, (int(min(h, w) * frac) | 1))
        binaries.append(
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10
            )
        )
    binaries.append(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    )
    # Canny closes the gap when the board is lit unevenly and neither global
    # nor local thresholding separates every square cleanly.
    edges = cv2.Canny(gray, 50, 150)
    binaries.append(cv2.dilate(edges, np.ones((3, 3), np.uint8)))

    found = []
    for binary in binaries:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < preset.min_area or area > max_area:
                continue
            peri = cv2.arcLength(contour, True)
            for eps in preset.approx_eps:
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = _order_ccw(approx.reshape(4, 2).astype(np.float64))
                # A square cut off by the frame edge is still a convex
                # quadrilateral and passes every shape test below, but it is the
                # intersection of the square with the frame, not the square - so
                # its centre is displaced by half of whatever was cut away.
                # Silently fitting those is worth tens of pixels of error.
                if (
                    quad[:, 0].min() <= _BORDER_MARGIN_PX
                    or quad[:, 1].min() <= _BORDER_MARGIN_PX
                    or quad[:, 0].max() >= w - 1 - _BORDER_MARGIN_PX
                    or quad[:, 1].max() >= h - 1 - _BORDER_MARGIN_PX
                ):
                    continue
                sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
                if min(sides) < preset.min_side_px:
                    continue
                if max(sides) / min(sides) > preset.max_side_ratio:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(approx))
                if hull_area <= 0 or area / hull_area < preset.min_solidity:
                    continue
                found.append((area, quad))
                break

    kept = []
    for area, quad in sorted(found, key=lambda z: -z[0]):
        centre = _diagonal_centre(quad)
        radius = 0.35 * np.sqrt(area)
        if all(np.linalg.norm(centre - k[0]) > radius for k in kept):
            kept.append((centre, quad, area))
    return kept


def _mean_colour(bgr, quad, centre, shrink=0.55):
    """Mean BGR inside a shrunk copy of the quad (avoids edge bleed)."""
    mask = np.zeros(bgr.shape[:2], np.uint8)
    inner = np.round(centre + (quad - centre) * shrink).astype(np.int32)
    cv2.fillConvexPoly(mask, inner, 255)
    if cv2.countNonZero(mask) < 3:
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        if cv2.countNonZero(mask) < 1:
            return None
    return np.array(cv2.mean(bgr, mask)[:3], dtype=np.float64)


def _colour_scores(bgr, candidates):
    """How red / green / blue each candidate is, independent of white balance.

    Absolute HSV thresholds fail badly on a board seen from across a room: the
    markers wash out to the point where the blue square and the white field
    share a hue, separated only by saturation, while a cool white balance
    pushes every neutral surface towards blue anyway.

    Two traps to avoid. Normalised chromaticity (b:g:r ratios) looks like the
    obvious white-balance-free measure, but it is meaningless on dark patches:
    a near-black square whose mean is (3, 2, 2) reads as violently blue, and
    then out-scores the real blue marker. So measure the colour excess in
    *absolute* intensity units, where a black square is by construction almost
    neutral. The illuminant is then removed by noting that a neutral surface's
    colour cast is proportional to how brightly it is lit: fit that
    brightness-proportional part across all the candidates - 21 of the board's
    24 squares are black and the field is white, so the fit is dominated by
    genuinely neutral material - and what is left is real pigment.

    Returns an (N, 3) array of scores in B, G, R order, roughly in units of
    fraction-of-full-scale.
    """
    means = np.array(
        [_mean_colour(bgr, quad, centre) for centre, quad, _ in candidates],
        dtype=object,
    )
    means = np.array(
        [m if m is not None else np.full(3, np.nan) for m in means], dtype=np.float64
    )

    brightness = means.mean(axis=1)
    # Colour excess per channel, summing to zero across channels by construction.
    excess = means - brightness[:, None]

    valid = np.isfinite(brightness)
    tint = np.zeros(3)
    if valid.any():
        b = brightness[valid]
        e = excess[valid]
        weight = np.ones(len(b))
        for _ in range(2):
            denominator = float((weight * b * b).sum())
            if denominator < 1e-9:
                break
            tint = (weight[:, None] * e * b[:, None]).sum(axis=0) / denominator
            residual = np.linalg.norm(e - np.outer(b, tint), axis=1)
            cutoff = np.percentile(residual, 80)
            weight = (residual <= max(cutoff, 1e-6)).astype(float)

    pigment = excess - np.outer(brightness, tint)
    return np.nan_to_num(pigment / 255.0, nan=-1.0)


def _shortlist(scores, candidates, channel, top_k, min_score, within=None):
    """Best candidates for one marker colour, spatially de-duplicated.

    ``within`` is an optional ``(centre, radius)`` box that restricts the search
    to the board's own neighbourhood.
    """
    order = [
        int(i)
        for i in np.argsort(-scores[:, channel])
        if scores[int(i), channel] > min_score
    ][:200]
    if within is not None:
        anchor, radius = within
        order = [i for i in order if np.linalg.norm(candidates[i][0] - anchor) <= radius]

    # Several thresholds see the same physical square, so the raw ranking is
    # full of near-duplicates. Cluster them by position and let each cluster be
    # represented by its *largest* quad: a nested fragment scores just as red
    # as the square containing it, but its size is wrong, which later trips the
    # geometric checks.
    picked = []
    for i in order:
        centre, _, area = candidates[i]
        hit = None
        for slot, rep in enumerate(picked):
            other_centre, _, other_area = candidates[rep]
            if np.linalg.norm(centre - other_centre) < 0.6 * np.sqrt(
                max(area, other_area, 1.0)
            ):
                hit = slot
                break
        if hit is None:
            picked.append(i)
        elif area > candidates[picked[hit]][2]:
            picked[hit] = i
    return picked[:top_k]


def _cell_axes(quad):
    """The quad's two edge vectors, averaged over opposite sides."""
    e1 = ((quad[1] - quad[0]) + (quad[2] - quad[3])) * 0.5
    e2 = ((quad[3] - quad[0]) + (quad[2] - quad[1])) * 0.5
    return e1, e2


_MARKER_CELLS = {"red": CELL_RED, "green": CELL_GREEN, "blue": CELL_BLUE}
_MARKER_CHANNEL = {"blue": 0, "green": 1, "red": 2}


def _match_partners(centres, origin, axes, deltas, remaining, found):
    """Find the other markers given one square's position and local board axes.

    Walks ``remaining`` in order, predicting each marker's position from the
    current axis estimate and yielding every combination of nearby candidates,
    so an ambiguous match becomes two hypotheses rather than a coin toss.

    The important part is that matching an *adjacent* marker replaces the
    predicted axis with the measured one before the next marker is predicted.
    Marker cells are arranged so that one is always a diagonal neighbour of
    another, and predicting a diagonal straight from the seed means
    extrapolating two cell steps from edge vectors measured on a square that may
    be 13 x 5 pixels. That error can exceed half a cell - past the point where
    any tolerance could accept it without also reaching the wrong square.
    Measuring one axis first removes the error instead of trying to absorb it.
    """
    if not remaining:
        yield dict(found)
        return

    other = remaining[0]
    delta = deltas[other]
    span = max(abs(delta).sum(), 1.0)
    predicted = origin + delta[0] * axes[0] + delta[1] * axes[1]
    reach = abs(delta[0]) * np.linalg.norm(axes[0]) + abs(delta[1]) * np.linalg.norm(
        axes[1]
    )
    tol = max(0.45 * reach / span, 4.0)

    distances = np.linalg.norm(centres - predicted, axis=1)
    for k in np.argsort(distances)[:2]:
        if distances[k] > tol:
            break
        k = int(k)
        if k in found.values():
            continue
        refined = list(axes)
        if span == 1:
            axis = 0 if abs(delta[0]) == 1 else 1
            refined[axis] = (centres[k] - origin) / delta[axis]
        yield from _match_partners(
            centres, origin, refined, deltas, remaining[1:], {**found, other: k}
        )


def _marker_confidence(bgr, candidates, assignment):
    """Check the colour markers landed in the cells the board says they should.

    Run *after* a grid hypothesis, on just the squares that hypothesis claimed.
    That is the cleanest colour measurement available anywhere in the pipeline:
    21 of the 24 are black, so the neutral reference is exact, and no background
    clutter can take part. It is also what distinguishes a correct labelling
    from its mirror image, which fits the geometry exactly as well and would
    otherwise hand the calibration a left-handed camera.

    Returns ``(n_best, confidence)``: how many marker cells hold the outright
    best square of their colour, and how far they stand out in total. Two of the
    three must be best; the third is allowed to merely rank well, because the
    markers wash out badly at distance and blue in particular can be nearly
    indistinguishable from a black square. Vetoing on all three would throw away
    correct detections, while requiring none would let the mirrored labelling
    through.
    """
    failed = (-1, -np.inf)
    members = [k for k in assignment if k != -1]
    if len(members) < 8:
        return failed
    subset_scores = _colour_scores(bgr, [candidates[k] for k in members])
    row_of = {k: i for i, k in enumerate(members)}
    index_of_cell = {cell: i for i, cell in enumerate(cells())}

    confidence = 0.0
    n_best = 0
    n_seen = 0
    for role, cell in _MARKER_CELLS.items():
        k = assignment[index_of_cell[cell]]
        if k == -1:
            continue
        column = subset_scores[:, _MARKER_CHANNEL[role]]
        mine = float(column[row_of[k]])
        rest = np.delete(column, row_of[k])
        n_seen += 1
        # Being the *best* of the detected squares is not enough on its own.
        # When the board is half out of frame the true marker may not be among
        # them at all, and then the best blue of twenty black squares is noise
        # that happens to win - which is how a mirrored labelling slips through.
        # A real marker carries real pigment, so demand some.
        if mine > rest.max() and mine > _MIN_MARKER_PIGMENT:
            n_best += 1
        confidence += mine - float(np.median(column))
    if n_seen < 2 or n_best < min(2, n_seen):
        return failed
    return n_best, confidence


def _anchor_triples(bgr, candidates, top_k=12, min_score=0.015, max_triples=24):
    """Plausible (red, green, blue) candidate index triples, best first.

    Yields several guesses rather than one answer: which square is "the red one"
    is genuinely ambiguous when the markers are a few pixels across and the room
    contains skin tones, a red stool and a blue air bed. The caller tries each
    until the 24-cell grid falls into place, which is a far stronger test than
    any colour measurement on its own.

    Geometry does the real work. The three markers sit in known cells, and one
    cell step is 90/50 = 1.8 times a square's own edge - a ratio that survives
    perspective, because the projection is locally affine and affine maps
    preserve ratios along a line. So *any* one square's shape predicts where the
    other two markers must be, to within a fraction of a square. We hypothesise
    each shortlisted square in each marker role, under each of the 8 ways the
    board axes can align with its edges, and keep the hypotheses whose two
    predicted partners land on squares of the right colour. Nothing across the
    room can satisfy that, however red it looks.
    """
    scores = _colour_scores(bgr, candidates)
    centres = np.array([c[0] for c in candidates])
    step = PITCH_MM / SQUARE_MM

    seeds = []
    for role, channel in _MARKER_CHANNEL.items():
        for i in _shortlist(scores, candidates, channel, top_k, min_score):
            seeds.append((role, i))

    triples, seen = [], set()
    for role, i in seeds:
        centre, quad, _ = candidates[i]
        e1, e2 = _cell_axes(quad)
        if min(np.linalg.norm(e1), np.linalg.norm(e2)) < 1.0:
            continue
        home = np.array(_MARKER_CELLS[role], dtype=np.float64)
        others = [r for r in _MARKER_CELLS if r != role]
        deltas = {
            other: np.array(_MARKER_CELLS[other], dtype=np.float64) - home
            for other in others
        }
        # Nearest partner first: matching it lets the frame be re-anchored on a
        # measurement before the further one is predicted. See _match_partners.
        ordered = sorted(others, key=lambda o: abs(deltas[o]).sum())

        for axis_x, sign_x in ((e1, 1), (e1, -1), (e2, 1), (e2, -1)):
            axis_y = e2 if axis_x is e1 else e1
            for sign_y in (1, -1):
                ux = step * sign_x * axis_x
                uy = step * sign_y * axis_y

                for found in _match_partners(
                    centres, centre, [ux, uy], deltas, ordered, {role: i}
                ):
                    if len(set(found.values())) != 3:
                        continue

                    # Partners are pinned by geometry, so do not also demand
                    # each looks like its colour - blue in particular washes
                    # out to near-black at distance. Requiring two of three
                    # keeps this honest while still generating the mirrored
                    # alternative, which the marker check then rules out.
                    confident = sum(
                        1
                        for name, k in found.items()
                        if scores[k, _MARKER_CHANNEL[name]] > min_score
                    )
                    if confident < 2:
                        continue

                    key = (found["red"], found["green"], found["blue"])
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        (
                            scores[key[0], 2] + scores[key[1], 1] + scores[key[2], 0],
                            {"red": key[0], "green": key[1], "blue": key[2]},
                        )
                    )

    triples.sort(key=lambda z: -z[0])
    return [t[1] for t in triples[:max_triples]]


# --------------------------------------------------------------------------
# Grid assignment
# --------------------------------------------------------------------------


def _assign_grid(candidates, anchors, unit_mm=PITCH_MM):
    """Map each of the 24 board cells onto a detected quad.

    Starts from the affine transform implied by the three colour anchors,
    predicts where every cell should land, matches detections to predictions,
    then re-fits a full homography and repeats. Two or three passes is enough
    to converge even on steeply oblique views where the initial affine guess is
    poor at the far edge of the board.
    """
    model = object_centres(unit_mm)
    idx_of = {cell: i for i, cell in enumerate(cells())}

    src = np.float32(
        [model[idx_of[CELL_GREEN]][:2], model[idx_of[CELL_RED]][:2], model[idx_of[CELL_BLUE]][:2]]
    )
    dst = np.float32(
        [
            candidates[anchors["green"]][0],
            candidates[anchors["red"]][0],
            candidates[anchors["blue"]][0],
        ]
    )
    affine = cv2.getAffineTransform(src, dst)
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    mirrored = np.linalg.det(affine[:, :2]) < 0

    detected = np.array([c[0] for c in candidates])
    areas = np.array([c[2] for c in candidates])
    assignment = [-1] * len(model)

    for _ in range(5):
        predicted = cv2.perspectiveTransform(
            model[:, :2].reshape(-1, 1, 2).astype(np.float64), homography
        ).reshape(-1, 2)

        # local pitch in pixels, used both as match radius and as size sanity
        step = np.linalg.norm(predicted[idx_of[CELL_BLUE]] - predicted[idx_of[CELL_GREEN]])
        if not np.isfinite(step) or step < 2.0:
            return None
        tol = 0.45 * step
        expect_area = (SQUARE_MM / PITCH_MM * step) ** 2

        pairs = []
        for m in range(len(model)):
            d = np.linalg.norm(detected - predicted[m], axis=1)
            for k in np.argsort(d)[:3]:
                if d[k] > tol:
                    break
                ratio = areas[k] / expect_area
                if ratio < 0.25 or ratio > 4.0:
                    continue
                pairs.append((d[k], m, int(k)))

        assignment = [-1] * len(model)
        taken = set()
        for _, m, k in sorted(pairs):
            if assignment[m] == -1 and k not in taken:
                assignment[m] = k
                taken.add(k)

        matched = [(m, k) for m, k in enumerate(assignment) if k != -1]
        if len(matched) < 4:
            return None
        new_h, _ = cv2.findHomography(
            np.float64([model[m][:2] for m, _ in matched]),
            np.float64([detected[k] for _, k in matched]),
            0,
        )
        if new_h is None:
            return None
        settled = np.allclose(
            new_h / new_h[2, 2], homography / homography[2, 2], atol=1e-9
        )
        homography = new_h
        if settled:
            break

    # How well one homography explains the squares it claimed. A real board
    # lands well under a pixel; a chance arrangement of clutter does not, so
    # this is what separates a true detection from a plausible-looking fluke.
    matched = [(m, k) for m, k in enumerate(assignment) if k != -1]
    fitted = cv2.perspectiveTransform(
        np.float64([model[m][:2] for m, _ in matched]).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    observed = np.float64([detected[k] for _, k in matched])
    residual = float(np.sqrt((np.linalg.norm(fitted - observed, axis=1) ** 2).mean()))

    # Cell pitch in pixels, so the residual can be judged relative to it.
    neighbours = cv2.perspectiveTransform(
        np.float64(
            [[model[idx_of[CELL_GREEN]][:2]], [model[idx_of[CELL_BLUE]][:2]]]
        ),
        homography,
    ).reshape(2, 2)
    pitch = float(np.linalg.norm(neighbours[1] - neighbours[0]))
    return assignment, homography, mirrored, residual, max(pitch, 1e-6)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


# Upsampling exists to beat contour quantisation on squares a few pixels
# across; on an image that is already large it buys nothing and costs a great
# deal, so cap the working resolution.
_MAX_WORKING_PX = 3000


def _detect_once(bgr, gray, preset, unit_mm, refine, min_cells):
    """One detection attempt at a single set of tolerances."""
    scale = preset.upsample
    while scale > 1 and max(gray.shape) * scale > _MAX_WORKING_PX:
        scale //= 2
    if scale > 1:
        # Enlarging before contour extraction is not cosmetic: findContours
        # returns integer pixel coordinates, so on a square only four pixels
        # tall the corner quantisation alone is a ~12% geometric error.
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    candidates = _candidate_quads(gray, preset)
    if len(candidates) < 4:
        return None

    # Try each plausible colour triple and keep whichever explains the most
    # cells while still fitting one homography tightly. The grid either snaps
    # onto 24 squares or it does not, which is a much sharper test of "did I
    # find the markers" than the colours alone.
    assignment = homography = None
    mirrored = False
    best_n, best_residual = 0, np.inf
    best_rank = (-1, -np.inf)
    for anchors in _anchor_triples(bgr, candidates):
        assigned = _assign_grid(candidates, anchors, unit_mm)
        if assigned is None:
            continue
        n = sum(1 for k in assigned[0] if k != -1)
        residual = assigned[3] / assigned[4]
        if residual > _MAX_GRID_RESIDUAL_FRACTION:
            continue
        n_best, confidence = _marker_confidence(bgr, candidates, assigned[0])
        if n_best < 0:
            continue
        # How many markers verified comes FIRST, ahead of how many squares were
        # found. A hypothesis and its mirror image fit the geometry equally
        # well and can differ by a square or two either way, but accepting the
        # mirror yields a left-handed camera and silently ruins everything
        # downstream - one missing square costs almost nothing by comparison.
        rank = (n_best, n, confidence, -residual)
        if rank > (best_rank[0], best_n, best_rank[1], -best_residual):
            assignment, homography, mirrored = assigned[:3]
            best_n, best_residual = n, residual
            best_rank = (n_best, confidence)
        if n == COLS * ROWS and n_best == 3:
            break
    if assignment is None or best_n < min_cells:
        return None

    gray32 = gray.astype(np.float32)
    model_corners = object_corners(unit_mm)
    centres = np.full((len(assignment), 2), np.nan)
    corners = np.full((len(assignment), 4, 2), np.nan)
    refined_count = 0

    for m, k in enumerate(assignment):
        if k == -1:
            continue
        quad = candidates[k][1]
        if refine:
            better = _refine_quad_edges(gray32, quad)
            if better is not None:
                quad = better
                refined_count += 1

        # Put the quad's corners in canonical order by matching them against
        # the model corners pushed through the current homography.
        want = cv2.perspectiveTransform(
            model_corners[m][:, :2].reshape(-1, 1, 2), homography
        ).reshape(4, 2)
        order = [int(np.argmin(np.linalg.norm(quad - t, axis=1))) for t in want]
        if len(set(order)) != 4:
            continue
        corners[m] = quad[order]
        centres[m] = _diagonal_centre(quad)

    good = ~np.isnan(centres[:, 0])
    if int(good.sum()) < min_cells:
        return None

    if scale > 1:
        centres /= scale
        corners /= scale
        homography = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ homography

    height, width = gray.shape[0] // scale, gray.shape[1] // scale
    return Detection(
        centres=centres,
        corners=corners,
        image_size=(width, height),
        homography=homography,
        mirrored=bool(mirrored),
        refined=refined_count,
        debug={
            "n_candidates": len(candidates),
            "n_cells": int(good.sum()),
            "preset": preset.name,
            "upsample": scale,
            "grid_residual_frac": round(best_residual, 4),
            "markers_verified": int(best_rank[0]),
            "marker_confidence": round(float(best_rank[1]), 4),
        },
    )


def detect(bgr, unit_mm=PITCH_MM, refine=True, min_cells=12, presets=PRESETS):
    """Detect the square-grid board in a BGR image.

    Returns a :class:`Detection` holding the square centres and corners in
    canonical board order, or ``None`` if the board was not found. Cells that
    were not located (clipped at the frame edge, blown out by glare) are left
    as NaN; ``min_cells`` is the number that must survive for the detection to
    count. Partial boards are genuinely useful: OpenCV happily calibrates from
    per-view point sets of differing length, and rejecting them throws away
    exactly the wide-angle views that constrain distortion best.

    Tolerances are tried from strict to permissive and the first attempt that
    recovers the whole grid wins; if none does, the attempt that found the most
    squares is returned. Strict-first matters because the permissive settings
    needed for a distant floor board would otherwise let clutter in on an easy
    close-up.
    """
    if bgr is None:
        return None
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    best = None
    for preset in presets:
        found = _detect_once(bgr, gray, preset, unit_mm, refine, min_cells)
        if found is None:
            continue
        # Stop as soon as a detection is unambiguous and fits tightly. A board
        # short of a few squares is usually clipped by the frame edge, and no
        # amount of extra searching conjures up squares that were never
        # photographed - it only costs time.
        if (
            found.debug["markers_verified"] == 3
            and found.debug["grid_residual_frac"] < _GOOD_GRID_RESIDUAL_FRACTION
            and found.debug["n_cells"] >= int(0.75 * COLS * ROWS)
        ):
            return found
        if best is None or (
            found.debug["markers_verified"],
            found.debug["n_cells"],
            -found.debug["grid_residual_frac"],
        ) > (
            best.debug["markers_verified"],
            best.debug["n_cells"],
            -best.debug["grid_residual_frac"],
        ):
            best = found
    return best


def draw(bgr, detection, radius=4):
    """Annotate a copy of ``bgr`` with the detected grid, for eyeballing."""
    out = bgr.copy()
    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    marks = {CELL_RED: 0, CELL_GREEN: 1, CELL_BLUE: 2}
    for i, (c, r) in enumerate(cells()):
        if np.isnan(detection.centres[i, 0]):
            continue
        quad = detection.corners[i].astype(np.int32)
        cv2.polylines(out, [quad], True, (0, 200, 255), 1, cv2.LINE_AA)
        centre = tuple(np.round(detection.centres[i]).astype(int))
        colour = palette[marks[(c, r)]] if (c, r) in marks else (255, 255, 0)
        cv2.circle(out, centre, radius, colour, -1, cv2.LINE_AA)
        cv2.putText(
            out, f"{c}{r}", (centre[0] + 5, centre[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out
