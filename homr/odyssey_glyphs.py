"""Odyssey additions: find rests and hairpins with the help of Homr's own segmentation.

What Homr knows. Its segmentation network (homr/segmentation) labels five things: stems and
barlines, noteheads, clefs/keys/accidentals, staff lines, and brackets (`InputPredictions.symbols`
is the *brackets* mask, not a general symbol mask). Rests and hairpins are in none of them. The
transformer does read every rest (a token with an attention position) but not hairpins.

So a rest or hairpin can only be found as what is left of the page once everything Homr
recognised is taken away. `PageInk` does that: the ink of the page (binarised against the local
paper tone, so it holds up on shaded, blurred or noisy scans), the staff lines removed with Homr's
staff mask, and the ink that belongs to a notehead, stem, barline, clef, key or accidental marked
as Homr's. Anything connected to Homr's symbols (beams, flags, ties on a note, dots) is not a
rest; what stands alone inside a staff is a rest candidate, and what is a thin straight line below
a staff is a hairpin candidate.

`find_rest_candidates` and `find_hairpins` use it. The rests are then given to the transformer's
rest tokens by the same monotonic alignment used for noteheads (`odyssey_align.align`).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from homr.type_definitions import NDArray

# Printed size of each rest in staff spaces (width, height), measured on engraved and scanned pages.
# Half and whole rests are the same bar, 16th and shorter rests grow by a flag each.
REST_SIZES = {1: (1.2, 0.5), 2: (1.2, 0.5), 4: (1.0, 3.0), 8: (1.0, 2.1), 16: (1.2, 2.9), 32: (1.4, 3.8), 64: (1.6, 4.6)}
_WIDTH_TOLERANCE = (0.65, 1.7)  # of the printed width: wider if a dot or a neighbour touches it
_HEIGHT_TOLERANCE = (0.65, 1.5)
_REST_MIN_AREA = 0.18  # square staff spaces: smaller is a dot, an accent or dirt


@dataclass
class Candidate:
    """An isolated piece of ink inside a staff that Homr did not recognise as anything."""

    x: float  # centre of the bounding box
    y: float
    w: float
    h: float


class PageInk:
    """The page as the user supplied it, with Homr's segmentation taken away. Pixels are those of
    the supplied image (the masks are resampled into it)."""

    def __init__(
        self,
        gray: NDArray,
        masks: dict[str, NDArray] | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
        step: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """
        `gray`: the page, grayscale. `masks`: Homr's segmentation masks by name ("notehead",
        "stems_rest", "clefs_keys", "staff") as predicted, i.e. in the resized/cropped frame;
        `origin` and `step` place that frame in the page (the page point of mask pixel (0, 0) and
        the page distance of one mask pixel).
        """
        self.gray = gray
        self.shape = gray.shape
        self.darkness = _darkness(gray)
        self.ink = self.darkness > 0.3
        self.homr = np.zeros(self.shape, dtype=bool)
        self.staff = np.zeros(self.shape, dtype=bool)
        if masks is not None:
            shape = gray.shape

            def resample(mask: NDArray) -> NDArray:
                sx, sy = step
                matrix = np.array(
                    [[1 / sx, 0, -origin[0] / sx], [0, 1 / sy, -origin[1] / sy]], dtype=np.float32
                )
                return (
                    cv2.warpAffine(
                        mask.astype(np.uint8), matrix, (shape[1], shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
                    )
                    > 0
                )

            owned = np.zeros(shape, dtype=bool)
            for name in ("notehead", "stems_rest", "clefs_keys"):
                if name in masks:
                    owned |= resample(masks[name])
            self.homr = owned
            if "staff" in masks:
                self.staff = resample(masks["staff"])
        self._glyphs: NDArray | None = None
        self._labels: NDArray | None = None
        self._stats: NDArray | None = None
        self._owned: NDArray | None = None

    # -- the ink that is left once the staff lines are gone

    def glyphs(self, unit: float) -> NDArray:
        """
        Ink without the staff lines. A line is what Homr's staff mask marks, or a long horizontal
        run of ink, as long as it is thin: where ink is thicker than a line (a stem or a rest
        crossing it, a beam) it is not a line and stays.
        """
        if self._glyphs is None:
            ink = self.ink.astype(np.uint8)
            thick = max(3, int(round(0.4 * unit)))
            long_run = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(3, int(2.5 * unit))), np.uint8))
            on_line = (long_run > 0) | (cv2.dilate(self.staff.astype(np.uint8), np.ones((3, 1), np.uint8)) > 0)
            thick_ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((thick, 1), np.uint8))
            thick_ink = cv2.dilate(thick_ink, np.ones((3, 1), np.uint8)) > 0
            glyphs = self.ink & ~(on_line & ~thick_ink)
            self._glyphs = cv2.morphologyEx(glyphs.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).astype(bool)
        return self._glyphs

    def components(self, unit: float) -> tuple[NDArray, NDArray, NDArray]:
        """Labels, stats and per component whether it touches something Homr recognised."""
        if self._labels is None:
            glyphs = self.glyphs(unit)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                glyphs.astype(np.uint8), connectivity=8
            )
            # a few stray pixels of Homr's masks on a rest are noise; a stem, notehead or accidental
            # a component is built around is not
            claimed = np.bincount(labels[self.homr & glyphs], minlength=count)
            self._labels, self._stats, self._owned = labels, stats, claimed >= 0.1 * unit * unit
        assert self._stats is not None
        assert self._owned is not None
        return self._labels, self._stats, self._owned


def _darkness(gray: NDArray) -> NDArray:
    """1 - ink/paper: 0 on paper, towards 1 on solid ink, whatever the page's tone and shading."""
    gray = gray.astype(np.float32)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    paper = cv2.GaussianBlur(paper, (0, 0), 4)
    paper = cv2.resize(paper, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - gray / np.maximum(paper, 1.0), 0.0, 1.0)


def rest_kind(rhythm: str) -> int | None:
    """The note value of a rest token (`rest_8` -> 8, `rest_2.` -> 2); None for a multi measure rest."""
    name = rhythm.split("_", 1)[1] if "_" in rhythm else ""
    name = name.rstrip(".")
    return int(name) if name.isdigit() and int(name) in REST_SIZES else None


def rest_fits(kind: int, w: float, h: float) -> bool:
    """Whether a piece of ink `w` by `h` staff spaces can be the printed rest of that note value."""
    pw, ph = REST_SIZES[kind]
    return (
        _WIDTH_TOLERANCE[0] * pw <= w <= _WIDTH_TOLERANCE[1] * pw
        and _HEIGHT_TOLERANCE[0] * ph <= h <= _HEIGHT_TOLERANCE[1] * ph
    )


def find_rest_candidates(
    page: PageInk, line_ys: list[float], unit: float, x_min: float, x_max: float
) -> list[Candidate]:
    """
    Isolated bits of ink inside one staff of rest size and shape, left to right. Whatever touches a
    notehead, stem, barline, clef, key or accidental belongs to Homr's symbols (beams, flags, ties,
    ledger lines, augmentation dots...) and is skipped.
    """
    u = unit
    top, bottom = line_ys[0] - 0.9 * u, line_ys[-1] + 0.9 * u
    r0, r1 = max(int(top), 0), min(int(bottom) + 1, page.shape[0])
    c0, c1 = max(int(x_min), 0), min(int(x_max) + 1, page.shape[1])
    labels, stats, owned = page.components(u)
    sub = np.unique(labels[r0:r1, c0:c1])
    found: list[Candidate] = []
    for c in sub:
        if c == 0 or owned[c]:
            continue
        x, y, w, h, area = (float(v) for v in stats[c])
        cx, cy = x + w / 2, y + h / 2
        if not (c0 <= cx <= c1 and line_ys[0] - 0.3 * u <= cy <= line_ys[-1] + 0.3 * u):
            continue
        if not any(rest_fits(kind, w / u, h / u) for kind in REST_SIZES):
            continue
        if area < _REST_MIN_AREA * u * u:
            continue
        found.append(Candidate(cx, cy, w, h))
    return sorted(found, key=lambda cand: cand.x)


# -- hairpins

_SLOPES = np.arange(-0.3, 0.3001, 0.02)  # tangent of the angle of a wedge's arm
_MIN_ARM = 5.0  # staff spaces: the shortest arm of a hairpin
_MAX_OPENING = 3.5  # staff spaces: the widest a hairpin opens
_STRAIGHT = 0.15  # staff spaces: the most an arm may bend (a slur, even a flat one, bends more)
_MAX_TILT = 0.05  # the axis of a hairpin is level to within this (tangent): a tilted page, a warped staff


@dataclass
class _Arm:
    """A straight stroke: y = y0 + slope * (x - x0) from x0 to x1."""

    x0: float
    x1: float
    y0: float
    slope: float
    strength: float

    def y_at(self, x: float) -> float:
        return self.y0 + self.slope * (x - self.x0)


def _ridges(darkness: NDArray, unit: float) -> tuple[NDArray, NDArray]:
    """
    Strength and slope of the straight, thin, nearly level strokes of a darkness image: for every
    slope the image is sheared level, averaged along the stroke (2 staff spaces) and compared with
    the strip beside it. Stems and noteheads are thicker or not level and give nothing, a stroke a
    stem cuts is bridged by the averaging, and blur only lowers the strength, it doesn't remove it.
    """
    h, w = darkness.shape
    length = max(9, int(2.0 * unit) | 1)
    side = max(3, int(round(0.2 * unit)))
    best = np.zeros((h, w), dtype=np.float32)
    best_slope = np.zeros((h, w), dtype=np.float32)
    xc = w / 2
    for slope in _SLOPES:
        shear = np.array([[1, 0, 0], [-slope, 1, slope * xc]], dtype=np.float32)
        level = cv2.warpAffine(darkness, shear, (w, h), flags=cv2.INTER_LINEAR)
        mean = cv2.blur(level, (length, 1))
        up = np.roll(mean, side, axis=0)
        down = np.roll(mean, -side, axis=0)
        ridge = mean - 0.5 * (up + down)
        back = cv2.warpAffine(ridge, shear, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        better = back > best
        best[better] = back[better]
        best_slope[better] = slope
    return best, best_slope


def _fit_line(xs: NDArray, ys: NDArray, unit: float) -> tuple[NDArray, NDArray] | None:
    """
    A straight line through ridge pixels that ignores stray ones (the tip where two arms meet, a
    bit of the dynamic mark): fit, drop what is further than a third of a staff space, refit.
    Returns the fit and which pixels it keeps, or None if the pixels don't make a straight line.
    """
    keep = np.ones(len(xs), dtype=bool)
    fit = np.polyfit(xs, ys, 1)
    for _ in range(3):
        residual = np.abs(ys - np.polyval(fit, xs))
        keep = residual <= max(1.5, 0.12 * unit)
        if keep.sum() < 0.7 * len(xs) or keep.sum() < 2:
            return None
        fit = np.polyfit(xs[keep], ys[keep], 1)
    residual = np.abs(ys - np.polyval(fit, xs))
    keep = residual <= max(1.5, 0.12 * unit)
    if keep.sum() < 0.8 * len(xs):
        return None
    return fit, keep


def _arms(darkness: NDArray, unit: float) -> list[_Arm]:
    """Straight strokes of at least `_MIN_ARM` staff spaces in a darkness image."""
    strength, _slope = _ridges(darkness, unit)
    up, down = np.roll(strength, 1, axis=0), np.roll(strength, -1, axis=0)
    ridge = (strength > 0.2) & (strength >= up) & (strength > down)
    ridge = cv2.morphologyEx(
        ridge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, max(3, int(0.5 * unit))), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ridge, connectivity=8)
    arms: list[_Arm] = []
    for c in range(1, count):
        x, y, w, h, _area = stats[c]
        if w < _MIN_ARM * unit or h > _MAX_OPENING * unit:
            continue
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == c)
        xs = (xs + x).astype(float)
        ys = (ys + y).astype(float)
        groups = [np.ones(len(xs), dtype=bool)]
        if _fit_line(xs, ys, unit) is None:
            # two arms that meet at the apex: the axis between them splits the pixels, and the
            # pixels at the apex, which belong to both, go to both
            axis = np.polyfit(xs, ys, 1)
            residual = ys - np.polyval(axis, xs)
            near = np.abs(residual) <= 0.08 * unit
            groups = [(residual < 0) | near, (residual > 0) | near]
        for group in groups:
            if group.sum() < _MIN_ARM * unit * 0.8:
                continue
            gx, gy = xs[group], ys[group]
            if gx.max() - gx.min() < _MIN_ARM * unit:
                continue
            fitted = _fit_line(gx, gy, unit)
            if fitted is None:
                continue  # curved: a slur or a tie
            fit, keep = fitted
            x0, x1 = float(gx[keep].min()), float(gx[keep].max())
            if x1 - x0 < _MIN_ARM * unit:
                continue
            arms.append(
                _Arm(x0, x1, float(np.polyval(fit, x0)), float(fit[0]), float(strength[gy[keep].astype(int), gx[keep].astype(int)].mean()))
            )
    return arms


def find_hairpins(
    page: PageInk, line_ys: list[float], unit: float, x_min: float, x_max: float
) -> list[dict]:
    """
    Hairpins (crescendo and diminuendo wedges) below a staff. Neither Homr's segmentation nor its
    transformer knows them, so they are found as two straight, thin strokes that meet at one end
    and open at the other, in the page with everything Homr recognised taken away. Text such as
    "cresc." has no long straight stroke, a slur is curved, a staff or ledger line is level and
    parallel to nothing: none of them pair into a wedge. `open_start`/`open_end` say which end
    is not closed (a hairpin cut by a system break has an open end where it was cut).
    """
    u = unit
    top = int(line_ys[-1] + 0.35 * u)
    bottom = min(int(line_ys[-1] + 7 * u), page.shape[0])
    x0, x1 = int(max(x_min - u, 0)), int(min(x_max + u, page.shape[1]))
    if bottom - top < 2 * u or x1 - x0 < 6 * u:
        return []
    darkness = page.darkness[top:bottom, x0:x1].copy()
    claimed = cv2.dilate(page.homr[top:bottom, x0:x1].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # The staff lines of the next staff run through this strip. Homr's staff mask also marks pieces
    # of a hairpin, so only the components that run the whole width of the staff are taken out.
    side = max(3, int(0.6 * u) | 1)  # the line, and the blur and skew around it
    strip = page.staff[top:bottom, x0:x1].astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)
    for c in range(1, count):
        if stats[c][2] >= 0.85 * (x_max - x_min):
            darkness[cv2.dilate((labels == c).astype(np.uint8), np.ones((side, 1), np.uint8)) > 0] = 0.0
    darkness[claimed] = 0.0
    arms = _arms(darkness, u)
    # every two arms that make a hairpin, the best matched first: a hairpin's arms start and end
    # together, so a stray line that only partly overlaps an arm loses to the arm's real partner
    candidates: list[tuple[float, int, int, dict]] = []
    for i, a in enumerate(arms):
        for j in range(i + 1, len(arms)):
            b = arms[j]
            wedge = _pair(a, b, u)
            if wedge is None:
                continue
            together = (min(a.x1, b.x1) - max(a.x0, b.x0)) / (max(a.x1, b.x1) - min(a.x0, b.x0))
            candidates.append((together * min(a.strength, b.strength), i, j, wedge))
    wedges: list[dict] = []
    used: set[int] = set()
    for _score, i, j, wedge in sorted(candidates, key=lambda c: -c[0]):
        if i in used or j in used:
            continue
        used.update((i, j))
        wedge["x0"] += x0
        wedge["x1"] += x0
        wedge["y"] += top
        wedges.append(wedge)
    return sorted(wedges, key=lambda h: h["x0"])


def _pair(a: _Arm, b: _Arm, u: float) -> dict | None:
    """The hairpin made by two arms, if they make one."""
    left, right = max(a.x0, b.x0), min(a.x1, b.x1)
    shorter = min(a.x1 - a.x0, b.x1 - b.x0)
    longer = max(a.x1 - a.x0, b.x1 - b.x0)
    if right - left < 0.8 * shorter or shorter < 0.6 * longer:
        return None
    if abs(a.slope - b.slope) < 0.01:
        return None  # parallel: staff lines, ledger lines, a tie's two edges
    if abs(a.slope + b.slope) / 2 > _MAX_TILT:
        return None  # the arms of a hairpin open about a level axis; a bit of a slur doesn't
    gap_left = abs(a.y_at(left) - b.y_at(left))
    gap_right = abs(a.y_at(right) - b.y_at(right))
    if max(gap_left, gap_right) > _MAX_OPENING * u or abs(gap_right - gap_left) < 0.4 * u:
        return None
    # the arms converge on one side: the wedge is closed there (or cut by a system break)
    if (a.y_at(left) - b.y_at(left)) * (a.y_at(right) - b.y_at(right)) < 0 and min(gap_left, gap_right) > 0.45 * u:
        return None  # they cross inside the span: an X, not a wedge
    kind = "crescendo" if gap_left < gap_right else "diminuendo"
    closed = 0.8 * u
    return {
        "type": kind,
        "x0": float(left),
        "x1": float(right),
        "y": float((a.y_at(left) + b.y_at(left)) / 2),
        "open_start": bool(kind == "diminuendo" or gap_left > closed),
        "open_end": bool(kind == "crescendo" or gap_right > closed),
    }
