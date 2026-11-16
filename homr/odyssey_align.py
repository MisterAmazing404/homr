"""Odyssey additions: snap the transformer's imprecise symbol positions to detections.

The transformer reports where each symbol is from its attention, which can be off by
several note widths and isn't monotonic. The segmentation stage finds noteheads and
barlines exactly, but doesn't know which token they belong to. Both sequences run left to
right, so we align them (edit distance) and take the detected position for matched symbols.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from homr.type_definitions import NDArray

_GAP_COST = 1.0
_MAX_MATCH_COST = 1.5
UNMATCHED_PENALTY = 0.6


@dataclass
class StaffGeometry:
    """One five-line staff in the coordinates of the image the user provided."""

    system: int
    part: int
    half: int  # 0: upper (or only) staff, 1: lower staff of a grand staff
    line_ys: list[float]  # average y of each of the 5 lines
    x_min: float
    x_max: float
    unit: float  # distance between two lines
    bars: list[float] = field(default_factory=list)  # detected barline x positions, sorted
    # hairpins below the staff: {"type": "crescendo" | "diminuendo", "x0", "x1", "y", "open_start", "open_end"}
    hairpins: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "system": self.system,
            "part": self.part,
            "half": self.half,
            "line_ys": self.line_ys,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "unit": self.unit,
            "bars": self.bars,
            "hairpins": self.hairpins,
        }


def align(symbol_xs: list[float | None], candidate_xs: list[float], unit: float) -> dict[int, int]:
    """
    Minimum cost monotonic matching of symbols to detections.
    Returns {symbol index: candidate index} for the matched pairs.
    """
    n, m = len(symbol_xs), len(candidate_xs)
    if n == 0 or m == 0:
        return {}
    scale = 8.0 * max(unit, 1.0)

    def match_cost(i: int, j: int) -> float:
        x = symbol_xs[i]
        if x is None:
            return 0.5
        return min(abs(x - candidate_xs[j]) / scale, _MAX_MATCH_COST)

    cost = np.zeros((n + 1, m + 1))
    cost[:, 0] = np.arange(n + 1) * _GAP_COST
    cost[0, :] = np.arange(m + 1) * _GAP_COST
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + match_cost(i - 1, j - 1),
                cost[i - 1, j] + _GAP_COST,
                cost[i, j - 1] + _GAP_COST,
            )
    result: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        if np.isclose(cost[i, j], cost[i - 1, j - 1] + match_cost(i - 1, j - 1)):
            result[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif np.isclose(cost[i, j], cost[i - 1, j] + _GAP_COST):
            i -= 1
        else:
            j -= 1
    return result


def _is_pitched_note(symbol: EncodedSymbol) -> bool:
    return symbol.rhythm.startswith("note") and symbol.pitch not in (nonote, empty)


def _is_lower(symbol: EncodedSymbol) -> bool:
    return symbol.position.startswith("lower")


class OdysseyContext:
    """Collects staff geometry and snaps symbol positions while homr parses a page."""

    def __init__(self, bar_lines: list[RotatedBoundingBox], image: NDArray | None = None) -> None:
        """`image` is the page as the user supplied it, grayscale; it is used to find rests."""
        self.bar_lines = bar_lines
        self.geometry: list[StaffGeometry] = []
        self.ink = image < 128 if image is not None else None

    def annotate(
        self,
        symbols: list[EncodedSymbol],
        staff: Staff,
        system: int,
        part: int,
        to_input: PointMapping,
    ) -> None:
        halves: list[Staff] = [staff]
        if staff.is_grandstaff:
            split = staff.split()
            if split is not None:
                halves = list(split)
        for half_no, half in enumerate(halves):
            half_symbols = [
                s for s in symbols if len(halves) == 1 or _is_lower(s) == (half_no == 1)
            ]
            geometry = self._geometry(half, system, part, half_no, to_input)
            self._snap_notes(half_symbols, half, geometry, to_input)
            self._snap_bars(half_symbols, geometry)
            self._snap_rests(half_symbols, geometry)
            if self.ink is not None:
                geometry.hairpins = find_hairpins(self.ink, geometry)
            self.geometry.append(geometry)
        for symbol in symbols:
            symbol.odyssey_system = system
            symbol.odyssey_part = part

    def _geometry(
        self, staff: Staff, system: int, part: int, half: int, to_input: PointMapping
    ) -> StaffGeometry:
        lines = len(staff.grid[0].y)
        mapped = [
            [to_input((p.x, p.y[i])) for p in staff.grid] for i in range(lines)
        ]  # per line: points
        line_ys = [float(np.mean([pt[1] for pt in line])) for line in mapped]
        xs = [pt[0] for pt in mapped[0]]
        unit = float(np.mean(np.diff(line_ys)))
        x_min, x_max = float(min(xs)), float(max(xs))
        top, bottom = line_ys[0] - 2 * unit, line_ys[-1] + 2 * unit
        bars = []
        for box in self.bar_lines:
            x, y = to_input(box.center)
            if top <= y <= bottom and x_min - unit <= x <= x_max + unit:
                bars.append(float(x))
        bars.sort()
        deduped: list[float] = []
        for x in bars:
            if not deduped or x - deduped[-1] > 0.75 * unit:
                deduped.append(x)
        return StaffGeometry(system, part, half, line_ys, x_min, x_max, unit, deduped)

    def _snap_notes(
        self,
        symbols: list[EncodedSymbol],
        staff: Staff,
        geometry: StaffGeometry,
        to_input: PointMapping,
    ) -> None:
        notes = [s for s in symbols if _is_pitched_note(s)]
        candidates = sorted(to_input(n.center) for n in staff.get_notes())
        xs = [s.image_coordinates[0] if s.image_coordinates else None for s in notes]
        matches = align(xs, [c[0] for c in candidates], geometry.unit)
        for i, j in matches.items():
            notes[i].image_coordinates = (float(candidates[j][0]), float(candidates[j][1]))
            notes[i].snapped = True
        # a note with no notehead found where it should be is more likely a misreading
        for note in notes:
            if not note.snapped and note.confidence is not None:
                note.confidence *= UNMATCHED_PENALTY

    def _snap_bars(self, symbols: list[EncodedSymbol], geometry: StaffGeometry) -> None:
        bars = [s for s in symbols if "barline" in s.rhythm]
        y = geometry.line_ys[2]
        xs = [s.image_coordinates[0] if s.image_coordinates else None for s in bars]
        matches = align(xs, geometry.bars, geometry.unit)
        for i, j in matches.items():
            bars[i].image_coordinates = (geometry.bars[j], y)
            bars[i].snapped = True

    def _snap_rests(self, symbols: list[EncodedSymbol], geometry: StaffGeometry) -> None:
        """
        Rests have no detector, so look for them in the scan: between the rest's neighbours
        that have a position, remove the staff lines, mask the neighbouring notes (with their
        accidentals, stems, flags and dots) and take the remaining blob of ink.
        """
        if self.ink is None:
            return
        u = geometry.unit
        top = int(geometry.line_ys[0] - u)
        bottom = int(geometry.line_ys[-1] + u) + 1
        band = _without_staff_lines(self.ink, geometry, top, bottom)
        width = band.shape[1]

        known = [
            (i, s) for i, s in enumerate(symbols)
            if s.snapped and s.image_coordinates is not None
        ]
        excluded = np.zeros(width, dtype=bool)
        for _, s in known:
            x = s.image_coordinates[0]  # type: ignore[index]
            if "barline" in s.rhythm:
                lo, hi = x - 0.4 * u, x + 0.4 * u
            else:
                has_accidental = s.lift not in (nonote, empty)
                lo = x - (2.8 if has_accidental else 1.1) * u
                hi = x + 1.9 * u
            excluded[max(int(lo), 0) : max(int(hi) + 1, 0)] = True

        rests = [i for i, s in enumerate(symbols) if s.rhythm.startswith("rest") and not s.rhythm.endswith("m")]
        # group consecutive rests that lie between the same two positioned symbols
        groups: dict[tuple[int, int], list[int]] = {}
        for i in rests:
            before = max((k for k, _ in known if k < i), default=-1)
            after = min((k for k, _ in known if k > i), default=-1)
            groups.setdefault((before, after), []).append(i)

        for (before, after), members in groups.items():
            left = symbols[before].image_coordinates[0] if before >= 0 else geometry.x_min + 4 * u  # type: ignore[index]
            right = symbols[after].image_coordinates[0] if after >= 0 else geometry.x_max  # type: ignore[index]
            lo, hi = int(max(left, geometry.x_min)), int(min(right, geometry.x_max))
            if hi - lo < u:
                continue
            columns = band[:, lo:hi].sum(axis=0).astype(float)
            columns[excluded[lo:hi]] = 0
            blobs = _blobs(columns, min_height=0.3 * u, max_width=3.2 * u, gap=int(0.3 * u) + 1)
            blobs = sorted(blobs, key=lambda b: -b[1])[: len(members)]
            blobs.sort(key=lambda b: b[0])
            if len(blobs) != len(members):
                continue
            y = geometry.line_ys[2]
            for i, (center, _mass) in zip(members, blobs, strict=True):
                symbols[i].image_coordinates = (float(lo + center), float(y))
                symbols[i].snapped = True


def _without_staff_lines(ink: NDArray, geometry: StaffGeometry, top: int, bottom: int) -> NDArray:
    """The staff band with the staff lines removed, except where something crosses them."""
    x0, x1 = 0, ink.shape[1]
    band = ink[max(top, 0) : bottom, x0:x1].copy()
    thickness = max(1, int(round(0.12 * geometry.unit)))
    for y in geometry.line_ys:
        row = int(round(y)) - max(top, 0)
        lo, hi = max(row - thickness, 0), min(row + thickness + 1, band.shape[0])
        above = band[max(lo - 2, 0)]
        below = band[min(hi + 1, band.shape[0] - 1)]
        band[lo:hi] &= above | below
    return band


def _blobs(columns: NDArray, min_height: float, max_width: float, gap: int) -> list[tuple[float, float]]:
    """Runs of inked columns as (centre, mass); runs wider than max_width (beams) are dropped."""
    on = columns >= min_height
    result: list[tuple[float, float]] = []
    start, last = None, None
    for x, flag in enumerate(list(on) + [False] * (gap + 1)):
        if flag:
            if start is None:
                start = x
            last = x
        elif start is not None and last is not None and x - last > gap:
            if last - start + 1 <= max_width:
                xs = np.arange(start, last + 1)
                weights = columns[start : last + 1]
                if weights.sum() > 0:
                    result.append((float((xs * weights).sum() / weights.sum()), float(weights.sum())))
            start, last = None, None
    return result


def find_hairpins(ink: NDArray, geometry: StaffGeometry) -> list[dict]:
    """
    Hairpins (crescendo/diminuendo wedges) below a staff. Homr's model doesn't read them, so
    find them in the scan: two thin straight lines that overlap horizontally, slope in opposite
    directions, and are close together at one end and apart at the other. A hairpin cut by a
    system break is open at both ends; `open_start`/`open_end` say which end is not closed.
    """
    u = geometry.unit
    top = int(geometry.line_ys[-1] + 0.35 * u)
    bottom = min(int(geometry.line_ys[-1] + 7 * u), ink.shape[0])
    x0, x1 = int(max(geometry.x_min - u, 0)), int(min(geometry.x_max + u, ink.shape[1]))
    if bottom - top < 2 * u or x1 - x0 < 6 * u:
        return []
    region = ink[top:bottom, x0:x1]
    found = cv2.HoughLinesP(
        region.astype(np.uint8) * 255, 1, np.pi / 720, threshold=int(2 * u),
        minLineLength=int(4 * u), maxLineGap=int(0.6 * u),
    )
    if found is None:
        return []
    lines = []
    for ax, ay, bx, by in np.asarray(found).reshape(-1, 4):
        if bx < ax:
            ax, ay, bx, by = bx, by, ax, ay
        slope = (by - ay) / max(bx - ax, 1)
        if not (0.012 < abs(slope) < 0.27) or not _is_thin(region, ax, ay, bx, by, 0.4 * u):
            continue
        lines.append((float(ax), float(ay), float(bx), float(by), float(slope)))
    lines = _merge_collinear(lines, u)

    result = []
    used: set[int] = set()
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if i >= j or i in used or j in used or a[4] * b[4] >= 0:
                continue
            left, right = max(a[0], b[0]), min(a[2], b[2])
            if right - left < 0.7 * min(a[2] - a[0], b[2] - b[0]) or right - left < 4 * u:
                continue
            gap_left = abs(_y_at(a, left) - _y_at(b, left))
            gap_right = abs(_y_at(a, right) - _y_at(b, right))
            if abs(gap_right - gap_left) < 0.5 * u or max(gap_left, gap_right) > 3.5 * u:
                continue
            closed = 0.45 * u
            kind = "crescendo" if gap_left < gap_right else "diminuendo"
            result.append({
                "type": kind,
                "x0": left + x0,
                "x1": right + x0,
                "y": (_y_at(a, left) + _y_at(b, left)) / 2 + top,
                "open_start": bool(kind == "diminuendo" or gap_left > closed),
                "open_end": bool(kind == "crescendo" or gap_right > closed),
            })
            used.update((i, j))
    return sorted(result, key=lambda h: h["x0"])


def _y_at(line: tuple[float, ...], x: float) -> float:
    return line[1] + (x - line[0]) * line[4]


def _is_thin(region: NDArray, ax: int, ay: int, bx: int, by: int, max_thickness: float) -> bool:
    """Beams and other thick strokes are not hairpin lines."""
    for t in (0.2, 0.4, 0.6, 0.8):
        x = int(ax + (bx - ax) * t)
        y = int(ay + (by - ay) * t)
        column = region[:, x]
        top = y
        while top > 0 and column[top - 1]:
            top -= 1
        bottom = y
        while bottom < len(column) - 1 and column[bottom + 1]:
            bottom += 1
        if bottom - top + 1 > max_thickness:
            return False
    return True


def _merge_collinear(lines: list[tuple[float, ...]], u: float) -> list[tuple[float, ...]]:
    """Hough finds a line several times (and in pieces); keep one segment per line."""
    merged: list[list[float]] = []
    for line in sorted(lines, key=lambda line: line[0]):
        for m in merged:
            same_slope = abs(m[4] - line[4]) < 0.02
            touching = line[0] <= m[2] + u
            if same_slope and touching and abs(_y_at(tuple(m), line[0]) - line[1]) < 0.3 * u:
                if line[2] > m[2]:
                    m[2], m[3] = line[2], line[3]
                m[4] = (m[3] - m[1]) / max(m[2] - m[0], 1)
                break
        else:
            merged.append(list(line))
    return [tuple(m) for m in merged]
