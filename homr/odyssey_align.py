"""Odyssey additions: snap the transformer's imprecise symbol positions to detections.

The transformer reports where each symbol is from its attention, which can be off by
several note widths and isn't monotonic. The segmentation stage finds noteheads and
barlines exactly, but doesn't know which token they belong to. Both sequences run left to
right, so we align them (edit distance) and take the detected position for matched symbols.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Note, Staff
from homr.odyssey_boxes import Box, box_of_detection, box_to_json
from homr.odyssey_glyphs import PageInk, find_rest_candidates, rest_fits, rest_kind
from homr.odyssey_glyphs import find_hairpins as find_page_hairpins
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
    # the detected box of each barline in `bars` (same order), as (x, y, width, height) pixels
    bar_boxes: list[Box] = field(default_factory=list)
    # the staff's extent in y: the top line's highest and the bottom line's lowest point (the
    # lines can slope on a scan); None falls back to the average line positions
    y_top: float | None = None
    y_bottom: float | None = None
    # hairpins below the staff: {"type": "crescendo" | "diminuendo", "x0", "x1", "y", "open_start", "open_end"}
    hairpins: list[dict] = field(default_factory=list)
    id: str = ""  # "<system>.<part>.<half>", the prefix of the ids of the symbols on this staff

    @property
    def box(self) -> Box:
        """The staff: from the first to the last line, over its whole width."""
        top = self.line_ys[0] if self.y_top is None else self.y_top
        bottom = self.line_ys[-1] if self.y_bottom is None else self.y_bottom
        return self.x_min, top, self.x_max - self.x_min, bottom - top

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "system": self.system,
            "part": self.part,
            "half": self.half,
            "line_ys": self.line_ys,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "unit": self.unit,
            "bars": self.bars,
            "bbox": box_to_json(self.box),
            "bar_boxes": [box_to_json(b) for b in self.bar_boxes],
            "hairpins": self.hairpins,
        }


def align(
    symbol_xs: list[float | None],
    candidate_xs: list[float],
    unit: float,
    allowed: Callable[[int, int], bool] | None = None,
) -> dict[int, int]:
    """
    Minimum cost monotonic matching of symbols to detections.
    Returns {symbol index: candidate index} for the matched pairs. `allowed(i, j)` rules out
    pairs that can't belong together (a rest token and a piece of ink that isn't its shape).
    """
    n, m = len(symbol_xs), len(candidate_xs)
    if n == 0 or m == 0:
        return {}
    scale = 8.0 * max(unit, 1.0)

    def match_cost(i: int, j: int) -> float:
        if allowed is not None and not allowed(i, j):
            return 2 * _GAP_COST + 1  # never better than leaving both unmatched
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

    def __init__(
        self,
        bar_lines: list[RotatedBoundingBox],
        image: NDArray | None = None,
        page: PageInk | None = None,
    ) -> None:
        """
        `image` is the page as the user supplied it, grayscale. `page` is that image with Homr's
        segmentation taken away (odyssey_glyphs); rests and hairpins are found in it. Without it
        the rests and hairpins are found by the older image-only methods, which need `image`.
        """
        self.bar_lines = bar_lines
        self.geometry: list[StaffGeometry] = []
        # every symbol read from the page by its payload id, for homr.odyssey_payload
        self.symbols: dict[str, EncodedSymbol] = {}
        self.page = page
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
            if self.page is not None:
                geometry.hairpins = find_page_hairpins(
                    self.page, geometry.line_ys, geometry.unit, geometry.x_min, geometry.x_max
                )
            elif self.ink is not None:
                geometry.hairpins = find_hairpins(self.ink, geometry)
            geometry.id = f"{system}.{part}.{half_no}"
            self.geometry.append(geometry)
            for index, symbol in enumerate(half_symbols):
                symbol.odyssey_half = half_no
                symbol.odyssey_id = f"{geometry.id}.{index}"
                self.symbols[symbol.odyssey_id] = symbol
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
        bars: list[tuple[float, Box]] = []
        for detection in self.bar_lines:
            x, y = to_input(detection.center)
            if top <= y <= bottom and x_min - unit <= x <= x_max + unit:
                bars.append((float(x), box_of_detection(detection, to_input)))
        bars.sort(key=lambda bar: bar[0])
        deduped: list[tuple[float, Box]] = []
        for bar in bars:
            if not deduped or bar[0] - deduped[-1][0] > 0.75 * unit:
                deduped.append(bar)
        return StaffGeometry(
            system, part, half, line_ys, x_min, x_max, unit,
            [x for x, _ in deduped], [box for _, box in deduped],
            y_top=float(min(pt[1] for pt in mapped[0])), y_bottom=float(max(pt[1] for pt in mapped[-1])),
        )

    def _snap_notes(
        self,
        symbols: list[EncodedSymbol],
        staff: Staff,
        geometry: StaffGeometry,
        to_input: PointMapping,
    ) -> None:
        notes = [s for s in symbols if _is_pitched_note(s)]
        heads: list[Note] = sorted(staff.get_notes(), key=lambda n: to_input(n.center))
        candidates = [to_input(n.center) for n in heads]
        xs = [s.image_coordinates[0] if s.image_coordinates else None for s in notes]
        matches = align(xs, [c[0] for c in candidates], geometry.unit)
        for i, j in matches.items():
            box = box_of_detection(heads[j].box, to_input)
            notes[i].odyssey_box = box
            notes[i].image_coordinates = (box[0] + box[2] / 2, box[1] + box[3] / 2)
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
            bars[i].odyssey_box = geometry.bar_boxes[j]
            bars[i].snapped = True

    def _snap_rests(self, symbols: list[EncodedSymbol], geometry: StaffGeometry) -> None:
        """
        Homr's segmentation has no rest class, but the transformer reads every rest and says
        roughly where it is. The page with everything Homr recognised taken away (`PageInk`) leaves
        the rests standing alone in the staff; each rest token is matched to one of them with the
        same monotonic alignment as the noteheads (shape and position decide), between the
        symbols already snapped on either side of it. A rest that finds no match keeps the
        transformer's position and stays unsnapped: that is where Homr gives us no better signal.
        """
        if self.page is None:
            self._snap_rests_by_blobs(symbols, geometry)
            return
        u = geometry.unit
        candidates = find_rest_candidates(
            self.page, geometry.line_ys, u, geometry.x_min, geometry.x_max
        )
        # time signature numerals aren't in Homr's masks; the transformer says where they are
        header = [
            s.image_coordinates[0]
            for s in symbols
            if s.rhythm.startswith("timeSignature") and s.image_coordinates
        ]
        candidates = [c for c in candidates if all(abs(c.x - x) > 2.5 * u for x in header)]
        known = [
            (i, s) for i, s in enumerate(symbols) if s.snapped and s.image_coordinates is not None
        ]
        rests = [
            i for i, s in enumerate(symbols) if s.rhythm.startswith("rest") and rest_kind(s.rhythm)
        ]
        groups: dict[tuple[int, int], list[int]] = {}
        for i in rests:
            before = max((k for k, _ in known if k < i), default=-1)
            after = min((k for k, _ in known if k > i), default=-1)
            groups.setdefault((before, after), []).append(i)
        for (before, after), members in groups.items():
            left = symbols[before].image_coordinates[0] if before >= 0 else geometry.x_min  # type: ignore[index]
            right = symbols[after].image_coordinates[0] if after >= 0 else geometry.x_max  # type: ignore[index]
            between = [c for c in candidates if left + 0.25 * u < c.x < right - 0.25 * u]
            if not between:
                continue
            xs = [
                symbols[i].image_coordinates[0] if symbols[i].image_coordinates else None
                for i in members
            ]

            def fits(m: int, c: int, members=members, xs=xs, between=between) -> bool:  # noqa: ANN001
                kind = rest_kind(symbols[members[m]].rhythm)
                return kind is not None and rest_fits(kind, between[c].w / u, between[c].h / u)

            for m, c in align(xs, [c.x for c in between], u, fits).items():
                cand = between[c]
                symbols[members[m]].image_coordinates = (float(cand.x), float(geometry.line_ys[2]))
                symbols[members[m]].odyssey_box = (
                    float(cand.x - cand.w / 2),
                    float(cand.y - cand.h / 2),
                    float(cand.w),
                    float(cand.h),
                )
                symbols[members[m]].snapped = True
        # A rest that touches a staff line or a stem is part of a bigger piece of ink and no
        # candidate: look for it between its neighbours the older way before giving up on it.
        if any(not symbols[i].snapped for i in rests):
            self._snap_rests_by_blobs(symbols, geometry, only=[i for i in rests if not symbols[i].snapped])

    def _snap_rests_by_blobs(
        self, symbols: list[EncodedSymbol], geometry: StaffGeometry, only: list[int] | None = None
    ) -> None:
        """
        The older method, kept for use without a `PageInk`. Rests have no detector, so look for them in the scan: between the rest's neighbours
        that have a position, remove the staff lines, mask the neighbouring notes (with their
        accidentals, stems, flags and dots) and take the remaining blob of ink.
        """
        ink = self.page.ink if self.page is not None else self.ink
        if ink is None:
            return
        u = geometry.unit
        top = int(geometry.line_ys[0] - u)
        bottom = int(geometry.line_ys[-1] + u) + 1
        band = _without_staff_lines(ink, geometry, top, bottom)
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

        rests = [
            i
            for i, s in enumerate(symbols)
            if s.rhythm.startswith("rest") and not s.rhythm.endswith("m") and (only is None or i in only)
        ]
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
            for i, (center, _mass, first, last) in zip(members, blobs, strict=True):
                symbols[i].image_coordinates = (float(lo + center), float(y))
                symbols[i].odyssey_box = _ink_box(band, max(top, 0), lo + first, lo + last + 1)
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


def _ink_box(band: NDArray, band_top: int, x0: int, x1: int) -> Box:
    """
    The box of the ink in columns x0..x1 of the staff band (which starts at row `band_top`).
    Of the separate runs of inked rows (gaps of up to 2 rows are bridged) only the heaviest
    counts, so a slur or a ledger line passing close to the rest isn't part of its box.
    """
    rows = band[:, x0:x1].sum(axis=1)
    inked = np.nonzero(rows)[0]
    if len(inked) == 0:
        return float(x0), float(band_top), float(x1 - x0), float(band.shape[0])
    breaks = np.nonzero(np.diff(inked) > 3)[0] + 1  # noqa: PLR2004
    runs = np.split(inked, breaks)
    best = max(runs, key=lambda run: rows[run].sum())
    return float(x0), float(band_top + best[0]), float(x1 - x0), float(best[-1] - best[0] + 1)


def _blobs(
    columns: NDArray, min_height: float, max_width: float, gap: int
) -> list[tuple[float, float, int, int]]:
    """
    Runs of inked columns as (centre, mass, first column, last column); runs wider than
    max_width (beams) are dropped.
    """
    on = columns >= min_height
    result: list[tuple[float, float, int, int]] = []
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
                    result.append(
                        (float((xs * weights).sum() / weights.sum()), float(weights.sum()), start, last)
                    )
            start, last = None, None
    return result


def find_hairpins(ink: NDArray, geometry: StaffGeometry) -> list[dict]:
    """
    Hairpins (crescendo/diminuendo wedges) below a staff. Homr's model doesn't read them, so
    find them in the scan. Keep the thin strokes (no stems, noteheads, beams), bridge the gaps
    stems cut into them, and fit straight lines to the top and bottom edge of each stroke. A
    wedge drawn in one stroke shows two edges that meet; a wedge whose ends don't touch (cut by
    a system break) shows as two straight lines with opposite slopes, which are paired.
    `open_start`/`open_end` say which end is not closed.
    """
    u = geometry.unit
    top = int(geometry.line_ys[-1] + 0.35 * u)
    bottom = min(int(geometry.line_ys[-1] + 7 * u), ink.shape[0])
    x0, x1 = int(max(geometry.x_min - u, 0)), int(min(geometry.x_max + u, ink.shape[1]))
    if bottom - top < 2 * u or x1 - x0 < 6 * u:
        return []
    thin = _thin_strokes(ink[top:bottom, x0:x1], max_run=max(2, int(0.35 * u)))
    bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.4 * u)), 1))
    thin = cv2.morphologyEx(thin.astype(np.uint8), cv2.MORPH_CLOSE, bridge)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)

    wedges: list[dict] = []
    lines: list[tuple[float, ...]] = []
    for c in range(1, count):
        cx, cy, w, h, _area = stats[c]
        if w < 4 * u or h > 3.5 * u:
            continue
        ys, xs = np.nonzero(labels[cy : cy + h, cx : cx + w] == c)
        columns = np.unique(xs)
        upper = np.array([ys[xs == x].min() for x in columns], dtype=float)
        lower = np.array([ys[xs == x].max() for x in columns], dtype=float)
        fit_up, res_up = _fit(columns, upper)
        fit_lo, res_lo = _fit(columns, lower)
        if res_up > 0.25 * u or res_lo > 0.25 * u:
            continue  # curved: a slur or a tie
        left, right = float(columns[0]), float(columns[-1])
        gap_left = np.polyval(fit_lo, left) - np.polyval(fit_up, left)
        gap_right = np.polyval(fit_lo, right) - np.polyval(fit_up, right)
        if max(gap_left, gap_right) > 0.5 * u:
            if fit_up[0] * fit_lo[0] < 0 and abs(gap_right - gap_left) > 0.3 * u:
                wedges.append(_wedge(left + cx + x0, right + cx + x0, gap_left, gap_right,
                                     (np.polyval(fit_up, left) + np.polyval(fit_lo, left)) / 2 + cy + top, u))
            continue
        slope = float((fit_up[0] + fit_lo[0]) / 2)
        if abs(slope) < 0.004 or abs(slope) > 0.27:
            continue  # level: a ledger or staff line
        y_left = float((np.polyval(fit_up, left) + np.polyval(fit_lo, left)) / 2)
        y_right = float((np.polyval(fit_up, right) + np.polyval(fit_lo, right)) / 2)
        lines.append((left + cx + x0, y_left + cy + top, right + cx + x0, y_right + cy + top, slope))

    used: set[int] = set()
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if i >= j or i in used or j in used or a[4] * b[4] >= 0:
                continue
            left, right = max(a[0], b[0]), min(a[2], b[2])
            shorter = min(a[2] - a[0], b[2] - b[0])
            if right - left < 0.8 * shorter or shorter < 0.7 * max(a[2] - a[0], b[2] - b[0]):
                continue
            gap_left = abs(_y_at(a, left) - _y_at(b, left))
            gap_right = abs(_y_at(a, right) - _y_at(b, right))
            if abs(gap_right - gap_left) < 0.25 * u or max(gap_left, gap_right) > 3.5 * u:
                continue
            wedges.append(_wedge(left, right, gap_left, gap_right, (_y_at(a, left) + _y_at(b, left)) / 2, u))
            used.update((i, j))
    return sorted(wedges, key=lambda h: h["x0"])


def _wedge(left: float, right: float, gap_left: float, gap_right: float, y_left: float, u: float) -> dict:
    kind = "crescendo" if gap_left < gap_right else "diminuendo"
    closed = 0.45 * u
    return {
        "type": kind,
        "x0": float(left),
        "x1": float(right),
        "y": float(y_left),
        "open_start": bool(kind == "diminuendo" or gap_left > closed),
        "open_end": bool(kind == "crescendo" or gap_right > closed),
    }


def _fit(xs: NDArray, ys: NDArray) -> tuple[NDArray, float]:
    """Least-squares line and its largest deviation."""
    fit = np.polyfit(xs, ys, 1)
    return fit, float(np.abs(np.polyval(fit, xs) - ys).max())


def _thin_strokes(ink: NDArray, max_run: int) -> NDArray:
    """The ink that is part of a vertical run no longer than `max_run` pixels."""
    result = np.zeros_like(ink, dtype=bool)
    height = ink.shape[0]
    padded = np.vstack([np.zeros((1, ink.shape[1]), dtype=bool), ink, np.zeros((1, ink.shape[1]), dtype=bool)])
    diff = np.diff(padded.astype(np.int8), axis=0)
    for x in range(ink.shape[1]):
        starts = np.where(diff[:, x] == 1)[0]
        ends = np.where(diff[:, x] == -1)[0]
        for s0, e0 in zip(starts, ends, strict=True):
            if e0 - s0 <= max_run:
                result[s0:min(e0, height), x] = True
    return result


def _y_at(line: tuple[float, ...], x: float) -> float:
    return line[1] + (x - line[0]) * line[4]
