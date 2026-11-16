"""Odyssey additions: snap the transformer's imprecise symbol positions to detections.

The transformer reports where each symbol is from its attention, which can be off by
several note widths and isn't monotonic. The segmentation stage finds noteheads and
barlines exactly, but doesn't know which token they belong to. Both sequences run left to
right, so we align them (edit distance) and take the detected position for matched symbols.
"""

from dataclasses import dataclass, field

import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.model import Staff
from homr.point_mapping import PointMapping
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote

_GAP_COST = 1.0
_MAX_MATCH_COST = 1.5


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

    def __init__(self, bar_lines: list[RotatedBoundingBox]) -> None:
        self.bar_lines = bar_lines
        self.geometry: list[StaffGeometry] = []

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

    def _snap_bars(self, symbols: list[EncodedSymbol], geometry: StaffGeometry) -> None:
        bars = [s for s in symbols if "barline" in s.rhythm]
        y = geometry.line_ys[2]
        xs = [s.image_coordinates[0] if s.image_coordinates else None for s in bars]
        matches = align(xs, geometry.bars, geometry.unit)
        for i, j in matches.items():
            bars[i].image_coordinates = (geometry.bars[j], y)
            bars[i].snapped = True
