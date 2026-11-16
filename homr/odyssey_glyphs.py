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
        unit: float | None = None,
    ) -> None:
        """
        `gray`: the page, grayscale. `masks`: Homr's segmentation masks by name ("notehead",
        "stems_rest", "clefs_keys", "staff") as predicted, i.e. in the resized/cropped frame;
        `origin` and `step` place that frame in the page (the page point of mask pixel (0, 0) and
        the page distance of one mask pixel). `unit`: the staff space in pixels, if known; it sets the
        size of the structuring elements (taken from the first staff otherwise).
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
        self._unit = unit

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
