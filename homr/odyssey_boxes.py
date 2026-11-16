"""Odyssey: pixel bounding boxes of the detections symbols are snapped to.

A box is (x, y, width, height): axis-aligned, top-left corner, in the pixels of the image the
user supplied. This module is the only place that knows how a box is made and written, so the
transport (today a `box=x,y,w,h` field in the MusicXML comments, see music_xml_generator.py)
can change without touching the detection code.
"""

from collections.abc import Callable, Iterable

import cv2
import numpy as np

from homr.bounding_boxes import AngledBoundingBox, RotatedBoundingBox

Point = tuple[float, float]
Box = tuple[float, float, float, float]


def box_of_points(points: Iterable[Point]) -> Box:
    """The axis-aligned box around some points."""
    array = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    x0, y0 = array.min(axis=0)
    x1, y1 = array.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def box_of_detection(detection: AngledBoundingBox, to_input: Callable[[Point], Point]) -> Box:
    """
    The box of a detected notehead (ellipse) or barline (rotated rectangle), found on the page
    that homr works on, mapped through `to_input` into the pixels of the input image.
    """
    if isinstance(detection, RotatedBoundingBox):
        corners = cv2.boxPoints(detection.box)
    else:
        corners = np.asarray(detection.polygon, dtype=np.float64)
    return box_of_points(to_input((float(x), float(y))) for x, y in corners.reshape(-1, 2))


def union(boxes: Iterable[Box]) -> Box | None:
    boxes = list(boxes)
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def box_to_json(box: Box) -> dict[str, float]:
    x, y, w, h = (round(v, 1) for v in box)
    return {"x": x, "y": y, "w": w, "h": h}


def box_field(box: Box | None) -> str:
    """The text a box is written as inside an Odyssey comment (leading space), or nothing."""
    if box is None:
        return ""
    return " box=" + ",".join(f"{v:.1f}" for v in box)
