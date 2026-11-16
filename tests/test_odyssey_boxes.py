import unittest

import numpy as np

from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.odyssey_align import StaffGeometry, _blobs, _ink_box
from homr.odyssey_boxes import box_field, box_of_detection, box_of_points, box_to_json, union

NO_CONTOURS = np.zeros((0, 1, 2), dtype=np.int32)


class TestOdysseyBoxes(unittest.TestCase):
    def test_box_of_points(self) -> None:
        self.assertEqual(box_of_points([(10, 20), (14, 31), (12, 25)]), (10.0, 20.0, 4.0, 11.0))

    def test_barline_box_is_mapped_to_the_input_image(self) -> None:
        bar = RotatedBoundingBox(((100.0, 50.0), (4.0, 80.0), 0.0), NO_CONTOURS)
        # the input image is twice as large and shifted by 5 pixels
        box = box_of_detection(bar, lambda p: (p[0] * 2 + 5, p[1] * 2 + 5))
        self.assertEqual(box, (201.0, 25.0, 8.0, 160.0))

    def test_notehead_box_surrounds_the_ellipse(self) -> None:
        head = BoundingEllipse(((200.0, 100.0), (24.0, 18.0), 0.0), NO_CONTOURS)
        x, y, w, h = box_of_detection(head, lambda p: p)
        self.assertAlmostEqual(x + w / 2, 200.0, delta=1)
        self.assertAlmostEqual(y + h / 2, 100.0, delta=1)
        self.assertAlmostEqual(w, 24.0, delta=1.5)
        self.assertAlmostEqual(h, 18.0, delta=1.5)

    def test_union_and_text_form(self) -> None:
        self.assertEqual(union([(0, 0, 10, 10), (5, 20, 10, 10)]), (0, 0, 15, 30))
        self.assertIsNone(union([]))
        self.assertEqual(box_field((1.04, 2.0, 3.0, 4.0)), " box=1.0,2.0,3.0,4.0")
        self.assertEqual(box_field(None), "")
        expected = {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
        self.assertEqual(box_to_json((1.04, 2.0, 3.0, 4.0)), expected)

    def test_staff_box_spans_the_sloping_lines(self) -> None:
        lines = [100, 110, 120, 130, 140]
        geometry = StaffGeometry(0, 0, 0, lines, 10, 510, 10, y_top=96, y_bottom=146)
        self.assertEqual(geometry.box, (10, 96, 500, 50))
        self.assertEqual(geometry.to_json()["bbox"], {"x": 10.0, "y": 96.0, "w": 500.0, "h": 50.0})
        flat = StaffGeometry(0, 0, 0, lines, 10, 510, 10)
        self.assertEqual(flat.box, (10, 100, 500, 40))

    def test_rest_box_is_the_ink_of_the_blob(self) -> None:
        band = np.zeros((60, 100), dtype=bool)
        band[20:40, 30:38] = True  # the rest
        band[5, 30:38] = True  # a trace of ink above (a slur crossing it) is not part of the rest
        columns = band.sum(axis=0).astype(float)
        (_centre, _mass, first, last), = _blobs(columns, min_height=3, max_width=30, gap=2)
        self.assertEqual(_ink_box(band, 200, first, last + 1), (30.0, 220.0, 8.0, 20.0))
