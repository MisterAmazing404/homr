import json
import os
import tempfile
import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.odyssey import IGNORE, _group_by_assignment, _split
from homr.point_mapping import chain, invert, undo_crop, undo_resize
from homr.staff_position_save_load import (
    is_precise_staff_positions,
    load_precise_staff_positions,
    load_staff_positions,
    save_staff_positions,
)


def _staff(top: float, x_end: int = 200, lines: int = 5) -> Staff:
    unit = 10.0
    return Staff(
        [
            StaffPoint(float(x), [top + unit * i for i in range(lines)], 0.01)
            for x in range(0, x_end, 10)
        ]
    )


def _systems() -> list[MultiStaff]:
    grand = _staff(100).merge(_staff(200))  # a piano system, then a violin system
    return [MultiStaff([grand], [], certain=True), MultiStaff([_staff(400)], [])]


class TestStaffPositionSaveLoad(unittest.TestCase):
    def _round_trip(self, systems: list[MultiStaff]) -> list[MultiStaff]:
        # crop at (30, 20), then resize 600x400 -> 300x200 (the pipeline's way to the working image)
        to_input = chain(undo_resize((400, 600), (200, 300)), undo_crop(30, 20))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "page.staves.json")
            save_staff_positions(systems, path, to_input, (500, 700, 3))
            self.assertTrue(is_precise_staff_positions(path))
            with open(path) as file:
                saved = json.load(file)
            self.assertEqual(saved["image"], {"width": 700, "height": 500})
            return load_precise_staff_positions(path, invert(to_input), (500, 700, 3))

    def test_staffs_come_back_as_they_were_saved(self) -> None:
        systems = _systems()
        loaded = self._round_trip(systems)

        self.assertEqual([len(s.staffs) for s in loaded], [1, 1])
        self.assertEqual([s.certain for s in loaded], [True, False])
        self.assertEqual([s.staffs[0].is_grandstaff for s in loaded], [True, False])
        for saved, back in zip(systems, loaded, strict=True):
            for a, b in zip(saved.staffs, back.staffs, strict=True):
                self.assertEqual(len(a.grid), len(b.grid))
                for p, q in zip(a.grid, b.grid, strict=True):
                    self.assertAlmostEqual(p.x, q.x, places=9)
                    self.assertEqual(len(p.y), len(q.y))
                    for y, z in zip(p.y, q.y, strict=True):
                        self.assertAlmostEqual(y, z, places=9)

    def test_load_staff_positions_picks_the_format_from_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "page.staves.json")
            save_staff_positions(_systems(), path, input_shape=(500, 700, 3))
            loaded = load_staff_positions(None, None, path, input_shape=(500, 700, 3))  # type: ignore[arg-type]
            self.assertEqual(len(loaded), 2)

    def test_staffs_saved_for_another_image_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "page.staves.json")
            save_staff_positions(_systems(), path, input_shape=(500, 700, 3))
            with self.assertRaises(ValueError):
                load_precise_staff_positions(path, input_shape=(501, 700, 3))


class TestGroupByAssignment(unittest.TestCase):
    def test_split_lists_a_grand_staff_as_two_staffs(self) -> None:
        grand, single = _systems()
        self.assertEqual([len(g) for g in _split(grand)], [2])
        self.assertEqual([len(g) for g in _split(single)], [1])

    def test_staffs_are_grouped_by_their_place_in_the_assignment(self) -> None:
        # first system: a piano staff (two halves) and a lone staff; second: two single staffs
        first = [_staff(100), _staff(200), _staff(300)]
        second = [_staff(500), _staff(600)]
        systems = [[(s,) for s in first], [(s,) for s in second]]

        rows, slots = _group_by_assignment(systems, [[0, 0, 1], [IGNORE, 1]])

        self.assertEqual(slots, [(0, 1), (1,)])
        self.assertTrue(rows[0].staffs[0].is_grandstaff)  # the two staffs of part 0 were joined
        self.assertEqual(len(rows[0].staffs[0].grid[0].y), 10)
        self.assertEqual(rows[1].staffs[0].min_y, 500 + 100)  # the ignored one is left out

    def test_an_assignment_which_does_not_fit_is_refused(self) -> None:
        systems = [[(_staff(100),)]]
        with self.assertRaises(ValueError):
            _group_by_assignment(systems, [[0], [0]])
        with self.assertRaises(ValueError):
            _group_by_assignment(systems, [[0, 1]])
