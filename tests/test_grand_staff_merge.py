from homr.model import Staff, StaffPoint
from homr.staff_relations import _merge_grand_staff


def _staff(xs: list[float], top: float) -> Staff:
    return Staff([StaffPoint(x, [top + 10 * i for i in range(5)], 0.0) for x in xs])


def test_grand_staff_keeps_every_grid_point_where_both_staffs_reach() -> None:
    # Odyssey: broken staff lines on a scan give the two staffs few common x positions; the
    # merged grid must not have gaps there, or notes in the gaps are lost.
    upper = _staff([0, 10, 20, 30, 40, 50, 300], 100)
    lower = _staff([0, 150, 250, 300, 310], 200)
    merged = _merge_grand_staff(upper, lower)
    assert [p.x for p in merged.grid] == [0, 10, 20, 30, 40, 50, 150, 250, 300]
    assert all(len(p.y) == 10 for p in merged.grid)  # noqa: PLR2004
    assert merged.grid[6].y == [100, 110, 120, 130, 140, 200, 210, 220, 230, 240]
    assert merged.is_grandstaff


def test_grand_staff_without_overlap_is_the_upper_staff() -> None:
    upper = _staff([0, 10], 100)
    lower = _staff([50, 60], 200)
    assert _merge_grand_staff(upper, lower) is upper
