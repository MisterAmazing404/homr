"""Odyssey entry point: run homr on one page and return MusicXML plus a payload.

The MusicXML is plain: it is the way the notation reaches music21, which builds the score
from it. Everything else Odyssey needs (positions, confidence, snapping, system and part,
staff lines, barlines, and where each symbol is in the MusicXML) is in the payload, a
JSON-serialisable dict described in homr.odyssey_payload.
"""

import xml.etree.ElementTree as ET

import cv2
from dataclasses import dataclass
from typing import Any

import numpy as np

from homr import main as homr_main
from homr.model import MultiStaff, Staff
from homr.music_xml_generator import TimeSignature, XmlGeneratorArguments, generate_xml
from homr.odyssey_align import OdysseyContext, StaffGeometry
from homr.odyssey_payload import build_payload
from homr.onnx_providers import coreml_available, cuda_available, rocm_available
from homr.note_detection import add_notes_to_staffs
from homr.point_mapping import invert
from homr.simple_logging import eprint
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_precise_staff_positions, save_staff_positions
from homr.system_repair import repair_systems
from homr.transformer.configs import Config


# Per system, the slot (part) of each of its staves from the top, as detect_layout lists them; -1 = a stave to ignore
PageAssignment = list[list[int]]
IGNORE = -1


@dataclass
class LayoutStaff:
    """One five-line staff as Odyssey's layout pass shows it, in the pixels of the input image."""

    line_ys: list[float]
    x_start: float
    x_end: float


def detect_layout(image_path: str, staves_path: str) -> list[list[LayoutStaff]]:
    """
    The initial read of a page: homr's own staff detection and grouping model find the staffs
    and the systems they form, top to bottom. They are saved to staves_path exactly (every
    staff grid, the grand staffs, the systems), and run_page reads the page from that file
    without detecting or grouping staffs again. Odyssey shows the staffs returned here for
    the user to assign parts to; a grand staff is returned as its two staffs.
    """
    config = _processing_config()
    single = homr_main.detect_single_staffs(image_path, config)
    systems = homr_main.group_staffs(single.predictions, single.staffs)
    systems = repair_systems(single.debug, systems)
    single.debug.clean_debug_files_from_previous_runs()
    systems = sorted(systems, key=lambda m: m.staffs[0].min_y)
    save_staff_positions(systems, staves_path, single.to_input_image, single.input_shape)
    return [
        [_layout_staff(staff, single.to_input_image) for group in _split(system) for staff in group]
        for system in systems
    ]


def _split(system: MultiStaff) -> list[tuple[Staff, ...]]:
    """
    The staffs of a system top to bottom, a grand staff as its upper and lower staff. This
    is the order of the staffs in detect_layout and in a page assignment.
    """
    result: list[tuple[Staff, ...]] = []
    for staff in system.staffs:
        halves: tuple[Staff, ...] = (staff,)
        if staff.is_grandstaff:
            split = staff.split()
            if split is not None:
                halves = split
        result.append(halves)
    return result


def _layout_staff(staff: Staff, to_input) -> LayoutStaff:  # noqa: ANN001
    lines = len(staff.grid[0].y)
    points = [[to_input((p.x, p.y[i])) for p in staff.grid] for i in range(lines)]
    xs = [x for x, _ in points[0]]
    return LayoutStaff(
        line_ys=[float(np.mean([y for _, y in line])) for line in points],
        x_start=float(min(xs)),
        x_end=float(max(xs)),
    )


@dataclass
class PageResult:
    musicxml: str
    payload: dict[str, Any]  # see homr.odyssey_payload
    geometry: list[StaffGeometry]
    last_times: list[TimeSignature | None]


def _processing_config() -> homr_main.ProcessingConfig:
    transformer_gpu = cuda_available() or rocm_available()
    segnet_gpu = transformer_gpu or coreml_available()
    return homr_main.ProcessingConfig(
        enable_debug=False,
        enable_cache=False,
        write_staff_positions=False,
        read_staff_positions=False,
        selected_staff=-1,
        transformer_use_gpu=transformer_gpu,
        segnet_use_gpu=segnet_gpu,
        coreml_encoder=False,
        title_detection=False,
    )


def init_models() -> None:
    """Download the model weights if they are missing."""
    config = _processing_config()
    homr_main.download_weights(config.segnet_use_gpu, config.transformer_use_gpu, False)


def run_page(
    image_path: str,
    staves_path: str,
    inherited_times: list[TimeSignature | None] | None = None,
    assignment: PageAssignment | None = None,
    slot_is_grandstaff: list[bool] | None = None,
) -> PageResult:
    """
    The full read of one page, on the staffs detect_layout saved to staves_path: staffs are
    not detected or grouped again, only the page's symbols are segmented. With an assignment
    the staffs are grouped into systems and parts as the user said (hidden staffs, grand
    staffs, ignored staffs); without, as saved.
    """
    config = _processing_config()
    page = homr_main.read_page_symbols(image_path, config)
    saved = load_precise_staff_positions(
        staves_path, invert(page.to_input_image), page.input_shape
    )
    systems = [_split(system) for system in saved]
    add_notes_to_staffs(
        [staff for system in systems for group in system for staff in group],
        page.noteheads_with_stems,
        page.predictions.symbols,
        page.predictions.notehead,
    )
    slots: list[tuple[int, ...]] | None = None
    if assignment is not None and slot_is_grandstaff is not None:
        multi_staffs, slots = _group_by_assignment(systems, assignment)
        eprint("Staffs grouped as assigned:", slots)
    else:
        multi_staffs = [
            MultiStaff(
                [_join(group) for group in system],
                [],
                certain=saved_system.certain,
            )
            for system, saved_system in zip(systems, saved, strict=True)
        ]
    eprint("Found", len(multi_staffs), "systems:", [len(m.staffs) for m in multi_staffs])

    transformer_config = Config()
    transformer_config.use_gpu_inference = config.transformer_use_gpu
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    context = OdysseyContext(page.bar_lines, image)
    voices = parse_staffs(
        page.debug,
        multi_staffs,
        page.predictions.preprocessed,
        selected_staff=-1,
        config=transformer_config,
        page_to_input_image=page.to_input_image,
        odyssey=context,
        slots=slots,
        slot_is_grandstaff=tuple(slot_is_grandstaff) if slots is not None and slot_is_grandstaff else None,
    )
    xml = generate_xml(XmlGeneratorArguments(False, None, None, image_positions=False), voices, "", inherited_times)
    page.debug.clean_debug_files_from_previous_runs()
    from homr.music_xml_generator import last_time_signatures  # noqa: PLC0415

    payload = build_payload(xml, context.symbols, context.geometry, (image.shape[1], image.shape[0]))
    return PageResult(
        musicxml=ET.tostring(xml, encoding="unicode"),
        payload=payload,
        geometry=context.geometry,
        last_times=last_time_signatures(xml),
    )


def _join(group: tuple[Staff, ...]) -> Staff:
    """The staff as it was saved: the two halves of a grand staff are merged again."""
    if len(group) == 1:
        return group[0]
    return group[0].merge(group[1])


def _group_by_assignment(
    systems: list[list[tuple[Staff, ...]]], assignment: PageAssignment
) -> tuple[list[MultiStaff], list[tuple[int, ...]]]:
    """
    Groups the saved staffs into the systems and parts of the assignment. The assignment
    names the part of every staff of every system in the order of detect_layout, so each
    staff is found by its place, not by where it is on the page.
    """
    if len(assignment) != len(systems):
        raise ValueError(
            f"The assignment covers {len(assignment)} systems, the saved page has {len(systems)}"
        )
    rows: list[MultiStaff] = []
    slots: list[tuple[int, ...]] = []
    for number, (system, assigned) in enumerate(zip(systems, assignment, strict=True)):
        staffs = [staff for group in system for staff in group]
        if len(assigned) != len(staffs):
            raise ValueError(
                f"System {number} has {len(staffs)} staffs, the assignment names {len(assigned)}"
            )
        members: dict[int, list[Staff]] = {}
        for staff, slot in zip(staffs, assigned, strict=True):
            if slot != IGNORE:
                members.setdefault(slot, []).append(staff)
        if not members:
            continue
        row: list[tuple[Staff, int]] = []
        for slot, found in members.items():
            found.sort(key=lambda st: st.min_y)
            merged = None
            if len(found) >= 2:
                try:
                    merged = found[0].merge(found[1])
                except (IndexError, ValueError):
                    eprint("Couldn't join two staffs into a grand staff, using the upper one")
            row.append((merged if merged is not None else found[0], slot))
        row.sort(key=lambda item: item[0].min_y)
        rows.append(MultiStaff([st for st, _ in row], [], certain=True))
        slots.append(tuple(slot for _, slot in row))
    return rows, slots
