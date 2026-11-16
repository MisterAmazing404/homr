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
from homr.simple_logging import eprint
from homr.staff_parsing import parse_staffs
from homr.system_repair import repair_systems
from homr.transformer.configs import Config


# Per system, (top, bottom, slot) of each stave in the pixels of the input image; slot -1 = a stave to ignore
PageTemplate = list[list[tuple[float, float, int]]]
IGNORE = -1


@dataclass
class LayoutStaff:
    """One five-line staff as Odyssey's layout pass shows it, in the pixels of the input image."""

    line_ys: list[float]
    x_start: float
    x_end: float


def detect_layout(image_path: str) -> list[list[LayoutStaff]]:
    """
    The staffs of a page grouped into systems, top to bottom, by homr's own detection and
    grouping model. Odyssey shows these for the user to assign parts to, so the template the
    user makes refers to exactly the staffs that run_page will find. A grand staff is returned
    as its two staffs.
    """
    config = _processing_config()
    single = homr_main.detect_single_staffs(image_path, config)
    systems = homr_main.group_staffs(single.predictions, single.staffs)
    systems = repair_systems(single.debug, systems)
    single.debug.clean_debug_files_from_previous_runs()
    result = []
    for system in sorted(systems, key=lambda m: m.staffs[0].min_y):
        staffs: list[LayoutStaff] = []
        for staff in system.staffs:
            halves = [staff]
            if staff.is_grandstaff:
                split = staff.split()
                if split is not None:
                    halves = list(split)
            for half in halves:
                staffs.append(_layout_staff(half, single.to_input_image))
        result.append(sorted(staffs, key=lambda s: s.line_ys[0]))
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
    inherited_times: list[TimeSignature | None] | None = None,
    template: PageTemplate | None = None,
    slot_is_grandstaff: list[bool] | None = None,
) -> PageResult:
    """
    Read one page. With a template, the staffs homr detects are grouped into systems and parts
    as the template says (staffs hidden in some systems, grand staffs) instead of by homr's own
    model; staffs the template doesn't cover are ignored.
    """
    config = _processing_config()
    single = homr_main.detect_single_staffs(image_path, config)
    slots: list[tuple[int, ...]] | None = None
    multi_staffs = None
    if template and slot_is_grandstaff is not None:
        grouped = _group_by_template(single, template)
        if grouped is not None:
            multi_staffs, slots = grouped
            eprint("Staffs grouped by the template:", slots)
    if multi_staffs is None:
        multi_staffs = homr_main.group_staffs(single.predictions, single.staffs)
        multi_staffs = repair_systems(single.debug, multi_staffs)
    eprint("Found", len(multi_staffs), "systems:", [len(m.staffs) for m in multi_staffs])

    transformer_config = Config()
    transformer_config.use_gpu_inference = config.transformer_use_gpu
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    context = OdysseyContext(single.bar_lines, image)
    voices = parse_staffs(
        single.debug,
        multi_staffs,
        single.predictions.preprocessed,
        selected_staff=-1,
        config=transformer_config,
        page_to_input_image=single.to_input_image,
        odyssey=context,
        slots=slots,
        slot_is_grandstaff=tuple(slot_is_grandstaff) if slots is not None and slot_is_grandstaff else None,
    )
    xml = generate_xml(XmlGeneratorArguments(False, None, None, image_positions=False), voices, "", inherited_times)
    single.debug.clean_debug_files_from_previous_runs()
    from homr.music_xml_generator import last_time_signatures  # noqa: PLC0415

    payload = build_payload(xml, context.symbols, context.geometry, (image.shape[1], image.shape[0]))
    return PageResult(
        musicxml=ET.tostring(xml, encoding="unicode"),
        payload=payload,
        geometry=context.geometry,
        last_times=last_time_signatures(xml),
    )


def _group_by_template(
    single: "homr_main.SingleStaffs", template: PageTemplate
) -> tuple[list[MultiStaff], list[tuple[int, ...]]] | None:
    """Match homr's staffs to the template's staves by their vertical centre."""
    from homr import staff_relations  # noqa: PLC0415

    staffs = staff_relations.join_pieces_with_model(single.predictions, single.staffs)

    def centre(staff: Staff) -> float:
        mid = staff.grid[len(staff.grid) // 2]
        return single.to_input_image((mid.x, float(np.mean(mid.y))))[1]

    rows: list[MultiStaff] = []
    slots: list[tuple[int, ...]] = []
    used: set[int] = set()
    for system in template:
        members: dict[int, list[Staff]] = {}
        for top, bottom, slot in system:
            margin = (bottom - top) / 2
            candidates = [
                (abs(centre(st) - (top + bottom) / 2), i, st)
                for i, st in enumerate(staffs)
                if i not in used and top - margin <= centre(st) <= bottom + margin
            ]
            if not candidates:
                continue
            _, i, st = min(candidates, key=lambda c: c[0])
            used.add(i)
            if slot != IGNORE:
                members.setdefault(slot, []).append(st)
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
    if not rows:
        return None
    if len(used) < len(staffs):
        # never drop music the template doesn't know about: read the page with homr's own grouping
        eprint("The template doesn't cover", len(staffs) - len(used), "staffs; using homr's own grouping")
        return None
    return rows, slots
