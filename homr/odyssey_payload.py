"""The payload Odyssey takes from homr next to the MusicXML.

MusicXML has no place for most of what Odyssey needs (where a symbol is on the page, how sure
the decoder was, whether the position was snapped to a detected notehead, which system and
part it came from, the staff lines, the barlines). The payload is a JSON-serialisable dict
with all of it, and says where in the MusicXML each symbol ended up, so a reader can match the
two without relying on anything hidden in the XML itself.

    {
      "format": "odyssey-homr-payload", "version": 1,
      "image": {"width", "height"},
      "staves":  [{"id", "system", "part", "half", "line_ys", "x_min", "x_max", "unit",
                   "bars", "hairpins"}],              # pixels of the image the user supplied
      "parts":   [{"index", "staves", "measures": [{"index", "number", "new_system",
                                                    "barline", "bar_x"}]}],
      "symbols": [{"id", "kind", "rhythm", "pitch", "lift", "articulation", "slur", "position",
                   "x", "y", "confidence", "snapped", "system", "part", "half",
                   "xml": {"part", "measure", "staff", "voice", "offset", "duration",
                           "chord", "seq", "pitch"} | null}]
    }

A symbol id is "<system>.<part>.<half>.<n>": n counts the symbols read from that staff, left
to right. `kind` is note, rest, barline, clef, key, time, dynamic or other. `xml` is set for
the symbols that became a `<note>`: `part` and `measure` are 0-based indexes into the MusicXML,
`offset` and `duration` are quarter notes as "n/d" strings (the offset is from the start of the
measure), `chord` is true for a note that sounds with the previous one, `seq` is the note's place among
the `<note>` elements of its measure and `pitch` is {step, alter, octave} as written in the XML
(null for a rest).
In `parts`, `barline` is the id of the barline symbol that ended the measure and `bar_x` its x
when it was snapped to a detected barline.
"""

import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any

from homr.music_xml_generator import (
    ODYSSEY_BARLINE_ATTRIBUTE,
    ODYSSEY_SYMBOL_ATTRIBUTE,
)
from homr.odyssey_align import StaffGeometry
from homr.odyssey_boxes import box_to_json
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote

FORMAT = "odyssey-homr-payload"
VERSION = 1


def symbol_kind(symbol: EncodedSymbol) -> str:
    rhythm = symbol.rhythm
    if rhythm.startswith(("note", "rest")):
        is_rest = rhythm.startswith("rest") or symbol.pitch in (empty, nonote)
        return "rest" if is_rest else "note"
    if "barline" in rhythm or rhythm.startswith(("repeat", "volta")):
        return "barline"
    if rhythm.startswith("clef"):
        return "clef"
    if rhythm.startswith("keySignature"):
        return "key"
    if rhythm.startswith("timeSignature"):
        return "time"
    if rhythm.startswith("dynamic") or symbol.articulation.startswith("dynamic"):
        return "dynamic"
    return "other"


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def collect_xml_locations(root: ET.Element) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Where each payload symbol went in the MusicXML, and the measures of each part. Removes the
    temporary `odyssey-*` attributes the generator left, so the XML is plain MusicXML again.
    """
    locations: dict[str, dict[str, Any]] = {}
    parts: list[dict[str, Any]] = []
    for part_index, part in enumerate(root.findall("part")):
        staves = 1
        measures: list[dict[str, Any]] = []
        divisions = 1
        for measure_index, measure in enumerate(part.findall("measure")):
            staves = max(staves, int(measure.findtext("attributes/staves", "1") or 1))
            divisions = int(measure.findtext("attributes/divisions", str(divisions)) or divisions)
            print_el = measure.find("print")
            measures.append(
                {
                    "index": measure_index,
                    "number": int(measure.get("number", "0")),
                    "new_system": print_el is not None and print_el.get("new-system") == "yes",
                    "barline": measure.attrib.pop(ODYSSEY_BARLINE_ATTRIBUTE, None),
                }
            )
            position = 0  # in divisions
            start = 0
            seq = 0  # the order of the <note> elements in the measure
            for child in measure:
                duration = int(child.findtext("duration", "0") or 0)
                if child.tag == "backup":
                    position -= duration
                elif child.tag == "forward":
                    position += duration
                elif child.tag == "note":
                    chord = child.find("chord") is not None
                    seq += 1
                    if not chord:
                        start = position
                        position += duration
                    symbol_id = child.attrib.pop(ODYSSEY_SYMBOL_ATTRIBUTE, None)
                    if symbol_id is not None:
                        locations[symbol_id] = {
                            "part": part_index,
                            "measure": measure_index,
                            "staff": int(child.findtext("staff", "1") or 1),
                            "voice": child.findtext("voice", "1"),
                            "offset": _fraction(Fraction(start, divisions)),
                            "duration": _fraction(Fraction(duration, divisions)),
                            "chord": chord,
                            "seq": seq - 1,
                            "pitch": _xml_pitch(child),
                        }
        parts.append({"index": part_index, "staves": staves, "measures": measures})
    return locations, parts


def _xml_pitch(note: ET.Element) -> dict[str, Any] | None:
    """The pitch as written in the MusicXML (after octave clefs and shifts), None for a rest."""
    pitch = note.find("pitch")
    if pitch is None:
        return None
    return {
        "step": pitch.findtext("step", ""),
        "alter": int(float(pitch.findtext("alter", "0") or 0)),
        "octave": int(pitch.findtext("octave", "0") or 0),
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def _symbol_json(
    symbol_id: str, symbol: EncodedSymbol, location: dict[str, Any] | None
) -> dict[str, Any]:
    x, y = symbol.image_coordinates if symbol.image_coordinates is not None else (None, None)
    return {
        "id": symbol_id,
        "kind": symbol_kind(symbol),
        "rhythm": symbol.rhythm,
        "pitch": symbol.pitch,
        "lift": symbol.lift,
        "articulation": symbol.articulation,
        "slur": symbol.slur,
        "position": symbol.position,
        "x": _round(x),
        "y": _round(y),
        "box": box_to_json(symbol.odyssey_box) if symbol.odyssey_box is not None else None,
        "confidence": None if symbol.confidence is None else round(float(symbol.confidence), 3),
        "snapped": bool(symbol.snapped),
        "system": symbol.odyssey_system,
        "part": symbol.odyssey_part,
        "half": symbol.odyssey_half,
        "xml": location,
    }


def build_payload(
    root: ET.Element,
    symbols: dict[str, EncodedSymbol],
    geometry: list[StaffGeometry],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """The payload for a page whose MusicXML is `root`. Strips the generator's temporary attributes."""
    locations, parts = collect_xml_locations(root)
    for part in parts:
        for measure in part["measures"]:
            barline = symbols.get(measure["barline"]) if measure["barline"] else None
            snapped = barline is not None and barline.snapped and barline.image_coordinates is not None
            measure["bar_x"] = _round(barline.image_coordinates[0]) if snapped else None  # type: ignore[union-attr, index]
            measure["bar_box"] = (
                box_to_json(barline.odyssey_box) if snapped and barline is not None and barline.odyssey_box is not None else None
            )
    return {
        "format": FORMAT,
        "version": VERSION,
        "image": {"width": image_size[0], "height": image_size[1]},
        "staves": [g.to_json() for g in geometry],
        "parts": parts,
        "symbols": [
            _symbol_json(symbol_id, symbol, locations.get(symbol_id))
            for symbol_id, symbol in symbols.items()
        ],
    }
