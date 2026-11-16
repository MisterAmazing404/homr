import json
import unittest
import xml.etree.ElementTree as ET

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.odyssey_payload import build_payload, symbol_kind
from homr.transformer.vocabulary import EncodedSymbol


def _symbol(symbol_id: str, rhythm: str, pitch: str = ".", position: str = "upper") -> EncodedSymbol:
    lift = "_" if pitch not in (".",) else "."
    symbol = EncodedSymbol(rhythm, pitch, lift, "_" if pitch != "." else ".", "_" if pitch != "." else ".", position)
    symbol.odyssey_id = symbol_id
    symbol.odyssey_system, symbol.odyssey_part, symbol.odyssey_half = 0, 0, 0
    symbol.image_coordinates = (100.0 + 10 * len(symbol_id), 50.0)
    symbol.confidence = 0.9
    symbol.snapped = True
    return symbol


def _page() -> tuple[ET.Element, dict[str, EncodedSymbol]]:
    symbols = [
        _symbol("0.0.0.0", "clef_G2", position="upper"),
        _symbol("0.0.0.1", "note_4", "C5"),
        _symbol("0.0.0.2", "note_4", "E5"),  # chord with the previous note
        _symbol("0.0.0.3", "rest_4", "_"),
        _symbol("0.0.0.4", "note_2", "G4"),
        _symbol("0.0.0.5", "barline"),
        _symbol("0.0.0.6", "note_1", "C5"),
        _symbol("0.0.0.7", "barline"),
    ]
    flat = [*symbols[:2], EncodedSymbol("chord"), symbols[2], *symbols[3:]]
    # the second note of a chord is written after a "chord" marker in homr's token stream
    root = generate_xml(XmlGeneratorArguments(False, None, None, image_positions=False), [flat], "")
    return root, {s.odyssey_id: s for s in symbols if s.odyssey_id}


class OdysseyPayloadTest(unittest.TestCase):
    def test_kinds(self) -> None:
        self.assertEqual(symbol_kind(EncodedSymbol("note_4", "C5")), "note")
        self.assertEqual(symbol_kind(EncodedSymbol("rest_4", "_")), "rest")
        self.assertEqual(symbol_kind(EncodedSymbol("barline")), "barline")
        self.assertEqual(symbol_kind(EncodedSymbol("clef_G2")), "clef")

    def test_symbols_say_where_they_went_in_the_xml(self) -> None:
        root, symbols = _page()
        payload = build_payload(root, symbols, [], (200, 100))
        by_id = {s["id"]: s for s in payload["symbols"]}
        first, second, rest, half = (by_id[f"0.0.0.{i}"]["xml"] for i in (1, 2, 3, 4))
        # a chord is two notes at the same place, the second one a chord member
        self.assertEqual((first["measure"], first["offset"]), (0, "0/1"))
        self.assertEqual((second["measure"], second["offset"]), (0, "0/1"))
        self.assertEqual(sorted([first["chord"], second["chord"]]), [False, True])
        self.assertEqual(sorted([first["seq"], second["seq"]]), [0, 1])
        self.assertEqual((rest["offset"], half["offset"]), ("1/1", "2/1"))
        self.assertEqual(by_id["0.0.0.6"]["xml"]["measure"], 1)
        self.assertIsNone(by_id["0.0.0.0"]["xml"])  # a clef isn't a note
        self.assertEqual(by_id["0.0.0.1"]["confidence"], 0.9)
        self.assertTrue(by_id["0.0.0.1"]["snapped"])
        self.assertEqual(payload["image"], {"width": 200, "height": 100})

    def test_barlines_and_measures(self) -> None:
        root, symbols = _page()
        payload = build_payload(root, symbols, [], (200, 100))
        measures = payload["parts"][0]["measures"]
        self.assertEqual([m["barline"] for m in measures], ["0.0.0.5", "0.0.0.7"])
        self.assertEqual(measures[0]["bar_x"], round(symbols["0.0.0.5"].image_coordinates[0], 1))

    def test_the_xml_is_plain_musicxml_again(self) -> None:
        root, symbols = _page()
        build_payload(root, symbols, [], (200, 100))
        text = ET.tostring(root, encoding="unicode")
        self.assertNotIn("odyssey", text)
        self.assertNotIn("<!--", text)

    def test_payload_is_json(self) -> None:
        root, symbols = _page()
        json.dumps(build_payload(root, symbols, [], (200, 100)))


if __name__ == "__main__":
    unittest.main()
