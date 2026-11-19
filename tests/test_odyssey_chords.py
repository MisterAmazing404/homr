import unittest

from homr.odyssey_align import _is_lower, chords_of
from homr.transformer.vocabulary import EncodedSymbol


def note(pitch: str, position: str = "upper") -> EncodedSymbol:
    return EncodedSymbol("note_8", pitch, position=position)


def chord() -> EncodedSymbol:
    return EncodedSymbol("chord", ".", position=".")


def half(symbols: list[EncodedSymbol], lower: bool) -> list[EncodedSymbol]:
    # as OdysseyContext.annotate picks a grand staff half (chord tokens have no position)
    return [s for s in symbols if _is_lower(s) == lower]


def pitches(chords: list[list[EncodedSymbol]]) -> list[list[str]]:
    return [[s.pitch for s in c] for c in chords]


class TestOdysseyChords(unittest.TestCase):
    def test_a_chord_across_the_grand_staff_does_not_join_the_next_note(self) -> None:
        # snippet 1795004083875: piano bar 1, right hand C5 D5 ... with the left hand's C3 under C5
        stream = [
            note("C5"), chord(), note("C3", "lower"),
            note("D5"),
            note("E5"),
            note("G5"), chord(), note("G2", "lower"),
            note("A5"),
        ]
        self.assertEqual(
            pitches(chords_of(half(stream, False), stream)), [["C5"], ["D5"], ["E5"], ["G5"], ["A5"]]
        )
        self.assertEqual(pitches(chords_of(half(stream, True), stream)), [["C3"], ["G2"]])

    def test_a_chord_in_one_hand_stays_together(self) -> None:
        stream = [note("A5"), chord(), note("G5"), chord(), note("C3", "lower"), note("E5")]
        self.assertEqual(pitches(chords_of(half(stream, False), stream)), [["A5", "G5"], ["E5"]])

    def test_a_rest_is_not_joined_to_a_chord(self) -> None:
        stream = [note("C5"), EncodedSymbol("rest_8", position="upper"), chord(), note("D5")]
        self.assertEqual(pitches(chords_of(stream, stream)), [["C5"], ["D5"]])


if __name__ == "__main__":
    unittest.main()
