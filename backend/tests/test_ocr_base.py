from image_to_text.ocr.base import Line, Word, ms, to_ms


def test_line_geometry_from_box_corners():
    line = Line("Sugar", 0.9, ((10, 20), (110, 22), (110, 52), (12, 50)))
    assert line.top == 20
    assert line.left == 10
    assert line.height == 32


def test_line_keeps_word_confidences():
    words = (Word("Wheat", 0.95), Word("Flour", 0.8))
    line = Line("Wheat Flour", 0.8, ((0, 0), (1, 0), (1, 1), (0, 1)), words)
    assert [w.conf for w in line.words] == [0.95, 0.8]


def test_to_ms_rounds_to_tenths():
    assert to_ms(0.12345) == 123.5


def test_ms_treats_missing_stage_as_zero():
    assert ms(det=0.01, rec=None) == {"det": 10.0, "rec": 0.0}
