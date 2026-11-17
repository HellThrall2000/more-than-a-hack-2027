import pytest

from possession.clock import clock_to_frame, format_clock, frame_to_clock


def test_kickoffs_are_the_clock_anchors():
    assert clock_to_frame("00:00") == 11185
    assert clock_to_frame("45:00", half=2) == 113742
    assert clock_to_frame("51:35", half=2) == 113742 + (6 * 60 + 35) * 25


def test_frame_and_clock_round_trip_inside_both_halves():
    for frame in (11185, 12851, 90651, 113742, 150000, 190197):
        half, sec = frame_to_clock(frame)
        assert clock_to_frame(format_clock(sec), half) == pytest.approx(frame, abs=3)    # 0.1 s display


def test_format_never_shows_60_seconds():
    assert format_clock(299.96) == "05:00.0" and format_clock(66.64) == "01:06.6"


def test_clock_from_45_minutes_needs_the_half():
    with pytest.raises(ValueError, match="ambiguous"):
        clock_to_frame("47:10")
    assert clock_to_frame("47:10", half=1) < clock_to_frame("47:10", half=2)
    with pytest.raises(ValueError, match="after the end"):
        clock_to_frame("80:00", half=1)
    with pytest.raises(ValueError, match="before the kick-off"):
        clock_to_frame("10:00", half=2)


def test_no_clock_is_not_the_same_as_ungraded():
    assert frame_to_clock(110808)[0] is None and "graded and all D" in frame_to_clock(110808)[1]
    assert frame_to_clock(113741)[0] is None
    assert "ungraded" in frame_to_clock(100000)[1]          # half-time
    assert "ungraded" in frame_to_clock(5000)[1]
    assert "ungraded" in frame_to_clock(195000)[1]
