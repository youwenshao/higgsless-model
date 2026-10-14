from fractions import Fraction

import pytest

from higgsless.core.timecode import (
    format_rate,
    frame_to_sample,
    frame_to_timecode,
    parse_rate,
    plan_frame_policy,
    seconds_to_frame,
)
from higgsless.core.windows import gen_size, length_4n1, plan_windows


def test_parse_rate_exact_and_ntsc():
    assert parse_rate("24000/1001") == Fraction(24000, 1001)
    assert parse_rate("23.976") == Fraction(24000, 1001)
    assert parse_rate(29.97) == Fraction(30000, 1001)
    assert parse_rate("25") == 25
    assert format_rate(Fraction(24000, 1001)) == "24000/1001"
    assert format_rate(Fraction(25)) == "25"


def test_seconds_frames_and_samples_are_contiguous():
    rate = Fraction(24000, 1001)
    assert seconds_to_frame(0, rate) == 0
    assert seconds_to_frame(Fraction(1001, 24000) * 10, rate) == 10
    # Consecutive frames map to consecutive, non-overlapping sample ranges.
    starts = [frame_to_sample(f, rate, 48000) for f in range(100)]
    assert starts == sorted(starts)
    assert starts[24] == 48048  # 24 frames at 23.976 fps = 1.001 s = 48048 samples at 48 kHz
    assert frame_to_timecode(4316, rate) == "00:03:00.013"


def test_frame_policy_decimates_fast_sources_by_integer_factor():
    assert plan_frame_policy(Fraction(24000, 1001)).decimate == 1
    assert plan_frame_policy(Fraction(30)).decimate == 1
    policy = plan_frame_policy(Fraction(60000, 1001))
    assert policy.decimate == 2
    assert policy.gen_rate == Fraction(30000, 1001)
    assert plan_frame_policy(Fraction(120)).decimate == 4
    assert policy.gen_frames(101) == 51


def test_length_4n1():
    assert [length_4n1(n) for n in (1, 2, 5, 6, 80, 81, 82)] == [1, 5, 5, 9, 81, 81, 85]


def test_gen_size_snaps_to_multiple():
    assert gen_size(1920, 1080, 720, 32) == (1280, 704)
    assert gen_size(1920, 1080, 480, 32) == (864, 480)  # 853.3 rounds to the nearest multiple of 32
    assert gen_size(1080, 1920, 720, 16) == (720, 1280)


@pytest.mark.parametrize("total", [1, 30, 81, 82, 200, 333])
@pytest.mark.parametrize("overlap", [0, 1, 5])
def test_windows_cover_every_frame_exactly_once(total, overlap):
    windows = plan_windows(total, window=81, overlap=overlap)
    emitted = []
    for w in windows:
        assert (w.length - 1) % 4 == 0 and w.length <= 81
        emitted.extend(range(w.emit_start, w.emit_start + w.emit_count))
    assert emitted == list(range(total))
    assert windows[0].overlap == 0
    assert all(w.overlap == overlap for w in windows[1:])


def test_short_shot_uses_small_window():
    (w,) = plan_windows(30, window=81, overlap=1)
    assert w.length == 33 and w.pad == 3 and w.emit_count == 30
