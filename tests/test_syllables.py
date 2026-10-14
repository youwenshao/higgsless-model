from higgsless.audio.syllables import fit, syllables, units

SEGMENT = [{"index": "v0", "text": "<SP> who says you are <SP>", "note_pitch": "0 60 62 64 65 0"}]


def test_english_units_are_words_and_syllables_are_counted():
    assert units("who says you are", "en") == ["who", "says", "you", "are"]
    assert syllables("beautiful", "en") == 3
    assert syllables("people", "en") == 2
    assert syllables("the fire", "en") == 2


def test_chinese_and_korean_count_characters():
    assert units("你好世界", "zh") == ["你", "好", "世", "界"]
    assert syllables("你好世界", "yue") == 4
    assert units("안녕 world", "ko") == ["안", "녕", "world"]


def test_fit_reports_lines_that_do_not_match_the_sung_notes():
    bad = fit(SEGMENT, ["one two"], "en")
    assert not bad["ok"] and bad["mismatches"][0]["expected_notes"] == 4
    good = fit(SEGMENT, ["who says you are"], "en")
    assert good["ok"]
    assert not fit(SEGMENT, ["only one line", "and another"], "en")["ok"]
