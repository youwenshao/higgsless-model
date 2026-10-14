"""Syllable counting and lyric fitting. New lyrics must carry the same number of sung notes."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# A few words the letter rules get wrong. Extend from mismatches you hear.
_EN = {
    "the": 1,
    "a": 1,
    "every": 2,
    "different": 3,
    "family": 3,
    "business": 2,
    "beautiful": 3,
    "fire": 1,
    "hour": 1,
    "people": 2,
    "little": 2,
    "something": 2,
    "nothing": 2,
    "being": 2,
    "real": 1,
    "idea": 3,
    "create": 2,
    "poem": 2,
    "quiet": 2,
    "science": 2,
}
_VOWELS = set("aeiouy")
_SMALL_KANA = set("ゃゅょぁぃぅぇぉャュョァィゥェォっッ")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_HANGUL = re.compile(r"[\uac00-\ud7a3]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_LATIN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def syllables_en(word: str) -> int:
    word = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not word:
        return 0
    if word in _EN:
        return _EN[word]
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in _VOWELS:
        count += 1
    return max(1, count) if re.search(r"[a-z]", word) else 0


def syllables(text: str, language: str) -> int:
    """Sung syllables in `text` for en, zh, yue, ja or ko. Markup like <SP> is ignored."""
    text = re.sub(r"<[^>]+>", " ", text)
    language = language.lower()
    if language in {"zh", "yue", "chinese", "mandarin", "cantonese"}:
        return len(_CJK.findall(text))
    if language in {"ko", "korean"}:
        hangul = len(_HANGUL.findall(text))
        latin = sum(syllables_en(w) for w in _LATIN.findall(text))
        return hangul + latin
    if language in {"ja", "japanese"}:
        return _japanese(text)
    return sum(syllables_en(w) for w in _LATIN.findall(text))


def _japanese(text: str) -> int:
    try:
        import fugashi  # type: ignore[import-not-found]

        tagger = fugashi.Tagger()
        morae = 0
        for word in tagger(text):
            reading = getattr(word.feature, "kana", None) or getattr(word.feature, "pron", None) or word.surface
            morae += sum(1 for ch in str(reading) if ch not in _SMALL_KANA and unicodedata.category(ch).startswith("L"))
        return morae
    except Exception:  # noqa: BLE001 - fugashi is optional
        kana = sum(1 for ch in text if _KANA.match(ch) and ch not in _SMALL_KANA)
        kanji = len(_CJK.findall(text))
        return kana + kanji


def units(text: str, language: str) -> list[str]:
    """The pieces that each take one sung note: words in English, characters in Chinese."""
    text = re.sub(r"<[^>]+>", " ", text)
    language = language.lower()
    if language in {"zh", "yue", "chinese", "mandarin", "cantonese"}:
        return _CJK.findall(text)
    if language in {"ko", "korean"}:
        return _HANGUL.findall(text) + _LATIN.findall(text)
    if language in {"ja", "japanese"}:
        return [ch for ch in text if (_KANA.match(ch) and ch not in _SMALL_KANA) or _CJK.match(ch)]
    return _LATIN.findall(text)


def sung_notes(segment: dict[str, Any]) -> int:
    """Notes that carry a pitch (rests, the leading 0s, are not sung)."""
    pitches = str(segment.get("note_pitch", "")).split()
    if not pitches:
        return 0
    return sum(1 for p in pitches if p not in {"0", "0.0"})


def segment_text(segment: dict[str, Any]) -> str:
    return re.sub(r"\s*<SP>\s*", " ", str(segment.get("text", ""))).strip()


def fit(metadata: list[dict[str, Any]], new_lines: list[str], language: str) -> dict[str, Any]:
    """Compare new lyric lines to the sung notes of each transcribed segment."""
    if len(new_lines) != len(metadata):
        return {
            "ok": False,
            "language": language,
            "error": f"{len(new_lines)} lyric lines for {len(metadata)} sung segments",
            "mismatches": [],
        }
    mismatches = []
    for segment, line in zip(metadata, new_lines, strict=True):
        expected = sung_notes(segment)
        pieces = units(line, language)
        if expected != len(pieces):
            mismatches.append(
                {
                    "index": segment.get("index"),
                    "expected_notes": expected,
                    "got_units": len(pieces),
                    "syllables": syllables(line, language),
                    "original": segment_text(segment),
                    "new": line,
                }
            )
    return {"ok": not mismatches, "language": language, "segments": len(metadata), "mismatches": mismatches}
