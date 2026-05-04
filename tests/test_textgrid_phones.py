from pathlib import Path

from cv_preprocess.audio.textgrid_phones import extract_phone_tokens_from_textgrid, phones_to_space_string


def test_extract_phones_minimal_textgrid(tmp_path: Path) -> None:
    tg = tmp_path / "u.TextGrid"
    tg.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item [1]:
    class = "IntervalTier"
    name = "phones"
    xmin = 0
    xmax = 0.5
    intervals: size = 2
    intervals [1]:
        xmin = 0
        xmax = 0.1
        text = "a"
    intervals [2]:
        xmin = 0.1
        xmax = 0.5
        text = "b"
""",
        encoding="utf-8",
    )
    toks = extract_phone_tokens_from_textgrid(tg)
    assert toks == ["a", "b"]
    assert phones_to_space_string(toks) == "a b"


def test_skips_silence_labels(tmp_path: Path) -> None:
    tg = tmp_path / "u.TextGrid"
    tg.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 0.2
tiers? <exists>
size = 1
item [1]:
    class = "IntervalTier"
    name = "foo - phones"
    xmin = 0
    xmax = 0.2
    intervals: size = 2
    intervals [1]:
        xmin = 0
        xmax = 0.1
        text = "sp"
    intervals [2]:
        xmin = 0.1
        xmax = 0.2
        text = "k"
""",
        encoding="utf-8",
    )
    assert extract_phone_tokens_from_textgrid(tg) == ["k"]
