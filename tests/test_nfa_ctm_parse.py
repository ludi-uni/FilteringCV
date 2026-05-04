"""NFA token CTM のパース。"""

from pathlib import Path

from cv_preprocess.audio.nfa_batch import parse_token_ctm_file, tokens_to_space_string


def test_parse_token_ctm_file_roundtrip(tmp_path: Path) -> None:
    # NeMo get_ctm_line 相当: utt channel start dur token conf type speaker
    lines = [
        "u00000001 1 0.00 0.10 ▁テ lex NA unknown\n",
        "u00000001 1 0.10 0.05 スト lex NA unknown\n",
    ]
    p = tmp_path / "u00000001.ctm"
    p.write_text("".join(lines), encoding="utf-8")
    toks = parse_token_ctm_file(p)
    assert toks == ["▁テ", "スト"]
    assert tokens_to_space_string(toks) == "▁テ スト"
