from pathlib import Path

import pytest

from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml, map_mfa_space_separated_to_g2p_tokens
from cv_preprocess.text.phoneme_compare import phoneme_sequences_accept


def test_map_mfa_to_g2p_roundtrip() -> None:
    m = {"MFA_X": "a i", "MFA_Y": ""}
    assert map_mfa_space_separated_to_g2p_tokens("MFA_X MFA_Y MFA_X", m) == "a i a i"


def test_map_unmapped_passthrough() -> None:
    assert map_mfa_space_separated_to_g2p_tokens("k o", {}) == "k o"


def test_load_yaml(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("a: b c\n", encoding="utf-8")
    d = load_mfa_token_map_yaml(p)
    assert d == {"a": "b c"}


def test_mfa_mapped_accepted_by_phoneme_compare() -> None:
    g2p = "a b"
    mfa_raw = "X Y"
    m = {"X": "a", "Y": "b"}
    mapped = map_mfa_space_separated_to_g2p_tokens(mfa_raw, m)
    assert mapped == "a b"
    assert phoneme_sequences_accept(g2p, mapped, max_token_error_rate=0.0)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mfa_token_map_yaml(tmp_path / "does_not_exist.yaml")
