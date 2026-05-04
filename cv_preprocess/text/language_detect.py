from __future__ import annotations

from cv_preprocess.text.normalize import is_mostly_japanese


def passes_locale_expected(locale: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    if not locale:
        return True
    return locale.strip().lower().startswith(expected.strip().lower())


def passes_japanese_policy(text: str, require: bool) -> bool:
    if not require:
        return True
    return is_mostly_japanese(text)
