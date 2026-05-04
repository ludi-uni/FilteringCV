"""MFA CLI 解決ヘルパー。"""

import sys

from cv_preprocess.audio.mfa_batch import mfa_cli_available


def test_mfa_cli_available_finds_python() -> None:
    assert mfa_cli_available(sys.executable) is True


def test_mfa_cli_available_missing_command() -> None:
    assert mfa_cli_available("__no_such_mfa_command_xyz__") is False
