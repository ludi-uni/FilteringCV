from __future__ import annotations

from pathlib import Path

from cv_preprocess.web.last_config import write_last_config
from cv_preprocess.web.session_resolve import resolve_gui_config_path


def test_cli_config_wins(tmp_path: Path) -> None:
    cfg = tmp_path / "a.yaml"
    cfg.write_text("x\n", encoding="utf-8")
    write_last_config(tmp_path, "b.yaml")
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=cfg) == cfg.resolve()


def test_falls_back_to_last(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "default.yaml"
    cfg.parent.mkdir()
    cfg.write_text("x\n", encoding="utf-8")
    write_last_config(tmp_path, "config/default.yaml")
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=None) == cfg.resolve()


def test_none_when_no_cli_no_last(tmp_path: Path) -> None:
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=None) is None
