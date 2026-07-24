from __future__ import annotations

from pathlib import Path

from cv_preprocess.web.app import create_app
from cv_preprocess.web.last_config import write_last_config
from cv_preprocess.web.session_resolve import (
    resolve_gui_config_path,
    resolve_gui_startup_config,
)


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


def test_startup_unloadable_last_config_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "config" / "bad.yaml"
    bad.parent.mkdir()
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    write_last_config(tmp_path, "config/bad.yaml")
    assert resolve_gui_config_path(project_root=tmp_path, cli_config=None) == bad.resolve()
    assert resolve_gui_startup_config(project_root=tmp_path, cli_config=None) is None


def test_startup_valid_last_config_returned(tmp_path: Path) -> None:
    cfg = tmp_path / "config" / "default.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "\n".join(
            [
                "schema_version: 2",
                "input:",
                "  corpus_root: .",
                "  clip_tsv: validated.tsv",
                "dataset_builder:",
                "  enabled: true",
                "  work_dir: work",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_last_config(tmp_path, "config/default.yaml")
    assert resolve_gui_startup_config(project_root=tmp_path, cli_config=None) == cfg.resolve()


def test_startup_cli_path_returned_even_if_unloadable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    assert resolve_gui_startup_config(project_root=tmp_path, cli_config=bad) == bad.resolve()


def test_startup_unloadable_last_config_create_app_unbound(tmp_path: Path) -> None:
    bad = tmp_path / "config" / "bad.yaml"
    bad.parent.mkdir()
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    write_last_config(tmp_path, "config/bad.yaml")
    resolved = resolve_gui_startup_config(project_root=tmp_path, cli_config=None)
    assert resolved is None
    from fastapi.testclient import TestClient

    with TestClient(create_app(resolved, tmp_path)) as client:
        status = client.get("/api/session")
        assert status.status_code == 200
        assert status.json()["bound"] is False
