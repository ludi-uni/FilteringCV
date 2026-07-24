from __future__ import annotations

from pathlib import Path

import pytest

from cv_preprocess.web.last_config import (
    read_last_config,
    resolve_last_config_file,
    to_project_relative,
    write_last_config,
)


def test_write_and_read_relative(tmp_path: Path) -> None:
    rel = write_last_config(tmp_path, "config/default.yaml")
    assert rel.name == "last_config.json"
    assert read_last_config(tmp_path) == "config/default.yaml"


def test_to_project_relative_rejects_outside(tmp_path: Path) -> None:
    outside = tmp_path.parent / "other.yaml"
    outside.write_text("x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        to_project_relative(tmp_path, outside)


def test_resolve_missing_returns_none(tmp_path: Path) -> None:
    assert resolve_last_config_file(tmp_path) is None


def test_resolve_stale_path_returns_none(tmp_path: Path) -> None:
    write_last_config(tmp_path, "config/missing.yaml")
    assert resolve_last_config_file(tmp_path) is None


def test_resolve_valid_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir()
    target = cfg / "default.yaml"
    target.write_text("schema_version: 2\n", encoding="utf-8")
    write_last_config(tmp_path, "config/default.yaml")
    assert resolve_last_config_file(tmp_path) == target.resolve()
