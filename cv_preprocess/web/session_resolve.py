from __future__ import annotations

from pathlib import Path

from cv_preprocess.web.last_config import resolve_last_config_file


def resolve_gui_config_path(*, project_root: Path, cli_config: Path | None) -> Path | None:
    if cli_config is not None:
        return cli_config.resolve()
    return resolve_last_config_file(project_root)
