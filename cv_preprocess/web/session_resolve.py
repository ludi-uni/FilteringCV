from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from cv_preprocess.config import load_config
from cv_preprocess.web.last_config import resolve_last_config_file

logger = logging.getLogger(__name__)


def resolve_gui_config_path(*, project_root: Path, cli_config: Path | None) -> Path | None:
    if cli_config is not None:
        return cli_config.resolve()
    return resolve_last_config_file(project_root)


def resolve_gui_startup_config(*, project_root: Path, cli_config: Path | None) -> Path | None:
    """Resolve config for GUI process startup.

    Explicit ``-c`` paths are returned as-is (caller must fail hard on load).
    When only ``last_config`` is available, unloadable YAML yields ``None``
    (unbound Setup) after a warning instead of crashing startup.
    """
    if cli_config is not None:
        return cli_config.resolve()
    path = resolve_last_config_file(project_root)
    if path is None:
        return None
    try:
        load_config(path)
    except (ValidationError, ValueError, OSError, TypeError) as exc:
        logger.warning(
            "last_config %s is unloadable (%s); starting unbound setup",
            path,
            exc,
        )
        return None
    return path
