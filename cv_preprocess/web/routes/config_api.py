from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from cv_preprocess.config import PipelineConfig
from cv_preprocess.web.dependencies import build_app_state, get_app_session, get_app_state

router = APIRouter()

# Top-level sections shown in the GUI (order matters for UX).
CONFIG_SECTIONS: list[dict[str, str]] = [
    {"id": "input", "label": "Input", "group": "corpus"},
    {"id": "speakers", "label": "Speakers & filters", "group": "filters"},
    {"id": "output", "label": "Output", "group": "corpus"},
    {"id": "dataset_builder", "label": "Dataset builder", "group": "builder"},
    {"id": "compute", "label": "Compute", "group": "builder"},
    {"id": "quality_gate", "label": "Quality gate", "group": "audio"},
    {"id": "early_audio_gate", "label": "Early audio gate", "group": "audio"},
    {"id": "audio_pipeline", "label": "Audio pipeline", "group": "audio"},
    {"id": "audio_pipeline_align", "label": "Audio pipeline (align)", "group": "audio"},
    {"id": "audio_pipeline_enhance", "label": "Audio pipeline (enhance)", "group": "audio"},
    {"id": "two_pass_denoise", "label": "Two-pass denoise", "group": "audio"},
    {"id": "mfa_gate", "label": "MFA gate", "group": "gates"},
    {"id": "nfa_gate", "label": "NFA gate", "group": "gates"},
    {"id": "asr_gate", "label": "ASR gate", "group": "gates"},
    {"id": "text", "label": "Text / G2P", "group": "text"},
    {"id": "split", "label": "Legacy split", "group": "corpus"},
    {"id": "snr", "label": "SNR", "group": "audio"},
    {"id": "secondary", "label": "Secondary", "group": "advanced"},
    {"id": "phoneme_manifest", "label": "Phoneme manifest", "group": "advanced"},
    {"id": "schema_version", "label": "Schema version", "group": "meta"},
]


class ConfigResponse(BaseModel):
    path: str
    relative_path: str
    yaml_text: str
    data: dict[str, Any]
    sections: list[dict[str, str]] = Field(default_factory=lambda: list(CONFIG_SECTIONS))
    json_schema: dict[str, Any]


class ConfigValidateRequest(BaseModel):
    data: dict[str, Any] | None = None
    yaml_text: str | None = None


class ConfigValidateResponse(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    yaml_text: str | None = None
    errors: list[str] = Field(default_factory=list)


class ConfigSaveRequest(BaseModel):
    data: dict[str, Any] | None = None
    yaml_text: str | None = None
    mode: str = "data"  # data | yaml


class ConfigSaveResponse(BaseModel):
    ok: bool
    path: str
    data: dict[str, Any]
    yaml_text: str
    message: str = "saved"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _config_to_data(config: PipelineConfig) -> dict[str, Any]:
    return _jsonable(config.model_dump(mode="python"))


def _dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _parse_payload(
    *,
    data: dict[str, Any] | None,
    yaml_text: str | None,
) -> dict[str, Any]:
    if yaml_text is not None and data is not None:
        raise HTTPException(status_code=400, detail="provide either data or yaml_text, not both")
    if yaml_text is not None:
        try:
            parsed = yaml.safe_load(yaml_text)
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"invalid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML root must be a mapping")
        return parsed
    if data is not None:
        return data
    raise HTTPException(status_code=400, detail="data or yaml_text is required")


def _validate_config_dict(raw: dict[str, Any]) -> tuple[PipelineConfig | None, list[str]]:
    try:
        cfg = PipelineConfig.model_validate(raw)
        return cfg, []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = err.get("msg", "validation error")
            errors.append(f"{loc}: {msg}" if loc else str(msg))
        return None, errors


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _relative_config_path(state_path: Path, project_root: Path) -> str:
    try:
        return str(state_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(state_path)


@router.get("", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    state = get_app_state(request)
    path = state.config_path
    if path.is_file():
        yaml_text = path.read_text(encoding="utf-8")
    else:
        yaml_text = _dump_yaml(_config_to_data(state.config))
    data = _config_to_data(state.config)
    return ConfigResponse(
        path=str(path),
        relative_path=_relative_config_path(path, state.project_root),
        yaml_text=yaml_text,
        data=data,
        json_schema=PipelineConfig.model_json_schema(),
    )


@router.post("/validate", response_model=ConfigValidateResponse)
def validate_config(request: Request, body: ConfigValidateRequest) -> ConfigValidateResponse:
    raw = _parse_payload(data=body.data, yaml_text=body.yaml_text)
    cfg, errors = _validate_config_dict(raw)
    if cfg is None:
        return ConfigValidateResponse(ok=False, errors=errors)
    data = _config_to_data(cfg)
    return ConfigValidateResponse(ok=True, data=data, yaml_text=_dump_yaml(data), errors=[])


@router.put("", response_model=ConfigSaveResponse)
def save_config(request: Request, body: ConfigSaveRequest) -> ConfigSaveResponse:
    state = get_app_state(request)
    if body.mode == "yaml":
        raw = _parse_payload(data=None, yaml_text=body.yaml_text)
        yaml_out = body.yaml_text if body.yaml_text is not None else _dump_yaml(raw)
    else:
        raw = _parse_payload(data=body.data, yaml_text=None)
        yaml_out = _dump_yaml(raw)

    cfg, errors = _validate_config_dict(raw)
    if cfg is None:
        raise HTTPException(status_code=400, detail="; ".join(errors) or "invalid config")

    # Always persist a validated dump so Path/enum values are canonical.
    data = _config_to_data(cfg)
    if body.mode != "yaml":
        yaml_out = _dump_yaml(data)

    target = state.config_path.resolve()
    # Refuse writing outside project root (config may live in config/).
    try:
        target.relative_to(state.project_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="config path outside project root") from exc

    _atomic_write_text(target, yaml_out if yaml_out.endswith("\n") else yaml_out + "\n")

    # Reload config while preserving the live job store/runner.
    rebuilt = build_app_state(target, state.project_root)
    refreshed = replace(
        rebuilt,
        job_store=state.job_store,
        job_runner=state.job_runner,
    )
    refreshed.job_runner.config_path = refreshed.config_path
    session = get_app_session(request)
    session.app_state = refreshed
    request.app.state.app_state = refreshed

    return ConfigSaveResponse(
        ok=True,
        path=str(target),
        data=_config_to_data(refreshed.config),
        yaml_text=target.read_text(encoding="utf-8"),
        message=f"saved to {target.name}",
    )
