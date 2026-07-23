from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app


@pytest.fixture
def api_client(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    work = tmp_path / "work"
    cache = work / "audio_cache" / "hash" / "ab"
    cache.mkdir(parents=True)
    wav = cache / "sample.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40 + b"WAVEfmt ")

    config_path.write_text(
        f"""
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
dataset_builder:
  enabled: true
  work_dir: {work.name}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with TestClient(create_app(config_path, tmp_path)) as client:
        yield client


def test_audio_full_file(api_client: TestClient) -> None:
    resp = api_client.get("/api/audio/hash/ab/sample.wav")
    assert resp.status_code == 200
    assert resp.content.startswith(b"RIFF")


def test_audio_range_request(api_client: TestClient) -> None:
    resp = api_client.get(
        "/api/audio/hash/ab/sample.wav",
        headers={"Range": "bytes=0-3"},
    )
    assert resp.status_code == 206
    assert resp.content == b"RIFF"
    assert resp.headers.get("content-range", "").startswith("bytes 0-3/")


def test_audio_range_suffix(api_client: TestClient) -> None:
    full = api_client.get("/api/audio/hash/ab/sample.wav")
    suffix = api_client.get(
        "/api/audio/hash/ab/sample.wav",
        headers={"Range": "bytes=-4"},
    )
    assert suffix.status_code == 206
    assert suffix.content == full.content[-4:]
