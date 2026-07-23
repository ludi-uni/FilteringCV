from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.web.app import DEFAULT_HOST, DEFAULT_PORT, create_app


@pytest.fixture
def api_client(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
dataset_builder:
  enabled: true
  work_dir: work
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with TestClient(create_app(config_path, tmp_path)) as client:
        yield client


def test_default_bind_constants() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8765


def test_path_traversal_rejected_on_compare(api_client: TestClient) -> None:
    resp = api_client.post(
        "/api/compare",
        json={"left": "../secret", "right": "work"},
    )
    assert resp.status_code == 400
    assert "traversal" in resp.json()["detail"].lower()


def test_absolute_path_rejected_on_audio(api_client: TestClient) -> None:
    resp = api_client.get("/api/audio//etc/passwd")
    assert resp.status_code in (400, 404)


def test_dotdot_path_rejected_on_audio(api_client: TestClient) -> None:
    resp = api_client.get("/api/audio/..%2F..%2Foutside.wav")
    assert resp.status_code == 400


def test_dashboard_does_not_expose_arbitrary_paths(api_client: TestClient, tmp_path: Path) -> None:
    resp = api_client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert str(tmp_path) in body["work_dir"] or body["work_dir"].endswith("/work")
