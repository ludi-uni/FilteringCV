from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app


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


def test_overrides_list_empty(api_client: TestClient) -> None:
    resp = api_client.get("/api/overrides")
    assert resp.status_code == 200
    body = resp.json()
    assert body["overrides"] == []


def test_overrides_upsert_and_delete(api_client: TestClient, tmp_path: Path) -> None:
    upsert = api_client.put(
        "/api/overrides",
        json={
            "clip_id": "clip-1",
            "action": "force_include",
            "reason": "manual",
        },
    )
    assert upsert.status_code == 200
    overrides = upsert.json()["overrides"]
    assert len(overrides) == 1
    assert overrides[0]["clip_id"] == "clip-1"
    assert overrides[0]["action"] == "force_include"

    listed = api_client.get("/api/overrides")
    assert listed.status_code == 200
    assert len(listed.json()["overrides"]) == 1

    deleted = api_client.delete("/api/overrides/clip-1")
    assert deleted.status_code == 200
    assert deleted.json()["overrides"] == []

    overrides_path = tmp_path / "work" / "overrides.jsonl"
    assert not overrides_path.is_file()


def test_overrides_persist_to_jsonl(api_client: TestClient, tmp_path: Path) -> None:
    api_client.put(
        "/api/overrides",
        json={"clip_id": "clip-2", "action": "force_exclude"},
    )
    overrides_path = tmp_path / "work" / "overrides.jsonl"
    assert overrides_path.is_file()
    lines = overrides_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["clip_id"] == "clip-2"
    assert payload["action"] == "force_exclude"
