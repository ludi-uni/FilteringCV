from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app


def _client(tmp_path: Path) -> TestClient:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
  max_clips: 10
speakers:
  include_client_ids: []
dataset_builder:
  enabled: true
  work_dir: work
  target_duration_hours: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return TestClient(create_app(config_path, tmp_path))


def test_get_config(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.get("/api/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["relative_path"].endswith("config.yaml")
        assert body["data"]["dataset_builder"]["enabled"] is True
        assert "input" in {s["id"] for s in body["sections"]}


def test_validate_and_save_config_overwrites_yaml(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        get_body = client.get("/api/config").json()
        data = get_body["data"]
        data["input"]["max_clips"] = 25
        data["speakers"]["include_client_ids"] = ["speaker_a"]

        bad = client.post(
            "/api/config/validate",
            json={"data": {**data, "dataset_builder": {**data["dataset_builder"], "target_duration_hours": -1}}},
        )
        assert bad.status_code == 200
        assert bad.json()["ok"] is False

        ok = client.post("/api/config/validate", json={"data": data})
        assert ok.status_code == 200
        assert ok.json()["ok"] is True

        save = client.put("/api/config", json={"data": data, "mode": "data"})
        assert save.status_code == 200
        assert save.json()["ok"] is True

        text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert "max_clips: 25" in text
        assert "speaker_a" in text

        again = client.get("/api/config").json()
        assert again["data"]["input"]["max_clips"] == 25
