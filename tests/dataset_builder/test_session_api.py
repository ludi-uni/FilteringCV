from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app

MINIMAL = """
schema_version: 2
input:
  corpus_root: .
  clip_tsv: validated.tsv
dataset_builder:
  enabled: true
  work_dir: work
""".strip() + "\n"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "example.yaml").write_text(MINIMAL, encoding="utf-8")
    return tmp_path


def test_unbound_session_and_jobs_503(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        s = client.get("/api/session")
        assert s.status_code == 200
        assert s.json()["bound"] is False
        assert client.get("/api/jobs").status_code == 503
        assert client.get("/api/dashboard").status_code == 503


def test_list_configs_includes_example(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        resp = client.get("/api/session/configs")
        assert resp.status_code == 200
        paths = {c["path"] for c in resp.json()["configs"]}
        assert "config/example.yaml" in paths


def test_create_bind_unbind(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        created = client.post(
            "/api/session/create",
            json={"path": "config/default.yaml", "overwrite": False},
        )
        assert created.status_code == 200
        assert created.json()["bound"] is True
        assert (project / "config" / "default.yaml").is_file()
        assert client.get("/api/dashboard").status_code == 200

        again = client.post(
            "/api/session/create",
            json={"path": "config/default.yaml", "overwrite": False},
        )
        assert again.status_code == 409

        unbound = client.post("/api/session/unbind")
        assert unbound.status_code == 200
        assert unbound.json()["bound"] is False
        assert client.get("/api/jobs").status_code == 503


def test_bind_invalid_yaml_400(project: Path) -> None:
    bad = project / "config" / "bad.yaml"
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    with TestClient(create_app(None, project)) as client:
        resp = client.post("/api/session/bind", json={"path": "config/bad.yaml"})
        assert resp.status_code == 400


def test_bind_invalid_preserves_existing_session(project: Path) -> None:
    cfg = project / "config" / "default.yaml"
    cfg.write_text(MINIMAL, encoding="utf-8")
    bad = project / "config" / "bad.yaml"
    bad.write_text("dataset_builder: []\n", encoding="utf-8")
    with TestClient(create_app(cfg, project)) as client:
        assert client.get("/api/dashboard").status_code == 200
        resp = client.post("/api/session/bind", json={"path": "config/bad.yaml"})
        assert resp.status_code == 400
        status = client.get("/api/session")
        assert status.status_code == 200
        assert status.json()["bound"] is True
        assert status.json()["config_path"] == "config/default.yaml"
        assert client.get("/api/dashboard").status_code == 200


def test_path_traversal_rejected(project: Path) -> None:
    with TestClient(create_app(None, project)) as client:
        resp = client.post("/api/session/bind", json={"path": "../outside.yaml"})
        assert resp.status_code in (400, 403)


def test_unbind_blocked_when_job_active(project: Path) -> None:
    cfg = project / "config" / "default.yaml"
    cfg.write_text(MINIMAL, encoding="utf-8")
    with TestClient(create_app(cfg, project)) as client:
        from unittest.mock import patch
        with patch("cv_preprocess.jobs.runner.JobRunner.start_job"):
            job = client.post("/api/jobs", json={"job_type": "scan", "force": False})
            assert job.status_code == 200
        # leave job queued
        resp = client.post("/api/session/unbind")
        assert resp.status_code == 409


def test_bind_blocked_when_job_active(project: Path) -> None:
    cfg = project / "config" / "default.yaml"
    cfg.write_text(MINIMAL, encoding="utf-8")
    other = project / "config" / "other.yaml"
    other.write_text(MINIMAL, encoding="utf-8")
    with TestClient(create_app(cfg, project)) as client:
        from unittest.mock import patch
        with patch("cv_preprocess.jobs.runner.JobRunner.start_job"):
            job = client.post("/api/jobs", json={"job_type": "scan", "force": False})
            assert job.status_code == 200
        resp = client.post("/api/session/bind", json={"path": "config/other.yaml"})
        assert resp.status_code == 409


def test_create_blocked_when_job_active(project: Path) -> None:
    cfg = project / "config" / "default.yaml"
    cfg.write_text(MINIMAL, encoding="utf-8")
    other = project / "config" / "other.yaml"
    with TestClient(create_app(cfg, project)) as client:
        from unittest.mock import patch
        with patch("cv_preprocess.jobs.runner.JobRunner.start_job"):
            job = client.post("/api/jobs", json={"job_type": "scan", "force": False})
            assert job.status_code == 200
        assert not other.exists()
        resp = client.post(
            "/api/session/create",
            json={"path": "config/other.yaml", "overwrite": False},
        )
        assert resp.status_code == 409
        assert not other.exists()
        assert client.get("/api/session").json()["bound"] is True
        assert client.get("/api/session").json()["config_path"] == "config/default.yaml"
