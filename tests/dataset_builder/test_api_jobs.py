from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cv_preprocess.jobs.models import JobStatus, JobType
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
    app = create_app(config_path, tmp_path)
    with TestClient(app) as client:
        yield client


def test_create_list_and_get_job(api_client: TestClient) -> None:
    with patch("cv_preprocess.jobs.runner.JobRunner.start_job") as start_job:
        create_resp = api_client.post(
            "/api/jobs",
            json={"job_type": "analyze", "force": False},
        )
        assert create_resp.status_code == 200
        payload = create_resp.json()
        assert payload["job_type"] == "analyze"
        assert payload["status"] == JobStatus.QUEUED.value
        job_id = payload["id"]
        start_job.assert_called_once_with(job_id)

    list_resp = api_client.get("/api/jobs")
    assert list_resp.status_code == 200
    jobs = list_resp.json()
    assert len(jobs) == 1
    assert jobs[0]["id"] == job_id

    status_resp = api_client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == JobStatus.QUEUED.value


def test_job_not_found(api_client: TestClient) -> None:
    resp = api_client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404


def test_create_coverage_build_job(api_client: TestClient) -> None:
    with patch("cv_preprocess.jobs.runner.JobRunner.start_job"):
        resp = api_client.post(
            "/api/jobs",
            json={"job_type": JobType.COVERAGE_BUILD.value, "force": False},
        )
    assert resp.status_code == 200
    assert resp.json()["job_type"] == JobType.COVERAGE_BUILD.value


def test_coverage_automation_report_endpoint(api_client: TestClient) -> None:
    resp = api_client.get("/api/reports/coverage-automation")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is True
    assert payload["enabled"] is False
    assert "run_dir" in payload
