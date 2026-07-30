from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cv_preprocess.web.app import create_app


def _write_frontend_dist(project: Path) -> None:
    dist = project / "frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>spa</div></body></html>\n",
        encoding="utf-8",
    )
    (assets / "app.js").write_text("console.log('ok')\n", encoding="utf-8")


def test_spa_fallback_serves_index_for_client_routes(tmp_path: Path) -> None:
    _write_frontend_dist(tmp_path)
    cfg = tmp_path / "config" / "default.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
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

    with TestClient(create_app(cfg, tmp_path)) as client:
        for path in ("/", "/config", "/jobs", "/coverage", "/clips", "/compare"):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert "spa" in resp.text
            assert "text/html" in resp.headers.get("content-type", "")

        asset = client.get("/assets/app.js")
        assert asset.status_code == 200
        assert "console.log" in asset.text

        # API routes must still win over the SPA catch-all.
        session = client.get("/api/session")
        assert session.status_code == 200
        assert "bound" in session.json()
