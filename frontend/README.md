# FilteringCV GUI

React + TypeScript + Vite frontend for the FilteringCV dataset builder ops dashboard.

## Development

Run the FastAPI backend on port **8765**, then start the Vite dev server:

```bash
# Terminal 1 — backend (from repo root)
python -m cv_preprocess.web  # or your existing serve command

# Terminal 2 — frontend
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:5173. Vite proxies `/api` and `/ws` to `http://127.0.0.1:8765`.

## Production

Build static assets and let FastAPI serve them from `frontend/dist`:

```bash
cd frontend
pnpm install
pnpm build
```

`cv_preprocess.web.app.create_app` mounts `frontend/dist` at `/` when that directory exists. Run only the backend; no separate frontend server is required.

## API alignment notes

The UI matches the FastAPI routes under `cv_preprocess/web/routes/`:

| Screen   | Endpoint |
|----------|----------|
| Dashboard | `GET /api/dashboard` |
| Jobs | `GET/POST /api/jobs`, `POST /api/jobs/{id}/cancel`, WS `/ws/jobs/{id}` |
| Coverage | `GET /api/reports/coverage` |
| Clips | `GET /api/catalog/clips` (`page` / `page_size`, not `offset` / `limit`) |
| Audio | `GET /api/audio/{relative_path}` |
| Overrides | `PUT /api/overrides` (not POST) |
| Compare | `POST /api/compare` with `{ left, right }` |

Quality score filtering is applied client-side on the current page only; the catalog API does not expose a `quality` query parameter.
