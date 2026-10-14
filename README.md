# image-to-text

A backend OCR service for product packaging. Send a photo of a label, check that
the photo is readable, extract the text, and get it back as structured lines
with confidences and positions.

> **Status:** project skeleton. `GET /health` works. The OCR engines, an image
> quality gate and text extraction are next.

```
backend/    Python 3.12 · uv · FastAPI      GET /health          :8100
RULES.md    project working agreement (read first)
```

Backend package `image_to_text` (`backend/src/`):

```
api/        app.py: FastAPI app factory and routes
ocr/        base.py: engine interface and result shapes (Line, Word, EngineResult)
config.py   settings, read from env
```

## Run it

Prerequisite: Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/).

```powershell
cd backend
uv sync --extra dev
uv run pytest
uv run uvicorn image_to_text.api.app:app --port 8100
```

```powershell
curl http://127.0.0.1:8100/health
```

## Settings (env)

Copy `.env.example` to `.env` to override defaults.

| Var | Default | Meaning |
|---|---|---|
| `ITT_CORS_ORIGINS` | *(empty: no browser origins allowed)* | comma-separated allowed origins |
| `ITT_MAX_UPLOAD_BYTES` | `12582912` (12 MiB) | upload size limit |
| `ITT_LOG_LEVEL` | `INFO` | Python log level |

## Development

```powershell
cd backend
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest --cov          # tests + coverage (target 80 %)
```

Before contributing, read [RULES.md](RULES.md). Commits and pushes happen only
with explicit approval.
