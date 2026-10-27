# image-to-text

A backend OCR service for product packaging. Send a photo of a label and get
its text back as structured lines with confidences and positions.

> **Status:** OCR works with two engines (docTR, RapidOCR). An image quality
> gate and ingredient extraction are planned.

```
backend/    Python 3.12 · uv · FastAPI      GET /health · POST /ocr     :8100
RULES.md    project working agreement (read first)
```

Backend package `image_to_text` (`backend/src/`):

```
api/        app.py: FastAPI app factory and routes
ocr/        base.py (Engine interface, Line/Word/EngineResult), router.py,
            doctr_engine.py, rapidocr_engine.py
imaging.py  upload bytes → RGB array (EXIF rotation applied)
config.py   settings, read from env
```

## Engines

Both run on CPU via ONNX Runtime, self-hosted, Apache-2.0. The engine is chosen
per request with `?engine=`.

| `engine=` | Models | Notes |
|---|---|---|
| `doctr` (default) | docTR `fast_base` + `crnn_vgg16_bn` via OnnxTR | word-level confidence; Latin script only |
| `rapidocr` | PP-OCR via RapidOCR | fastest; line-level confidence only |

The default is provisional until it has been validated on real packaging photos.
RapidOCR offers Devanagari models, but they are not configured yet; the
default PP-OCR models are used.

Models are fetched on first use: docTR ≈ 100 MB into `~/.cache/onnxtr`,
RapidOCR ≈ 30 MB into its package folder in `.venv`. With `ITT_WARM_UP=true`
(default) the default engine loads at startup; the other loads on its first
request.

## Run it

Prerequisite: Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/).

```powershell
cd backend
uv sync --extra dev
uv run pytest
uv run uvicorn image_to_text.api.app:app --port 8100
```

The server is ready when the log shows `Application startup complete`.
In another terminal (in Windows PowerShell use `curl.exe`; `curl` is an alias
for `Invoke-WebRequest`):

```powershell
curl.exe http://127.0.0.1:8100/health
curl.exe -F "image=@label.jpg" "http://127.0.0.1:8100/ocr?engine=rapidocr"
```

Interactive API docs: http://127.0.0.1:8100/docs

`POST /ocr` returns `engine`, `image` (width, height), `lines` and
`timings_ms`. Each line has `text`, `conf`, `box` (4 corners, image pixels) and
`words` (per-word confidence, docTR only; empty for RapidOCR). Errors: `400`
unknown engine, `413` upload too large, `422` not a readable image.

## Settings (env)

All settings are environment variables; the app does not read `.env` by
itself. To use a file, copy `.env.example` to `.env` (repo root) and pass it
to uv:

```powershell
uv run --env-file ../.env uvicorn image_to_text.api.app:app --port 8100
```

| Var | Default | Meaning |
|---|---|---|
| `ITT_CORS_ORIGINS` | *(empty: no browser origins allowed)* | comma-separated allowed origins |
| `ITT_MAX_UPLOAD_BYTES` | `12582912` (12 MiB) | upload size limit |
| `ITT_LOG_LEVEL` | `INFO` | log level of the `image_to_text` package |
| `ITT_DEFAULT_ENGINE` | `doctr` | `doctr` \| `rapidocr` |
| `ITT_DOCTR_DET_ARCH` / `ITT_DOCTR_RECO_ARCH` | `fast_base` / `crnn_vgg16_bn` | any OnnxTR architecture |
| `ITT_DET_LIMIT_SIDE_LEN` | `960` | RapidOCR detector max image side |
| `ITT_USE_ANGLE_CLS` | `false` | RapidOCR 180° text classifier |
| `ITT_ONNX_THREADS` | `8` | ONNX Runtime threads per engine |
| `ITT_TEXT_SCORE` | `0.5` | RapidOCR: drop lines below this confidence |
| `ITT_WARM_UP` | `true` | load and run the default engine at startup |

## Production image

```powershell
docker build -t image-to-text backend
docker run --rm -p 8100:8100 -e ITT_ONNX_THREADS=2 image-to-text
```

All OCR models (docTR, RapidOCR including the angle classifier) are downloaded
at build time and baked into the image (~1.2 GB), so containers start in
seconds with no network access and run with a read-only filesystem. The
container runs as a non-root user and has a Docker `HEALTHCHECK` on `/health`.
With both engines loaded a process uses about 1.3 GiB of RAM; set
`ITT_ONNX_THREADS` to the CPU cores available to each container.

`.github/workflows/image.yml` builds, scans (Trivy) and smoke-tests the image
on every PR. `.github/workflows/deploy.yml` pushes it to Artifact Registry and
deploys it to the GCP VM described in the `infra-platform` repo:
merges to `develop` go to the dev environment; merges to `main` go to
production only when the `PROD_ENABLED` repository variable is `true`. All
jobs are skipped until the `GCP_*` variables are set.

## Development

```powershell
cd backend
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest --cov          # unit tests + coverage (target 80 %), no models needed
$env:ITT_RUN_MODEL_TESTS="1"; uv run pytest -m models   # real engines (downloads models)
```

Before contributing, read [RULES.md](RULES.md). Commits and pushes happen only
with explicit approval.
