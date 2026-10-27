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

## Develop in a dev container (recommended)

`.devcontainer/` defines the environment every PR is built and tested in:
Python 3.12 on Debian bookworm, uv 0.11.16, dependencies installed from
`uv.lock` with `uv sync --frozen`. CI runs the same container.

### One-command setup with DevPod

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
running and [DevPod](https://devpod.sh) installed. On Windows, DevPod Desktop
installs its CLI as `C:\Program Files\DevPod\devpod-cli.exe` (not on `PATH`);
use that path in place of `devpod` below, or add the folder to `PATH`.

Once per machine:

```powershell
devpod provider add docker
```

Then, from a clone of the repo:

```powershell
devpod up . --ide none
```

When it finishes, open **http://localhost:8100/docs** in your browser. That one
command builds the container, creates `.env` from `.env.example` (only if
`.env` doesn't exist), installs dependencies from `uv.lock`, and starts the API.
No terminal needs to stay open. The first start takes a few minutes (image
build, ~100 MB of models); later starts take seconds.

| Task | Command |
|---|---|
| Open a shell in the workspace | `devpod ssh image-to-text` (repo at `/workspaces/image-to-text`) |
| Server log | `devpod ssh image-to-text --command "tail -f /tmp/itt-server.log"` |
| Apply `.env` changes | `devpod ssh image-to-text --command "bash /workspaces/image-to-text/.devcontainer/start-server.sh --restart"` |
| Stop / start again | `devpod stop image-to-text` / `devpod up . --ide none` (the API starts automatically) |
| Use another port (e.g. 8100 is busy) | `$env:ITT_HOST_PORT="8200"; devpod up . --ide none --recreate`, then http://localhost:8200 |
| Open in VS Code instead | `devpod up . --ide vscode` |

The API is published on `127.0.0.1` only, so it isn't reachable from other
machines on your network. Settings for the API come from `.env` (see
[Settings](#settings-env)). The host port is read from `ITT_HOST_PORT` in your
shell when the workspace is created, because DevPod sets up ports before
`.env` exists.

VS Code's Dev Containers extension reads the same files. Inside the container,
the commands below work as-is (from `backend/`). The virtualenv lives outside
the repo (`UV_PROJECT_ENVIRONMENT`), and docTR models are kept in the
`itt-onnxtr-models` volume across rebuilds.

## Run it

Prerequisite: Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/), or the dev
container above.

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
itself. In the dev container, `.env` is created on first setup and loaded
automatically when the API starts. Outside it, copy `.env.example` to `.env`
(repo root) and pass it to uv:

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

CI (`.github/workflows/ci.yml`) runs lint, format check and the unit tests with
coverage inside the dev container on every PR and push to `main`. The real
model tests run only when the workflow is started manually with
`model_tests` enabled.

Before contributing, read [RULES.md](RULES.md). Commits and pushes happen only
with explicit approval.
