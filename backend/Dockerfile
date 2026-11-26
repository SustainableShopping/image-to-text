# Production image (build context: backend/). Images pinned by digest.
FROM ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f AS uv

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Dependencies first so they stay cached when only the source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY src ./src
RUN uv sync --frozen --no-editable

FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
# OpenCV (pulled in by rapidocr) needs libGL and GLib at runtime. /data/journey-photos
# is owned by `app` so a volume mounted there (ITT_JOURNEY_PHOTO_DIR) is writable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /data/journey-photos && chown app /data/journey-photos
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ONNXTR_CACHE_DIR=/models
# Download every model now (docTR into /models, RapidOCR into its package dir,
# angle classifier included) so containers start without network access or
# writable model directories.
RUN python -c "from image_to_text.config import Settings; \
from image_to_text.ocr.router import EngineRegistry; \
r = EngineRegistry(Settings(use_angle_cls=True)); \
[r.get(name) for name in r.names]"
USER app
EXPOSE 8100
# start-period covers model loading + warm-up inference
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=2)"]
# Only the reverse proxy (Traefik) can reach the container, so forwarded headers are trusted.
CMD ["uvicorn", "image_to_text.api.app:app", "--host", "0.0.0.0", "--port", "8100", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
