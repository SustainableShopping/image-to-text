# Production image (build context: backend/). Images pinned by digest.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim-bookworm@sha256:6dd8b6dab6b3adc7812bab69eca5dbdc1a5cc690ad28306d234ffb4ef8ac9753 AS build
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

FROM python:3.12-slim-bookworm@sha256:6dd8b6dab6b3adc7812bab69eca5dbdc1a5cc690ad28306d234ffb4ef8ac9753
# OpenCV (pulled in by rapidocr) needs libGL and GLib at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home app
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
