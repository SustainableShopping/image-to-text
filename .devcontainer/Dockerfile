# Images pinned by digest so every build gets the same bytes; Dependabot
# (.github/dependabot.yml) proposes updates.
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm@sha256:7876580dc67fd460fd962f004cbeb480027e9bbc0657096f1087db11f9eaff39

# Same uv as the one that wrote backend/uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.11.16@sha256:440fd6477af86a2f1b38080c539f1672cd22acb1b1a47e321dba5158ab08864d /uv /uvx /usr/local/bin/

# OpenCV (pulled in by rapidocr) needs libGL and GLib at runtime.
# The base image's Yarn apt source has an expired key and breaks apt-get update.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pre-create mount targets so named volumes are owned by the dev user
RUN mkdir -p /home/vscode/.cache/onnxtr /home/vscode/.venvs \
    && chown -R vscode:vscode /home/vscode/.cache /home/vscode/.venvs
