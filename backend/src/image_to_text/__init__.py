"""image-to-text: packaging photo → quality check → OCR → structured text.

    config.py    settings from ITT_* env vars
    api/         FastAPI app factory; GET /health
    ocr/         engine interface and result shapes

Planned: engine adapters and a router, an image quality gate, and text
extraction.
"""

SERVICE_VERSION = "0.1.0"
