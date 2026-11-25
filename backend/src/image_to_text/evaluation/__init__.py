"""OCR evaluation: run the engines on label photos and score their text against the
ingredient section written out by hand, for accuracy (CER, WER, word recall) and speed.

    uv run python -m image_to_text.evaluation --images <photo folder>

Each case is a `<id>.case.json` file (`id`, `image`, `ocr_text`, optional `tags`); the
same file can also hold the expected ingredients, which the ingredient service's
evaluation scores. The results file (one JSON line per case and engine, with the OCR
text) is that evaluation's `--ocr-results` input, so the later steps can be scored on
real OCR output.
"""
