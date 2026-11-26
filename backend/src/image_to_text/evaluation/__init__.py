"""OCR evaluation: run the engines on label photos and score their text against the
ingredient section written out by hand, for accuracy (CER, WER, word recall) and speed.

    uv run python -m image_to_text.evaluation --images <photo folder>

Each case is a `<id>.case.json` file (`id`, `image`, `ocr_text`, optional `tags`).
Other keys in it (expected ingredients for later pipeline steps) are accepted unread.
The results file (one JSON line per case and engine, with the OCR text) is the input
for scoring those later steps on real OCR output.
"""
