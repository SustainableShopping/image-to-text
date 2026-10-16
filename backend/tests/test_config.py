import os

import pytest

from image_to_text.config import ENV_PREFIX, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)


def test_reads_engine_settings(monkeypatch):
    monkeypatch.setenv("ITT_DEFAULT_ENGINE", "RapidOCR")
    monkeypatch.setenv("ITT_USE_ANGLE_CLS", "yes")
    monkeypatch.setenv("ITT_WARM_UP", "off")
    monkeypatch.setenv("ITT_ONNX_THREADS", "4")
    monkeypatch.setenv("ITT_TEXT_SCORE", "0.6")

    s = Settings.from_env()

    assert s.default_engine == "rapidocr"
    assert s.use_angle_cls is True
    assert s.warm_up is False
    assert s.onnx_threads == 4
    assert s.text_score == 0.6


@pytest.mark.parametrize(
    ("name", "value"),
    [("ITT_WARM_UP", "maybe"), ("ITT_TEXT_SCORE", "high"), ("ITT_TEXT_SCORE", "1.5")],
)
def test_invalid_engine_settings_name_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_rejects_non_positive_threads():
    with pytest.raises(ValueError, match="ITT_ONNX_THREADS"):
        Settings(onnx_threads=0)


def test_defaults_when_env_is_empty():
    s = Settings.from_env()
    assert s == Settings()
    assert s.cors_origins == ()  # no browser origins unless configured
    assert s.max_upload_bytes == 12 * 1024 * 1024
    assert s.log_level == "INFO"


def test_reads_and_normalises_env(monkeypatch):
    monkeypatch.setenv("ITT_CORS_ORIGINS", " https://a.example , ,https://b.example ")
    monkeypatch.setenv("ITT_MAX_UPLOAD_BYTES", "1024")
    monkeypatch.setenv("ITT_LOG_LEVEL", "debug")

    s = Settings.from_env()

    assert s.cors_origins == ("https://a.example", "https://b.example")
    assert s.max_upload_bytes == 1024
    assert s.log_level == "DEBUG"


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("ITT_CORS_ORIGINS", " , ")
    monkeypatch.setenv("ITT_MAX_UPLOAD_BYTES", "  ")

    s = Settings.from_env()

    assert s.cors_origins == Settings().cors_origins
    assert s.max_upload_bytes == Settings().max_upload_bytes


def test_non_integer_upload_limit_names_the_variable(monkeypatch):
    monkeypatch.setenv("ITT_MAX_UPLOAD_BYTES", "12MB")
    with pytest.raises(ValueError, match="ITT_MAX_UPLOAD_BYTES"):
        Settings.from_env()


def test_rejects_non_positive_upload_limit():
    with pytest.raises(ValueError, match="ITT_MAX_UPLOAD_BYTES"):
        Settings(max_upload_bytes=0)


def test_rejects_unknown_log_level(monkeypatch):
    monkeypatch.setenv("ITT_LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError, match="ITT_LOG_LEVEL"):
        Settings.from_env()


def test_settings_are_immutable():
    s = Settings()
    with pytest.raises(AttributeError):
        s.log_level = "DEBUG"  # type: ignore[misc]
