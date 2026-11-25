import os

import pytest

from image_to_text.config import ENV_PREFIX, OTEL_ENDPOINT_ENV, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if n.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(name)
    monkeypatch.delenv(OTEL_ENDPOINT_ENV, raising=False)


def test_tracing_is_off_unless_the_otel_endpoint_is_set(monkeypatch):
    assert Settings.from_env().otel_endpoint is None
    monkeypatch.setenv(OTEL_ENDPOINT_ENV, " http://collector:4318 ")
    assert Settings.from_env().otel_endpoint == "http://collector:4318"


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


def test_journey_is_off_by_default():
    s = Settings.from_env()
    assert s.journey_enabled is False
    assert s.journey_photo_dir is None
    assert s.journey_photo_retention_days == 14
    assert s.journey_max_body_bytes == 512 * 1024
    assert s.journey_view_password is None
    assert s.journey_photo_max_mb == 2048


def test_reads_journey_settings(monkeypatch):
    monkeypatch.setenv("ITT_JOURNEY_ENABLED", "true")
    monkeypatch.setenv("ITT_JOURNEY_PHOTO_DIR", "/data/journey-photos")
    monkeypatch.setenv("ITT_JOURNEY_PHOTO_RETENTION_DAYS", "7")
    monkeypatch.setenv("ITT_JOURNEY_MAX_BODY_BYTES", "1000")
    monkeypatch.setenv("ITT_JOURNEY_VIEW_USER", "ops")
    monkeypatch.setenv("ITT_JOURNEY_VIEW_PASSWORD", "a-long-password")

    s = Settings.from_env()

    assert s.journey_enabled is True
    assert s.journey_photo_dir == "/data/journey-photos"
    assert s.journey_photo_retention_days == 7
    assert s.journey_max_body_bytes == 1000
    assert s.journey_view_user == "ops"
    assert s.journey_view_password == "a-long-password"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ITT_JOURNEY_PHOTO_RETENTION_DAYS", "0"),
        ("ITT_JOURNEY_MAX_BODY_BYTES", "-1"),
        ("ITT_JOURNEY_VIEW_PASSWORD", "short"),
        ("ITT_JOURNEY_PHOTO_MAX_MB", "0"),
    ],
)
def test_invalid_journey_settings_name_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()
