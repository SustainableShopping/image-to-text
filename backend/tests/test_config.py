import pytest

from image_to_text.config import Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ITT_CORS_ORIGINS", "ITT_MAX_UPLOAD_BYTES", "ITT_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)


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
