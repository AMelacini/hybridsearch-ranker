import importlib


def test_config_accepts_legacy_and_standard_env_names(monkeypatch) -> None:
    monkeypatch.setenv("REST_SERVER_PORT", "9000")
    monkeypatch.setenv("UI_PORT", "9001")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379")
    monkeypatch.delenv("HSR_REST_SERVER_PORT", raising=False)
    monkeypatch.delenv("HSR_UI_PORT", raising=False)
    monkeypatch.delenv("HSR_REDIS_URL", raising=False)

    import hsr_config

    importlib.reload(hsr_config)

    assert hsr_config.REST_SERVER_PORT == 9000
    assert hsr_config.UI_PORT == 9001
    assert hsr_config.REDIS_URL == "redis://redis:6379"
