import os, importlib
import pytest
from types import ModuleType

# helper to reload the config module with a clean cache
def fresh_config() -> ModuleType:
    cfg = importlib.import_module("rag_evaluation.config")
    importlib.reload(cfg)
    return cfg

def test_cache_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    cfg = fresh_config()

    # set one-time key and assert we get it back
    cfg.set_api_key("openai", "cache-key")
    assert cfg.get_api_key("openai") == "cache-key"

def test_env_used_when_no_cache(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-gem")
    cfg = fresh_config()

    assert cfg.get_api_key("gemini") == "env-gem"

def test_error_when_missing(monkeypatch):
    # make sure .env won’t be loaded on re-import
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = fresh_config()

    with pytest.raises(ValueError):
        cfg.get_api_key("openai")


    with pytest.raises(ValueError):
        cfg.get_api_key("openai")
