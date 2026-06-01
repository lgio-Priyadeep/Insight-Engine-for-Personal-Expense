import os
import threading
import pytest

# FIX M1: REMOVED module-level os.environ mutations.
# Previously: os.environ.setdefault("ENV", "test")
# If CI has ENV=production already set, setdefault is a no-op.
# _ensure_initialized then rejects SKIP_STARTUP_CHECKS, breaking all tests.
# Now uses a session-scoped autouse fixture that saves/restores original values.

@pytest.fixture(autouse=True, scope="session")
def _set_test_env():
    """Force test environment for entire session, restore originals on teardown."""
    # C1: Save and forcibly set BOTH env vars. Do not rely on CI or developer
    # shell state: if ENV is already 'production', SKIP_STARTUP_CHECKS would be
    # rejected by _ensure_initialized, breaking the entire test session.
    # Setting ENV=test here overrides any ambient value for the session.
    _keys = ("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", "ENV")
    original = {k: os.environ.get(k) for k in _keys}
    os.environ["INSIGHT_ENGINE_SKIP_STARTUP_CHECKS"] = "true"
    os.environ["ENV"] = "test"

    # C1: Reset passion_pipeline init state NOW, after env vars are set.
    # Stale _init_complete from a previous session would cause _ensure_initialized
    # to skip the env-var checks entirely and use whatever state was cached.
    import sys
    if "passion_pipeline" in sys.modules:
        import passion_pipeline as _pp
        _pp._init_complete.clear()
        _pp._init_failed.clear()

    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# P1.8 + P2.8: Reset threading.Event for _init_complete between tests.
# FIX M2: Also reset _init_lock. If a test crashes while holding the lock,
# subsequent tests would deadlock without this reset.
# FIX-10: Also reset _init_failed event.
# FIX 5: Reset _init_in_progress to False to prevent thread state contamination.
@pytest.fixture(autouse=True)
def _reset_pipeline_initialized(monkeypatch):
    import sys
    if "passion_pipeline" in sys.modules:
        monkeypatch.setattr("passion_pipeline._init_complete", threading.Event())
        monkeypatch.setattr("passion_pipeline._init_lock", threading.Lock())
        monkeypatch.setattr("passion_pipeline._init_failed", threading.Event())
        monkeypatch.setattr("passion_pipeline._init_in_progress", False)
    yield


# FIX L5 + FIX-T1-01: _secret_cache reset between tests.
@pytest.fixture(autouse=True)
def _reset_dev_secret():
    from log_utils import _reset_secret_cache
    _reset_secret_cache()
    yield

@pytest.fixture
def real_startup_env(monkeypatch):
    monkeypatch.setenv("ENV", "test")
    monkeypatch.delenv("INSIGHT_ENGINE_SKIP_STARTUP_CHECKS", raising=False)
    from log_utils import _reset_secret_cache
    _reset_secret_cache()
    import passion_pipeline
    monkeypatch.setattr(passion_pipeline, "_init_complete", threading.Event())
    monkeypatch.setattr(passion_pipeline, "_init_failed", threading.Event())
    monkeypatch.setattr(passion_pipeline, "_init_lock", threading.Lock())
    monkeypatch.setattr(passion_pipeline, "_init_in_progress", False)
    yield
