"""
tests/test_model_security.py — Model Serialization Security Tests
=================================================================
Validates SHA-256 checksum verification, path traversal prevention,
and unsigned model rejection.

Run with:
    pytest tests/test_model_security.py -v
"""

import sys
import os
import pickle
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insight_model import (
    load_insight_ranker,
    _compute_checksum,
    _verify_checksum,
    _validate_model_path,
    ModelSecurityError,
)


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory with a fake signed model."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create a minimal pickle file (just a dict, not a real pipeline)
    model_path = models_dir / "insight_ranker.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"fake": "model"}, f)

    # Create valid checksum
    checksum = _compute_checksum(str(model_path))
    checksum_path = models_dir / "insight_ranker.pkl.sha256"
    checksum_path.write_text(checksum)

    return models_dir


class TestChecksumComputation:

    def test_checksum_is_hex_string(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = _compute_checksum(str(f))
        assert len(result) == 64  # SHA-256 hex is 64 characters
        assert all(c in "0123456789abcdef" for c in result)

    def test_checksum_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"deterministic content")
        assert _compute_checksum(str(f)) == _compute_checksum(str(f))

    def test_different_content_different_checksum(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert _compute_checksum(str(f1)) != _compute_checksum(str(f2))


class TestChecksumVerification:

    def test_valid_checksum_passes(self, tmp_path):
        f = tmp_path / "model.pkl"
        f.write_bytes(b"valid model content")
        checksum = _compute_checksum(str(f))
        cs_file = tmp_path / "model.pkl.sha256"
        cs_file.write_text(checksum)
        assert _verify_checksum(str(f), str(cs_file)) is True

    def test_tampered_checksum_raises(self, tmp_path):
        f = tmp_path / "model.pkl"
        f.write_bytes(b"valid model content")
        cs_file = tmp_path / "model.pkl.sha256"
        cs_file.write_text("0" * 64)  # fake checksum
        with pytest.raises(ModelSecurityError, match="integrity check FAILED"):
            _verify_checksum(str(f), str(cs_file))


class TestPathValidation:

    def test_valid_model_path_accepted(self):
        # The default path should resolve within models/
        # We test with a path that would be valid
        from insight_model import _MODELS_DIR
        fake_path = os.path.join(_MODELS_DIR, "some_model.pkl")
        result = _validate_model_path(fake_path)
        assert os.path.realpath(fake_path) == result

    def test_path_traversal_rejected(self):
        with pytest.raises(ModelSecurityError, match="outside the allowed"):
            _validate_model_path("../../etc/passwd")

    def test_symlink_resolved(self, tmp_path):
        """Symlinks pointing outside models dir should be rejected."""
        from insight_model import _MODELS_DIR
        # Create a symlink in models/ pointing to /tmp
        target = tmp_path / "evil_model.pkl"
        target.write_bytes(b"evil")

        link_path = os.path.join(_MODELS_DIR, "__test_symlink__.pkl")
        try:
            os.makedirs(_MODELS_DIR, exist_ok=True)
            if os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(str(target), link_path)
            with pytest.raises(ModelSecurityError, match="outside the allowed"):
                _validate_model_path(link_path)
        finally:
            if os.path.exists(link_path):
                os.remove(link_path)


class TestLoadInsightRanker:

    def test_load_rejects_missing_checksum(self, tmp_path, monkeypatch):
        """Model exists but no .sha256 → returns None."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "insight_ranker.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"fake": True}, f)

        # Monkeypatch the _MODELS_DIR to point to our temp
        import insight_model
        monkeypatch.setattr(insight_model, "_MODELS_DIR", str(models_dir))

        result = load_insight_ranker(str(model_file))
        assert result is None

    def test_load_rejects_tampered_model(self, tmp_path, monkeypatch):
        """Model + checksum exist, but content was tampered → raises."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "insight_ranker.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"original": True}, f)

        # Write a valid checksum, then tamper the file
        checksum = _compute_checksum(str(model_file))
        cs_file = models_dir / "insight_ranker.pkl.sha256"
        cs_file.write_text(checksum)

        # Tamper the model
        with open(model_file, "wb") as f:
            pickle.dump({"tampered": True}, f)

        import insight_model
        monkeypatch.setattr(insight_model, "_MODELS_DIR", str(models_dir))

        with pytest.raises(ModelSecurityError, match="integrity check FAILED"):
            load_insight_ranker(str(model_file))

    def test_load_accepts_valid_signed_model(self, tmp_path, monkeypatch):
        """Valid model + valid checksum → returns loaded object."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "insight_ranker.pkl"
        fake_pipeline = {"valid": True}
        with open(model_file, "wb") as f:
            pickle.dump(fake_pipeline, f)

        checksum = _compute_checksum(str(model_file))
        cs_file = models_dir / "insight_ranker.pkl.sha256"
        cs_file.write_text(checksum)

        import insight_model
        monkeypatch.setattr(insight_model, "_MODELS_DIR", str(models_dir))

        result = load_insight_ranker(str(model_file))
        assert result is not None
        assert result == fake_pipeline

    def test_load_missing_file_returns_none(self):
        """Non-existent model path → returns None."""
        result = load_insight_ranker("models/nonexistent_model.pkl")
        assert result is None
