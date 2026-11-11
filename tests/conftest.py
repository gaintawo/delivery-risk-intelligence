import pytest

from src.pipeline import audit as audit_mod
from src.pipeline import config
from tests.synthetic import make_raw


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Every test writes logs and outputs to a temporary folder, never the repo."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(audit_mod, "_logger", None)
    import logging
    logging.getLogger("dri.audit").handlers.clear()
    yield


@pytest.fixture
def raw():
    return make_raw()
