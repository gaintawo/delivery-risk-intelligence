"""Privacy audit logging.

Every read, write, transformation and PII action in the pipeline is written as
one JSON line to logs/audit.log: who did it, when, to which dataset, how many
rows and columns, a SHA-256 fingerprint of any file touched, and why. The log
is append-only by convention and is kept for 24 months (docs/data_governance.md).
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from . import config

RUN_ID = uuid.uuid4().hex[:12]
_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("dri.audit")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(config.LOG_DIR / "audit.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _logger = logger
    return _logger


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(action: str, dataset: str, *, purpose: str, rows: int | None = None,
          cols: int | None = None, path: Path | None = None, **details) -> dict:
    """Record one auditable event and return it.

    action: read | write | transform | pii_remove | pii_pseudonymise | validate | bias_check
    """
    event = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "actor": config.ACTOR,
        "action": action,
        "dataset": dataset,
        "rows": rows,
        "cols": cols,
        "purpose": purpose,
    }
    if path is not None and Path(path).exists():
        event["file"] = str(Path(path).relative_to(config.ROOT)) if str(path).startswith(str(config.ROOT)) else str(path)
        event["sha256"] = file_sha256(Path(path))
    if details:
        event["details"] = details
    _get_logger().info(json.dumps(event, default=str))
    return event
