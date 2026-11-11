"""Structured (JSON-lines) logging shared by the offline pipeline and the API.

Each record is one JSON object per line: timestamp (UTC, ISO-8601), level, logger,
event, plus arbitrary structured fields passed via ``extra={"fields": {...}}`` or the
``log_event`` helper. Only the standard library is used so the downloader can run before
the project environment exists.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger emitting JSON lines to stderr and, optionally, appending to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = JsonFormatter()
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_pdeflow", False) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        sh._pdeflow = True  # type: ignore[attr-defined]
        logger.addHandler(sh)
    if log_file is not None:
        path = Path(log_file).resolve()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path for h in logger.handlers):
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})
