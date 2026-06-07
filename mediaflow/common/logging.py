"""Unified logging facade for MediaFlow.

Replaces the old :mod:`clipperX.common.log_wrappers` module. The log file
location can be overridden via the ``MEDIAFLOW_LOG_DIR`` environment variable;
it defaults to ``<project_root>/log`` (i.e. ``Path(__file__).parents[2]/log``).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

_DEFAULT_FORMAT = (
    "[%(asctime)s] PID:%(process)d %(filename)s->%(funcName)s "
    "line:%(lineno)d [%(levelname)s] %(message)s"
)

_project_root = Path(__file__).resolve().parents[2]


def _resolve_log_dir() -> Path:
    """Resolve the log output directory.

    Priority:
        1. ``MEDIAFLOW_LOG_DIR`` environment variable.
        2. ``<project_root>/log`` (two parents above this file).
    """
    override = os.environ.get("MEDIAFLOW_LOG_DIR")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = (_project_root / "log").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class Logging:
    """Lightweight logger factory used across the project.

    Each :class:`Logging` instance configures a file handler and a console
    handler on the named logger. Multiple instances for the same name reuse
    the existing logger to avoid duplicate handler attachment.
    """

    def __init__(self, logger_name: str | None = None, log_cate: str = "mediaflow"):
        self.logger = logging.getLogger(logger_name or "mediaflow")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        self.log_dir = _resolve_log_dir()
        self.log_filename = self.log_dir / f"{log_cate}_{time.strftime('%Y_%m_%d')}.log"

        formatter = logging.Formatter(_DEFAULT_FORMAT)

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(self.log_filename, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Return the configured :class:`logging.Logger` instance."""
        return self.logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Convenience wrapper returning a logger for the given module name."""
    return Logging(name).get_logger()


if __name__ == "__main__":
    logger = Logging(__name__).get_logger()
    logger.info("logging initialised")
