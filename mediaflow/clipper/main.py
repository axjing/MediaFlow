"""Thin ``mediaflow clipper`` CLI that delegates to :mod:`mediaflow.clipper.api`.

Kept for backwards compatibility with earlier versions of the project that
called ``python -m mediaflow.clipper.main``. The actual implementation lives
in :mod:`mediaflow.clipper.api` so both the CLI and the programmatic Python
API share the same code path.
"""
from __future__ import annotations

import sys

from mediaflow.clipper.api import main as api_main


def main(argv=None):
    """CLI entry point for the clipper pipeline."""
    return api_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
