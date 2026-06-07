"""Thin ``mediaflow tts`` CLI that delegates to :mod:`mediaflow.tts.api`.

Kept for backwards compatibility with earlier versions of the project that
called ``python -m mediaflow.tts.cli``. The actual implementation lives in
:mod:`mediaflow.tts.api` so both the CLI and the programmatic Python API
share the same code path.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(argv=None):
    """CLI entry point for the IndexTTS pipeline."""
    from mediaflow.tts.api import main as api_main

    return api_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
