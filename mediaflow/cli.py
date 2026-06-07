"""Unified command-line interface for MediaFlow.

Top-level dispatcher that routes ``mediaflow tts`` and ``mediaflow clipper``
subcommands to the per-subpackage entry points. This module is exposed as the
``mediaflow`` console script in :file:`pyproject.toml` and is also the
``__main__`` for ``python -m mediaflow``.

Usage::

    mediaflow tts "hello" -v voice.wav -o out.wav
    mediaflow clipper input.mp4 --cfg=config
    mediaflow --help

For ergonomic one-shot invocation the dispatcher defaults the first
positional argument to the ``tts`` subcommand when the user forgets to type
``tts`` explicitly. This is the common case for the project's headline
feature::

    mediaflow "hello" -v voice.wav -o out.wav        # equivalent to mediaflow tts ...
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional, Sequence

SUBCOMMANDS = ("tts", "clipper")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands wired in."""
    parser = argparse.ArgumentParser(
        prog="mediaflow",
        description=(
            "MediaFlow — unified toolkit for TTS, voice cloning, "
            "subtitle generation, filler-word cutting, and bilingual translation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<subcommand>")

    tts_parser = subparsers.add_parser(
        "tts",
        help="Run the IndexTTS text-to-speech pipeline.",
        description="Synthesise speech from text using IndexTTS / IndexTTS2.",
        add_help=False,
    )
    tts_parser.set_defaults(_handler=_run_tts)

    clipper_parser = subparsers.add_parser(
        "clipper",
        help="Run the clipper pipeline (transcribe, cut, daemon, to_md, srt_to_compact).",
        description=(
            "Generate subtitles, edit markdown task lists, and cut video/audio "
            "clips automatically with the MediaFlow clipper."
        ),
        add_help=False,
    )
    clipper_parser.set_defaults(_handler=_run_clipper)

    return parser


def _run_tts(argv: Sequence[str]) -> int:
    """Dispatch to :mod:`mediaflow.tts.api.main` with the subcommand stripped."""
    from mediaflow.tts.api import main as tts_main

    return tts_main(argv)


def _run_clipper(argv: Sequence[str]) -> int:
    """Dispatch to :mod:`mediaflow.clipper.api.main` with the subcommand stripped."""
    from mediaflow.clipper.api import main as clipper_main

    return clipper_main(argv)


_HANDLERS = {
    "tts": _run_tts,
    "clipper": _run_clipper,
}


def _looks_like_subcommand(token: str) -> bool:
    """Return ``True`` when ``token`` is an explicit MediaFlow subcommand."""
    return token in SUBCOMMANDS


def _looks_like_flag(token: str) -> bool:
    """Return ``True`` for tokens that begin with a flag marker."""
    return token.startswith("-") and token not in ("-", "--")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point. Returns a POSIX exit code."""
    raw: List[str] = list(argv) if argv is not None else list(sys.argv[1:])

    if not raw:
        build_parser().print_help()
        return 0

    if raw[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    if _looks_like_subcommand(raw[0]):
        handler = _HANDLERS[raw[0]]
        forwarded = [tok for tok in raw[1:] if tok != raw[0]]
        return handler(forwarded)

    if not _looks_like_flag(raw[0]):
        # Common ergonomic shortcut: `mediaflow "text" -v voice.wav`
        # is treated as `mediaflow tts "text" -v voice.wav`.
        return _run_tts(raw)

    # First token is a flag without a subcommand → show help.
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
