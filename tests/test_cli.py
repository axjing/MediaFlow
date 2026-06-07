"""Smoke tests for the unified CLI dispatcher."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from mediaflow.cli import build_parser, main


def _run(argv):
    """Run the CLI and capture stdout. Returns (returncode, output)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            rc = main(argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 0
    return rc, buf.getvalue()


def test_top_level_help_lists_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "tts" in help_text
    assert "clipper" in help_text


def test_no_args_prints_top_level_help() -> None:
    rc, output = _run([])
    assert rc == 0
    assert "MediaFlow" in output
    assert "tts" in output
    assert "clipper" in output


def test_explicit_dash_help_prints_top_level_help() -> None:
    rc, output = _run(["--help"])
    assert rc == 0
    assert "MediaFlow" in output


def test_tts_help_delegates() -> None:
    rc, output = _run(["tts", "--help"])
    assert "IndexTTS" in output


def test_clipper_help_delegates() -> None:
    rc, output = _run(["clipper", "--help"])
    assert "Edit videos" in output


def test_shortcut_defaults_to_tts_when_first_token_is_text() -> None:
    """`mediaflow "hello" -v missing.wav` should hit the TTS code path."""
    rc, output = _run(["hello", "-v", "missing.wav"])
    # The TTS path raises FileNotFoundError → "ERROR: voice prompt not found".
    assert "voice prompt not found" in output
    assert rc == 1


def test_unknown_subcommand_is_treated_as_tts() -> None:
    """Anything that isn't ``tts``/``clipper``/``--help``/a flag routes to TTS."""
    rc, output = _run(["some-text", "-v", "missing.wav"])
    assert "voice prompt not found" in output


def test_flag_alone_prints_help() -> None:
    rc, output = _run(["-h"])
    assert rc == 0
    assert "MediaFlow" in output
