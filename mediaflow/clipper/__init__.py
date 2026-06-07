"""MediaFlow clipper subpackage.

Provides the autocut pipeline: whisper-based transcription, optional
bilingual translation, markdown task list editing, and video/audio
cutting via moviepy.

Public programmatic API::

    from mediaflow.clipper import (
        ClipperConfig,
        TranscribeConfig,
        compact_subtitles,
        cut_media,
        run_daemon,
        to_markdown,
        transcribe,
    )

    config = ClipperConfig(inputs=["input.mp4"], cfg="config")
    transcribe(config)

Console-script entry point: :mod:`mediaflow.clipper.cli` (also wired into
:func:`mediaflow.cli.main` as the ``clipper`` subcommand).
"""
from mediaflow.clipper.api import (
    ClipperConfig,
    TranscribeConfig,
    compact_subtitles,
    cut_media,
    run_daemon,
    to_markdown,
    transcribe,
)

__all__ = [
    "ClipperConfig",
    "TranscribeConfig",
    "compact_subtitles",
    "cut_media",
    "run_daemon",
    "to_markdown",
    "transcribe",
]
