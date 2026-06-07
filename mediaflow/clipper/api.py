"""Programmatic Python API for the MediaFlow clipper pipeline.

This module exposes a thin facade over the per-mode entry points in
:mod:`mediaflow.clipper.main` so callers can run the autocut pipeline from
any Python program without going through :mod:`argparse`::

    from mediaflow.clipper import TranscribeConfig, transcribe

    config = TranscribeConfig(
        inputs=["input.mp4"],
        cfg="config",
        encoding="utf-8",
    )
    transcribe(config)

The same facade powers the ``mediaflow clipper`` console script.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


TRANSCRIBE = "transcribe"
CUT = "cut"
DAEMON = "daemon"
TO_MD = "to_md"
SRT_TO_COMPACT = "srt_to_compact"

_VALID_MODES = (TRANSCRIBE, CUT, DAEMON, TO_MD, SRT_TO_COMPACT)


def _ensure_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass
class ClipperConfig:
    """Base configuration shared by every clipper mode.

    Attributes:
        inputs: Filesystem paths consumed by the mode (videos, audios, SRTs
            or ``.md`` files depending on ``cut_mode``).
        cfg: Name of the YAML config to load. Defaults to ``"config"``.
        encoding: Text encoding for reading/writing subtitle and markdown
            files.
        force_write: Overwrite existing output files.
        bitrate: MoviePy bitrate string used during cutting.
        prompt: Initial whisper prompt.
        whisper_model: Whisper checkpoint name.
        language: Output language (``"zh"`` or ``"en"``).
        device: ``"cpu"`` or ``"cuda"``.
        use_VAD: VAD mode (``"1"``, ``"0"`` or ``"auto"``).
        cut_mode: One of :data:`TRANSCRIBE`, :data:`CUT`, :data:`DAEMON`,
            :data:`TO_MD`, :data:`SRT_TO_COMPACT`.
        translate: Enable subtitle translation.
        translator_type: ``"hunyuanmt"`` / ``"openai"`` / ``"deepl"`` /
            ``"google"`` / ``"local"``.
        bilingual: Emit bilingual subtitles when ``True``.
        bilingual_format: Layout of bilingual subtitles.
    """

    inputs: List[str] = field(default_factory=list)
    cfg: str = "config"
    encoding: str = "utf-8"
    force_write: bool = False
    bitrate: str = "10m"
    prompt: str = "Hello"
    whisper_model: str = "large-v3"
    language: str = "zh"
    device: str = "cuda"
    use_VAD: str = "auto"
    cut_mode: str = TRANSCRIBE
    translate: bool = False
    translator_type: str = "hunyuanmt"
    bilingual: bool = True
    bilingual_format: str = "dual_line"

    def __post_init__(self) -> None:
        self.inputs = _ensure_str_list(self.inputs)
        if not self.inputs:
            raise ValueError("ClipperConfig.inputs must be non-empty")
        if self.cut_mode not in _VALID_MODES:
            raise ValueError(
                f"cut_mode must be one of {_VALID_MODES}, got {self.cut_mode!r}"
            )


def _build_args(config: ClipperConfig) -> argparse.Namespace:
    """Translate a :class:`ClipperConfig` into the legacy argparse namespace."""
    args = argparse.Namespace(
        inputs=list(config.inputs),
        cfg=config.cfg,
        encoding=config.encoding,
        force_write=config.force_write,
        bitrate=config.bitrate,
        prompt=config.prompt,
        whisper_model=config.whisper_model,
        language=config.language,
        device=config.device,
        use_VAD=config.use_VAD,
        cut_mode=config.cut_mode,
        translate=config.translate,
        translator_type=config.translator_type,
        bilingual=config.bilingual,
        bilingual_format=config.bilingual_format,
    )
    return args


def _run_mode(args: argparse.Namespace) -> None:
    from mediaflow.common import compact_rst, trans_srt_to_md

    if args.cut_mode == TRANSCRIBE:
        from mediaflow.clipper.cores.transcriber import Transcribe

        Transcribe(args).run()
    elif args.cut_mode == TO_MD:
        if len(args.inputs) == 2:
            input_1, input_2 = args.inputs
            base, ext = os.path.splitext(input_1)
            if ext != ".srt":
                input_1, input_2 = input_2, input_1
            trans_srt_to_md(args.encoding, args.force_write, input_1, input_2)
        elif len(args.inputs) == 1:
            trans_srt_to_md(args.encoding, args.force_write, args.inputs[0])
        else:
            raise ValueError(
                "TO_MD mode requires 1 (SRT) or 2 (SRT, video) inputs"
            )
    elif args.cut_mode == CUT:
        from mediaflow.clipper.cores.cut import Cutter

        Cutter(args).run()
    elif args.cut_mode == DAEMON:
        from mediaflow.clipper.cores.daemon import Daemon

        Daemon(args).run()
    elif args.cut_mode == SRT_TO_COMPACT:
        compact_rst(args.inputs[0], args.encoding)


def transcribe(config: ClipperConfig) -> None:
    """Run whisper + VAD to produce ``.srt`` and ``.md`` artefacts."""
    config.cut_mode = TRANSCRIBE
    _run_mode(_build_args(config))


def cut_media(config: ClipperConfig) -> None:
    """Cut media based on a marked markdown file."""
    config.cut_mode = CUT
    _run_mode(_build_args(config))


def run_daemon(config: ClipperConfig) -> None:
    """Run the autocut daemon on a watched folder."""
    config.cut_mode = DAEMON
    _run_mode(_build_args(config))


def to_markdown(config: ClipperConfig) -> None:
    """Convert an SRT file (and optional video) into a markdown task list."""
    config.cut_mode = TO_MD
    _run_mode(_build_args(config))


def compact_subtitles(config: ClipperConfig) -> None:
    """Toggle an SRT file between verbose and compact forms."""
    config.cut_mode = SRT_TO_COMPACT
    _run_mode(_build_args(config))


# Backwards-compatible alias for the historical "TranscribeConfig" name.
TranscribeConfig = ClipperConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediaflow clipper",
        description="Edit videos based on transcribed subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        help="Inputs filenames/folders",
    )
    parser.add_argument(
        "-cfg",
        "--cfg",
        default="config",
        type=str,
        help="Your detailed configuration of Flow",
    )
    parser.add_argument(
        "--translator-type",
        type=str,
        default="hunyuanmt",
        choices=["openai", "deepl", "google", "hunyuanmt", "local"],
        help="选择翻译器类型",
    )
    parser.add_argument(
        "--bilingual-format",
        type=str,
        default="dual_line",
        choices=["dual_line", "brackets", "slash"],
        help="双语字幕格式",
    )
    parser.add_argument(
        "--no-bilingual",
        action="store_false",
        dest="bilingual",
        help="禁用双语字幕",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point for the clipper subcommand."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from mediaflow.common import Logging, YamlParser

    logger = Logging(__name__).get_logger()
    args = YamlParser(args.cfg, path="").add_args(args)
    logger.info("-------->>> Input parameters: <<<--------")
    logger.info(args)
    logger.info("-------->>> ================= <<<--------")

    config = ClipperConfig(
        inputs=args.inputs,
        cfg=args.cfg,
        encoding=args.encoding,
        force_write=args.force_write,
        bitrate=args.bitrate,
        prompt=args.prompt,
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        use_VAD=args.use_VAD,
        cut_mode=args.cut_mode,
        translate=args.translate,
        translator_type=args.translator_type,
        bilingual=args.bilingual,
        bilingual_format=args.bilingual_format,
    )
    _run_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
