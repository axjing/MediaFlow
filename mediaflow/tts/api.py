"""Programmatic Python API for the IndexTTS text-to-speech pipeline.

This module exposes a thin, dependency-free facade over
:mod:`mediaflow.tts.infer` so callers can synthesise speech from any Python
program without going through :mod:`argparse`::

    from mediaflow.tts import synthesize, SynthesisConfig

    config = SynthesisConfig(
        text="你好，欢迎使用 MediaFlow",
        voice_prompt="prompts/voice.wav",
        output_path="outputs/hello.wav",
    )
    synthesize(config)

Or via the :class:`IndexTTS` class directly for low-level control.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class SynthesisConfig:
    """Bundle of parameters for one-shot TTS synthesis.

    Attributes:
        text: Source text to be spoken.
        voice_prompt: Path to a WAV file used as the speaker prompt.
        output_path: Where to write the generated WAV file.
        config_path: YAML config consumed by :class:`IndexTTS`.
        model_dir: Directory containing the model checkpoints.
        fp16: Whether to use half-precision inference.
        device: ``"cpu"``, ``"cuda:0"``, ``"mps"``, ``"xpu"`` or ``None``
            (auto-detect).
        force: Overwrite ``output_path`` if it already exists.
    """

    text: str
    voice_prompt: str
    output_path: str = "gen.wav"
    config_path: str = "checkpoints/config.yaml"
    model_dir: str = "checkpoints"
    fp16: bool = False
    device: Optional[str] = None
    force: bool = False


def _resolve_device(device: Optional[str]) -> str:
    """Pick a torch device if the caller left ``device`` as ``None``."""
    if device is not None:
        return device
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is required
        raise RuntimeError("PyTorch is required to run TTS inference") from exc
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"


def _check_inputs(config: SynthesisConfig) -> None:
    if not config.text or not config.text.strip():
        raise ValueError("text is empty")
    if not os.path.exists(config.voice_prompt):
        raise FileNotFoundError(f"voice prompt not found: {config.voice_prompt}")
    if not os.path.exists(config.config_path):
        raise FileNotFoundError(f"config not found: {config.config_path}")
    if os.path.exists(config.output_path) and not config.force:
        raise FileExistsError(
            f"output {config.output_path} exists; pass force=True to overwrite"
        )


def synthesize(config: SynthesisConfig) -> str:
    """Synthesise speech from ``config.text`` and return the output path."""
    _check_inputs(config)
    device = _resolve_device(config.device)
    fp16 = config.fp16 and device != "cpu"

    from mediaflow.tts.infer import IndexTTS  # heavy import deferred

    output_path = config.output_path
    if os.path.exists(output_path):
        os.remove(output_path)

    tts = IndexTTS(
        cfg_path=config.config_path,
        model_dir=config.model_dir,
        use_fp16=fp16,
        device=device,
    )
    tts.infer(
        audio_prompt=config.voice_prompt,
        text=config.text.strip(),
        output_path=output_path,
    )
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console-script entry point that mirrors :func:`mediaflow.tts.cli.main`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mediaflow tts",
        description="IndexTTS Command Line",
    )
    parser.add_argument("text", type=str, help="Text to be synthesized")
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        required=True,
        help="Path to the audio prompt file (wav format)",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="gen.wav",
        help="Path to the output wav file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the config file. Default is 'checkpoints/config.yaml'",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="checkpoints",
        help="Path to the model directory. Default is 'checkpoints'",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 for inference if available",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force to overwrite the output file if it exists",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cpu, cuda, mps, xpu).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SynthesisConfig(
        text=args.text,
        voice_prompt=args.voice,
        output_path=args.output_path,
        config_path=args.config,
        model_dir=args.model_dir,
        fp16=args.fp16,
        device=args.device,
        force=args.force,
    )
    try:
        synthesize(config)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
