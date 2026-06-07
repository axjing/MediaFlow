"""Smoke tests for the public programmatic APIs.

The test suite is split into two groups:

* Tests that exercise only the lightweight facade (no heavy ML imports).
* Tests that touch the heavy translator stack and are auto-skipped when
  ``torch`` / ``opencc`` are not installed.
"""
from __future__ import annotations

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on env
    _HAS_TORCH = False

try:
    import opencc as _opencc  # noqa: F401

    _HAS_OPENCC = True
except ImportError:  # pragma: no cover - depends on env
    _HAS_OPENCC = False

import pytest

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
requires_opencc = pytest.mark.skipif(not _HAS_OPENCC, reason="opencc not installed")


# ---------------------------------------------------------------------------
# Lightweight tests — no heavy deps required.
# ---------------------------------------------------------------------------


def test_synthesis_config_defaults() -> None:
    from mediaflow.tts import SynthesisConfig

    config = SynthesisConfig(text="hi", voice_prompt="x.wav")
    assert config.fp16 is False
    assert config.force is False
    assert config.output_path == "gen.wav"


def test_clipper_config_validates_mode() -> None:
    from mediaflow.clipper import ClipperConfig

    with pytest.raises(ValueError):
        ClipperConfig(inputs=["x"], cut_mode="bogus")


def test_clipper_config_string_input() -> None:
    from mediaflow.clipper import ClipperConfig

    config = ClipperConfig(inputs="x.mp4")
    assert config.inputs == ["x.mp4"]


def test_clipper_config_to_args_round_trip() -> None:
    from mediaflow.clipper import ClipperConfig
    from mediaflow.clipper.api import _build_args

    config = ClipperConfig(
        inputs=["a.mp4", "b.mp4"],
        cfg="mycfg",
        encoding="utf-8",
        force_write=True,
        cut_mode="transcribe",
    )
    args = _build_args(config)
    assert args.inputs == ["a.mp4", "b.mp4"]
    assert args.cfg == "mycfg"
    assert args.cut_mode == "transcribe"


def test_synthesize_rejects_empty_text() -> None:
    from mediaflow.tts import SynthesisConfig, synthesize

    config = SynthesisConfig(text="   ", voice_prompt="x.wav")
    with pytest.raises(ValueError):
        synthesize(config)


def test_synthesize_rejects_missing_voice() -> None:
    from mediaflow.tts import SynthesisConfig, synthesize

    config = SynthesisConfig(text="hi", voice_prompt="missing.wav")
    with pytest.raises(FileNotFoundError):
        synthesize(config)


def test_resolve_device_explicit() -> None:
    from mediaflow.tts.api import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:0") == "cuda:0"


def test_transcribe_alias_matches_clipper_config() -> None:
    from mediaflow.clipper import ClipperConfig, TranscribeConfig

    assert TranscribeConfig is ClipperConfig


# ---------------------------------------------------------------------------
# Heavy tests — require torch / opencc.
# ---------------------------------------------------------------------------


@requires_torch
def test_resolve_device_autodetect() -> None:
    from mediaflow.tts.api import _resolve_device

    device = _resolve_device(None)
    assert device in {"cpu", "cuda:0", "xpu", "mps"}


@requires_torch
@requires_opencc
def test_subtitle_translator_local_returns_string() -> None:
    from mediaflow.clipper.cores.translator import SubtitleTranslator

    translator = SubtitleTranslator(
        {
            "translator_type": "local",
            "bilingual_subtitles": True,
            "bilingual_format": "dual_line",
        }
    )
    out = translator.translate("Hello, world.", "zh")
    assert isinstance(out, str)
    translator.cleanup()


@requires_torch
@requires_opencc
def test_subtitle_translator_no_bilingual_passthrough() -> None:
    from mediaflow.clipper.cores.translator import SubtitleTranslator

    translator = SubtitleTranslator(
        {
            "translator_type": "local",
            "bilingual_subtitles": False,
        }
    )
    out = translator.translate("Hello, world.", "zh")
    assert out == "Hello, world."
    translator.cleanup()


@requires_torch
@requires_opencc
def test_subtitle_translator_falls_back_to_local_when_optional_deps_missing(
    monkeypatch,
) -> None:
    """If openai/requests/transformers are unavailable, local fallback is used."""
    from mediaflow.clipper.cores.translator import LocalTranslator, SubtitleTranslator

    monkeypatch.setattr(
        "mediaflow.clipper.cores.translator.OPENAI_AVAILABLE", False
    )
    monkeypatch.setattr(
        "mediaflow.clipper.cores.translator.REQUESTS_AVAILABLE", False
    )
    monkeypatch.setattr(
        "mediaflow.clipper.cores.translator.TRANSFORMERS_AVAILABLE", False
    )
    translator = SubtitleTranslator({"translator_type": "hunyuanmt"})
    assert isinstance(translator.translator, LocalTranslator)
    translator.cleanup()


@requires_torch
def test_base_translator_cleanup_releases_attrs() -> None:
    from mediaflow.clipper.cores.translator import BaseTranslator

    class _Dummy(BaseTranslator):
        def __init__(self):
            super().__init__({})
            self.model = object()
            self.tokenizer = object()

        def translate(self, text, target_lang="zh"):
            return text

        def translate_batch(self, texts, target_lang="zh"):
            return list(texts)

    dummy = _Dummy()
    assert dummy.model is not None
    dummy.cleanup()
    assert dummy.model is None
    assert dummy.tokenizer is None
