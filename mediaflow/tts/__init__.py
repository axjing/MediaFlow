"""MediaFlow text-to-speech subpackage.

Contains the IndexTTS / IndexTTS2 inference code, BigVGAN vocoder, GPT
backbone, and the S2Mel diffusion decoder used to produce waveforms.

Public programmatic API::

    from mediaflow.tts import synthesize, SynthesisConfig

    config = SynthesisConfig(
        text="Hello world",
        voice_prompt="voice.wav",
        output_path="out.wav",
    )
    synthesize(config)

Console-script entry point: :mod:`mediaflow.tts.cli` (also wired into
:func:`mediaflow.cli.main` as the ``tts`` subcommand).
"""
from mediaflow.tts.api import SynthesisConfig, synthesize

__all__ = ["SynthesisConfig", "synthesize"]
