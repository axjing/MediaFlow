"""Semantic model + RepCodec wrappers used by IndexTTS2.

This module is a thin shim around the upstream ``transformers`` and
HuggingFace Hub APIs plus the locally bundled :mod:`mediaflow.tts.utils.maskgct`
sub-package. It deliberately contains **no** training utilities - those have
been removed during the MediaFlow refactor.
"""
from __future__ import annotations

import json5
import os
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from mediaflow.tts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec


def _coerce(node: Any) -> Any:
    """Tiny helper for converting dict-like configs to a JsonHParams tree."""
    if isinstance(node, dict):
        return JsonHParams(**node)
    return node


def _load_config(config_fn: str, lowercase: bool = False) -> dict:
    """Load a JSON5 config file with optional ``base_config`` override support."""
    with open(config_fn, "r", encoding="utf-8") as fh:
        data = fh.read()
    config_ = json5.loads(data)
    if "base_config" in config_:
        work_dir = os.getenv("WORK_DIR", "")
        p_config_path = os.path.join(work_dir, config_["base_config"])
        if os.path.isfile(p_config_path):
            config_ = _load_config(p_config_path)
    if lowercase:
        config_ = {k.lower(): _coerce(v) for k, v in config_.items()}
    return config_


def load_config(config_fn: str, lowercase: bool = False) -> "JsonHParams":
    """Load a JSON5 config and return it as a :class:`JsonHParams` object."""
    config_ = _load_config(config_fn, lowercase=lowercase)
    return JsonHParams(**config_)


class JsonHParams:
    """Recursive dict-like object with attribute access."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = JsonHParams(**value)
            self.__dict__[key] = value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __repr__(self) -> str:
        return repr(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()


def build_semantic_model(path: str = "./checkpoints/wav2vec2bert_stats.pt") -> tuple:
    """Construct the Wav2Vec2-BERT model used to extract semantic features."""
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    stat_mean_var = torch.load(path, map_location="cpu")
    semantic_mean = stat_mean_var["mean"]
    semantic_std = torch.sqrt(stat_mean_var["var"])
    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg: Any) -> RepCodec:
    """Construct the RepCodec used for quantising semantic features."""
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    return semantic_codec
