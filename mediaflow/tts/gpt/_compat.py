"""Compatibility shims for transformers 5.x.

Several utility classes/symbols that older versions of IndexTTS relied on
were removed or moved in transformers 5.x. The minimal re-implementations
below let the GPT2 wiring in :mod:`mediaflow.tts.gpt.model` and the
:mod:`mediaflow.tts.accel` module keep working without bundling the
upstream transformers source tree.
"""
from __future__ import annotations

import torch
from torch import nn


class Conv1D(nn.Module):
    """Reimplementation of ``transformers.pytorch_utils.Conv1D``.

    The upstream symbol was removed in transformers 5.x. GPT-2 historically
    used a "transposed" convolution shaped ``Conv1D(nf, nx)`` whose forward
    applies ``x @ weight.T + bias``.
    """

    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.weight.shape[1],)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(*size_out)


def find_pruneable_heads_and_indices(
    heads: set[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """Reimplementation of ``transformers.pytorch_utils.find_pruneable_heads_and_indices``."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_conv1d_layer(layer: Conv1D, index: torch.Tensor, dim: int = 1) -> Conv1D:
    """Reimplementation of ``transformers.pytorch_utils.prune_conv1d_layer``.

    Only ``dim=1`` is required by IndexTTS, so the other dimension is not
    implemented. Callers needing that branch should add it explicitly.
    """
    if dim != 1:
        raise NotImplementedError("prune_conv1d_layer only implements dim=1")
    index = index.to(layer.weight.device)
    new_layer = Conv1D(
        nf=layer.weight.shape[1] - len(index),
        nx=layer.weight.shape[0],
    ).to(layer.weight.device)
    new_layer.weight.data = layer.weight[:, index].clone()
    new_layer.bias.data = layer.bias.clone()
    return new_layer


class _LocalModelParallel:
    """Reimplements the small slice of ``model_parallel_utils`` that IndexTTS uses.

    The original ``transformers.utils.model_parallel_utils`` (and
    ``transformers.modeling_utils.model_parallel`` machinery) was removed in
    transformers 5.x. IndexTTS still gates a few code paths on
    ``self.model_parallel`` and calls ``self.parallelize(device_map)`` /
    ``self.deparallelize()``. This mixin restores that surface area so the
    rest of the GPT-2 stack keeps working without bundling transformers.
    """

    model_parallel: bool = False
    device_map: dict | None = None

    def parallelize(self, device_map: dict | None = None) -> None:
        self.device_map = device_map
        self.model_parallel = True

    def deparallelize(self) -> None:
        self.model_parallel = False
        self.device_map = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
