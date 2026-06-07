"""Helpers for working with timestamped segments (subtitle / audio cuts)."""
from __future__ import annotations

from typing import Dict, List, Sequence

Segment = Dict[str, float]


def expand_segments(
    segments: Sequence[Segment],
    expand_head: float,
    expand_tail: float,
    total_length: float,
) -> List[Segment]:
    """Pad each segment's head/tail while avoiding neighbour overlap."""
    if not segments:
        return []
    results: List[Segment] = []
    for idx, seg in enumerate(segments):
        prev_end = segments[idx - 1]["end"] if idx > 0 else 0
        next_start = segments[idx + 1]["start"] if idx < len(segments) - 1 else total_length
        start = max(seg["start"] - expand_head, prev_end)
        end = min(seg["end"] + expand_tail, next_start)
        results.append({"start": start, "end": end})
    return results


def remove_short_segments(segments: Sequence[Segment], threshold: float) -> List[Segment]:
    """Drop segments whose duration is shorter than ``threshold``."""
    return [seg for seg in segments if seg["end"] - seg["start"] > threshold]


def merge_adjacent_segments(segments: Sequence[Segment], threshold: float) -> List[Segment]:
    """Merge segments whose gap is smaller than ``threshold``."""
    if not segments:
        return []
    results: List[Segment] = []
    i = 0
    while i < len(segments):
        current = dict(segments[i])
        j = i + 1
        while j < len(segments) and segments[j]["start"] < current["end"] + threshold:
            current["end"] = segments[j]["end"]
            i = j
            j += 1
        i += 1
        results.append(current)
    return results
