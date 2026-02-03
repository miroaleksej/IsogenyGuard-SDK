"""
Visualization-only quantization helpers.

These helpers map large integer ranges into a fixed number of bins for
plotting/heatmaps. They MUST NOT be used for cryptographic inference.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def quantize_linear_value(value: int, max_range: int, bins: int = 100) -> int:
    """
    Maps a scalar in [0, max_range] to a 1-based bin index in [1, bins].

    Uses integer arithmetic to avoid floating-point precision loss.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if max_range <= 0:
        raise ValueError("max_range must be positive")

    # Allow values outside range but clamp at the end.
    if value < 0:
        return 1

    bin_index = (value * bins) // max_range + 1
    return _clamp(bin_index, 1, bins)


def quantize_linear(values: Iterable[int], max_range: int, bins: int = 100) -> List[int]:
    """
    Vectorized quantization for a list of values.
    """
    return [quantize_linear_value(v, max_range, bins) for v in values]


def quantize_uruz(
    uruz: Iterable[Tuple[int, int]],
    max_range: int,
    bins: int = 100,
) -> List[Tuple[int, int]]:
    """
    Quantizes (u_r, u_z) pairs into 1-based bin coordinates.
    """
    return [
        (
            quantize_linear_value(ur, max_range, bins),
            quantize_linear_value(uz, max_range, bins),
        )
        for ur, uz in uruz
    ]


def heatmap_counts(
    uruz: Iterable[Tuple[int, int]],
    max_range: int,
    bins: int = 100,
) -> List[List[int]]:
    """
    Builds a bins x bins heatmap of counts from quantized (u_r, u_z) pairs.
    """
    grid = [[0 for _ in range(bins)] for _ in range(bins)]
    for bx, by in quantize_uruz(uruz, max_range, bins):
        grid[by - 1][bx - 1] += 1
    return grid
