"""
Topological analysis module using a real TDA backend (ripser).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist


def point_cloud_from_uruz(
    uruz: Iterable[Tuple[int, int]],
    n: int,
    embedding: str = "torus",
) -> np.ndarray:
    """
    Converts (u_r, u_z) points into a Euclidean point cloud.

    embedding:
        - "torus": 4D embedding on S1 x S1 using cos/sin.
        - "square": 2D normalized square in [0,1]^2.
    """
    points = np.asarray(list(uruz), dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)

    if embedding == "square":
        return points / float(n)

    if embedding != "torus":
        raise ValueError(f"Unsupported embedding: {embedding}")

    theta = 2.0 * math.pi * points[:, 0] / float(n)
    phi = 2.0 * math.pi * points[:, 1] / float(n)
    return np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.cos(phi),
        np.sin(phi),
    ])


def calculate_persistence_diagram(points: np.ndarray, maxdim: int = 2) -> List[np.ndarray]:
    """
    Computes persistence diagrams using ripser.
    Returns a list of diagrams for dimensions 0..maxdim.
    """
    if points.size == 0:
        return [np.empty((0, 2)) for _ in range(maxdim + 1)]
    result = ripser(points, maxdim=maxdim)
    return result["dgms"]


def _estimate_betti_numbers(
    diagrams: List[np.ndarray],
    persistence_threshold: Optional[float] = None,
) -> Dict[str, int]:
    bettis = {"betti_0": 0, "betti_1": 0, "betti_2": 0}
    for dim, dgm in enumerate(diagrams[:3]):
        if dgm.size == 0:
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        inf_mask = np.isinf(deaths)

        if dim == 0:
            bettis["betti_0"] = int(np.sum(inf_mask))
            continue

        lifetimes = deaths[~inf_mask] - births[~inf_mask]
        if lifetimes.size == 0:
            continue
        if persistence_threshold is None:
            threshold = 0.1 * float(np.max(lifetimes))
        else:
            threshold = persistence_threshold
        count = int(np.sum(lifetimes >= threshold))
        bettis[f"betti_{dim}"] = count

    return bettis


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _distance_histogram(points: np.ndarray, bins: int) -> np.ndarray:
    if points.shape[0] < 2:
        return np.array([])
    distances = pdist(points)
    if distances.size == 0:
        return np.array([])
    counts, _ = np.histogram(distances, bins=bins)
    return counts


def _normalize_entropy(entropy: float, nonzero_bins: int) -> float:
    if nonzero_bins <= 1:
        return 0.0
    max_entropy = float(np.log(nonzero_bins))
    if max_entropy == 0.0:
        return 0.0
    return float(entropy / max_entropy)


def calculate_topological_entropy(
    points: np.ndarray,
    bins: int = 30,
    normalize: bool = False,
) -> float:
    """
    Computes a distance-based entropy over the point cloud.
    If normalize=True, returns H / log(k) where k is the number of non-empty bins.
    """
    counts = _distance_histogram(points, bins)
    if counts.size == 0:
        return 0.0
    entropy = _entropy_from_counts(counts)
    if not normalize:
        return entropy

    nonzero_bins = int(np.sum(counts > 0))
    return _normalize_entropy(entropy, nonzero_bins)


def check_betti_numbers(
    points: np.ndarray,
    n_expected: int = 2,
    maxdim: int = 2,
    persistence_threshold: Optional[float] = None,
    entropy_bins: int = 30,
) -> Dict[str, float]:
    """
    Runs TDA and returns Betti numbers, entropy, and a security flag.
    """
    diagrams = calculate_persistence_diagram(points, maxdim=maxdim)
    bettis = _estimate_betti_numbers(diagrams, persistence_threshold)
    entropy = calculate_topological_entropy(points, bins=entropy_bins, normalize=False)
    entropy_norm = calculate_topological_entropy(points, bins=entropy_bins, normalize=True)
    is_secure = (
        bettis["betti_0"] == 1
        and bettis["betti_1"] == n_expected
        and bettis["betti_2"] == 1
    )

    return {
        **bettis,
        "is_secure": bool(is_secure),
        "topological_entropy": float(entropy),
        "topological_entropy_norm": float(entropy_norm),
        "entropy_bins": int(entropy_bins),
    }


def build_report(
    points: np.ndarray,
    n_expected: int = 2,
    maxdim: int = 2,
    persistence_threshold: Optional[float] = None,
    entropy_bins: int = 30,
) -> Dict[str, float]:
    """
    Builds a compact report with metrics and diagram summaries.
    """
    diagrams = calculate_persistence_diagram(points, maxdim=maxdim)
    bettis = _estimate_betti_numbers(diagrams, persistence_threshold)
    counts = _distance_histogram(points, entropy_bins)
    entropy = _entropy_from_counts(counts) if counts.size else 0.0
    nonzero_bins = int(np.sum(counts > 0)) if counts.size else 0
    entropy_norm = _normalize_entropy(entropy, nonzero_bins) if counts.size else 0.0

    diagram_sizes = {
        f"diagram_dim_{idx}": int(dgm.shape[0]) for idx, dgm in enumerate(diagrams)
    }

    is_secure = (
        bettis["betti_0"] == 1
        and bettis["betti_1"] == n_expected
        and bettis["betti_2"] == 1
    )

    return {
        **bettis,
        **diagram_sizes,
        "topological_entropy": float(entropy),
        "topological_entropy_norm": float(entropy_norm),
        "entropy_bins": int(entropy_bins),
        "entropy_nonzero_bins": int(nonzero_bins),
        "is_secure": bool(is_secure),
        "points": int(points.shape[0]),
        "dimension": int(points.shape[1]) if points.size else 0,
    }
