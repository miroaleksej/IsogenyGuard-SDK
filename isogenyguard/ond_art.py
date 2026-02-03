"""
OND-ART v0.1 report generation utilities.

These functions implement a structural profile with bootstrap confidence
intervals and optional baseline comparison. This is diagnostic only and
makes no security claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any, Optional

import json
from importlib import resources

import numpy as np

from .visualization import quantize_linear_value

try:
    from jsonschema import validate as _jsonschema_validate
except Exception:  # pragma: no cover - optional import at runtime
    _jsonschema_validate = None


@dataclass
class BootstrapResult:
    metric_keys: List[str]
    metric_samples: np.ndarray  # shape (B, K)
    metrics_mean: Dict[str, float]
    metrics_ci95: Dict[str, Tuple[float, float]]
    samples: int
    seed: int


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _normalize_entropy(entropy: float, nonzero_bins: int) -> float:
    if nonzero_bins <= 1:
        return 0.0
    max_entropy = float(np.log(nonzero_bins))
    if max_entropy == 0.0:
        return 0.0
    return float(entropy / max_entropy)


def _ensure_list(observations: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    obs = [(int(ur), int(uz)) for ur, uz in observations]
    if not obs:
        return []
    if any(len(pair) != 2 for pair in obs):
        raise ValueError("observations must be an iterable of (u_r, u_z) pairs")
    return obs


def _flatten_numeric(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_numeric(value, prefix=path))
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            flat[path] = float(value)
    return flat


def flatten_report(report: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    Flattens a nested report into a flat dict suitable for CSV/JSONL.
    Lists are expanded by index.
    """
    flat: Dict[str, str] = {}
    for key, value in report.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_report(value, prefix=path))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                flat.update(flatten_report({str(idx): item}, prefix=path))
        else:
            flat[path] = str(value)
    return flat


def compute_ond_art_metrics(
    observations: Iterable[Tuple[int, int]],
    modulus: Optional[int] = None,
    bins: int = 100,
) -> Dict[str, Any]:
    """
    Computes OND-ART metrics on (u_r, u_z) observations.
    """
    obs = _ensure_list(observations)
    if not obs:
        return {
            "rank": {"entropy": 0.0, "entropy_norm": 0.0},
            "subspace": {"occupancy_ratio": 0.0, "nonempty_bins": 0, "total_bins": 0},
            "branching": {"mean_distinct_per_row": 0.0, "normalized": 0.0, "rows": 0},
        }

    max_val = max(max(ur, uz) for ur, uz in obs)
    n = int(modulus) if modulus else int(max_val + 1)
    bin_count = min(bins, max(n, 2))

    # Rank entropy (entropy of 1D distributions)
    counts_ur = np.zeros(bin_count, dtype=np.int64)
    counts_uz = np.zeros(bin_count, dtype=np.int64)
    for ur, uz in obs:
        bx = quantize_linear_value(int(ur), n, bins=bin_count) - 1
        by = quantize_linear_value(int(uz), n, bins=bin_count) - 1
        counts_ur[bx] += 1
        counts_uz[by] += 1

    entropies = []
    entropies_norm = []
    for counts in (counts_ur, counts_uz):
        entropy = _entropy_from_counts(counts)
        nonzero_bins = int(np.sum(counts > 0))
        entropies.append(entropy)
        entropies_norm.append(_normalize_entropy(entropy, nonzero_bins))

    rank_entropy = float(np.mean(entropies))
    rank_entropy_norm = float(np.mean(entropies_norm))

    # Subspace occupancy (quantized 2D grid)
    grid = np.zeros((bin_count, bin_count), dtype=np.int64)
    for ur, uz in obs:
        bx = quantize_linear_value(int(ur), n, bins=bin_count) - 1
        by = quantize_linear_value(int(uz), n, bins=bin_count) - 1
        grid[by, bx] += 1
    nonempty = int(np.sum(grid > 0))
    total_bins = int(bin_count * bin_count)
    occupancy_ratio = float(nonempty / total_bins) if total_bins else 0.0
    row_occupancy = np.sum(grid > 0, axis=1) / float(bin_count)
    row_occupancy_mean = float(np.mean(row_occupancy)) if row_occupancy.size else 0.0
    row_occupancy_std = float(np.std(row_occupancy)) if row_occupancy.size else 0.0

    # Branching index (distinct u_z per u_r)
    rows = {}
    for ur, uz in obs:
        rows.setdefault(int(ur), set()).add(int(uz))
    distinct_counts = [len(v) for v in rows.values()]
    mean_distinct = float(np.mean(distinct_counts)) if distinct_counts else 0.0
    median_distinct = float(np.median(distinct_counts)) if distinct_counts else 0.0
    std_distinct = float(np.std(distinct_counts)) if distinct_counts else 0.0
    normalizer = float(n if n > 0 else max(distinct_counts or [1]))
    branching_norm = float(mean_distinct / normalizer) if normalizer else 0.0

    return {
        "rank": {
            "entropy": rank_entropy,
            "entropy_norm": rank_entropy_norm,
            "bins": int(bin_count),
            "entropy_axis": {
                "u_r": float(entropies[0]),
                "u_z": float(entropies[1]),
            },
            "entropy_norm_axis": {
                "u_r": float(entropies_norm[0]),
                "u_z": float(entropies_norm[1]),
            },
        },
        "subspace": {
            "occupancy_ratio": occupancy_ratio,
            "nonempty_bins": nonempty,
            "total_bins": total_bins,
            "bins": int(bin_count),
            "row_occupancy_mean": row_occupancy_mean,
            "row_occupancy_std": row_occupancy_std,
        },
        "branching": {
            "mean_distinct_per_row": mean_distinct,
            "normalized": branching_norm,
            "rows": int(len(rows)),
            "std_distinct_per_row": std_distinct,
            "median_distinct_per_row": median_distinct,
        },
    }


def bootstrap_metrics(
    observations: Iterable[Tuple[int, int]],
    metric_fn,
    samples: int = 200,
    seed: int = 0,
) -> BootstrapResult:
    obs = _ensure_list(observations)
    n = len(obs)
    rng = np.random.default_rng(seed)

    metric_samples: List[np.ndarray] = []
    keys: List[str] = []

    for idx in range(samples):
        sample_idx = rng.integers(0, n, n)
        sample_obs = [obs[i] for i in sample_idx]
        metrics = metric_fn(sample_obs)
        flat = _flatten_numeric(metrics)

        if idx == 0:
            keys = sorted(flat.keys())
        else:
            if set(flat.keys()) != set(keys):
                raise ValueError("Inconsistent metric keys across bootstrap samples")

        metric_samples.append(np.array([flat[k] for k in keys], dtype=float))

    sample_matrix = np.vstack(metric_samples) if metric_samples else np.zeros((0, 0))
    means = np.mean(sample_matrix, axis=0) if sample_matrix.size else np.array([])
    lows = np.percentile(sample_matrix, 2.5, axis=0) if sample_matrix.size else np.array([])
    highs = np.percentile(sample_matrix, 97.5, axis=0) if sample_matrix.size else np.array([])

    metrics_mean = {k: float(v) for k, v in zip(keys, means)}
    metrics_ci95 = {k: (float(l), float(h)) for k, l, h in zip(keys, lows, highs)}

    return BootstrapResult(
        metric_keys=keys,
        metric_samples=sample_matrix,
        metrics_mean=metrics_mean,
        metrics_ci95=metrics_ci95,
        samples=samples,
        seed=seed,
    )


def _distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.linalg.norm(vec_a - vec_b))


def _baseline_thresholds(distances: np.ndarray, percentiles: Tuple[float, float, float]) -> Dict[str, float]:
    if distances.size == 0:
        return {"green": 0.0, "yellow": 0.0, "red": 0.0}
    return {
        "green": float(np.percentile(distances, percentiles[0])),
        "yellow": float(np.percentile(distances, percentiles[1])),
        "red": float(np.percentile(distances, percentiles[2])),
    }


def _classify(distance: float, thresholds: Dict[str, float]) -> str:
    if distance <= thresholds.get("green", 0.0):
        return "Within Baseline Envelope"
    if distance <= thresholds.get("yellow", 0.0):
        return "Deviating"
    return "Strong Deviation"


def build_ond_art_report(
    observations: Iterable[Tuple[int, int]],
    modulus: Optional[int] = None,
    bins: int = 100,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 0,
    baseline_observations: Optional[Iterable[Tuple[int, int]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    baseline_percentiles: Tuple[float, float, float] = (95.0, 99.0, 99.5),
    context: Optional[Dict[str, Any]] = None,
    observation_map: Optional[Dict[str, Any]] = None,
    obs_space: Optional[Dict[str, Any]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    obs = _ensure_list(observations)
    max_val = max(max(ur, uz) for ur, uz in obs) if obs else 0
    n = int(modulus) if modulus else int(max_val + 1)

    metric_fn = lambda x: compute_ond_art_metrics(x, modulus=n, bins=bins)
    metrics = metric_fn(obs)
    bootstrap = bootstrap_metrics(obs, metric_fn, samples=bootstrap_samples, seed=bootstrap_seed)

    baseline_report = None
    if baseline_observations is not None:
        baseline_obs = _ensure_list(baseline_observations)
        baseline_bootstrap = bootstrap_metrics(
            baseline_obs,
            metric_fn,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        )

        current_vec = np.array([_flatten_numeric(metrics)[k] for k in bootstrap.metric_keys], dtype=float)
        baseline_mean_vec = np.array([baseline_bootstrap.metrics_mean[k] for k in bootstrap.metric_keys], dtype=float)

        # Distances for current bootstrap samples to baseline mean
        if bootstrap.metric_samples.size:
            dist_samples = np.linalg.norm(bootstrap.metric_samples - baseline_mean_vec, axis=1)
        else:
            dist_samples = np.array([])

        distance = _distance(current_vec, baseline_mean_vec)
        distance_ci95 = (
            float(np.percentile(dist_samples, 2.5)) if dist_samples.size else 0.0,
            float(np.percentile(dist_samples, 97.5)) if dist_samples.size else 0.0,
        )

        baseline_distances = (
            np.linalg.norm(baseline_bootstrap.metric_samples - baseline_mean_vec, axis=1)
            if baseline_bootstrap.metric_samples.size
            else np.array([])
        )
        threshold_values = thresholds or _baseline_thresholds(baseline_distances, baseline_percentiles)

        baseline_report = {
            "distance": float(distance),
            "distance_ci95": [distance_ci95[0], distance_ci95[1]],
            "thresholds": threshold_values,
            "classification": _classify(distance, threshold_values),
        }

    report = {
        "ond_art_version": "0.1",
        "notes": notes or ["Diagnostic only; no security claim."],
        "context": context or {
            "scheme": "ECDSA-secp256k1",
            "params": {"hash": "sha256"},
        },
        "observation_map": observation_map or {
            "pi_id": "uruz-map",
            "pi_version": "1.0",
            "pi_spec_hash": "TBD",
        },
        "obs_space": obs_space or {
            "type": "Z_mod_m",
            "modulus": int(n),
        },
        "sampling": {
            "N": int(len(obs)),
            "B": int(bootstrap_samples),
        },
        "metrics": metrics,
        "bootstrap": {
            "samples": bootstrap.samples,
            "seed": bootstrap.seed,
            "metrics_mean": bootstrap.metrics_mean,
            "metrics_ci95": {k: [v[0], v[1]] for k, v in bootstrap.metrics_ci95.items()},
        },
        "baseline": baseline_report,
    }

    return report


def load_ond_art_schema() -> Dict[str, Any]:
    try:
        schema_path = resources.files("isogenyguard").joinpath("schemas/ond-art-report-0.1.schema.json")
        with schema_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except AttributeError:
        with resources.open_text("isogenyguard.schemas", "ond-art-report-0.1.schema.json", encoding="utf-8") as handle:
            return json.load(handle)


def validate_ond_art_report(report: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> None:
    if _jsonschema_validate is None:
        raise RuntimeError("jsonschema is not installed; cannot validate report")
    schema_obj = schema or load_ond_art_schema()
    _jsonschema_validate(instance=report, schema=schema_obj)
