import pytest

from isogenyguard.ecdsa import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    SECP256K1_ORDER,
)
from isogenyguard.ond_art import (
    compute_ond_art_metrics,
    bootstrap_metrics,
    build_ond_art_report,
    validate_ond_art_report,
)

try:
    import jsonschema  # noqa: F401
    _JSONSCHEMA_AVAILABLE = True
except Exception:
    _JSONSCHEMA_AVAILABLE = False


def _sample_observations(seed: int = 0, count: int = 200):
    rows = generate_synthetic_signatures(count=count, seed=seed)
    return signatures_to_uruz(rows, SECP256K1_ORDER)


def test_compute_ond_art_metrics_structure():
    obs = _sample_observations()
    metrics = compute_ond_art_metrics(obs, modulus=SECP256K1_ORDER, bins=50)
    assert "rank" in metrics
    assert "subspace" in metrics
    assert "branching" in metrics
    assert 0.0 <= metrics["rank"]["entropy_norm"] <= 1.0
    assert 0.0 <= metrics["subspace"]["occupancy_ratio"] <= 1.0
    assert "entropy_axis" in metrics["rank"]
    assert "row_occupancy_mean" in metrics["subspace"]


def test_bootstrap_metrics_runs():
    obs = _sample_observations(seed=1)
    result = bootstrap_metrics(
        obs,
        lambda x: compute_ond_art_metrics(x, modulus=SECP256K1_ORDER, bins=50),
        samples=50,
        seed=42,
    )
    assert result.samples == 50
    assert len(result.metric_keys) > 0
    assert len(result.metrics_mean) == len(result.metric_keys)
    assert len(result.metrics_ci95) == len(result.metric_keys)


def test_build_ond_art_report_with_baseline():
    obs = _sample_observations(seed=7, count=200)
    baseline = _sample_observations(seed=8, count=200)
    report = build_ond_art_report(
        obs,
        modulus=SECP256K1_ORDER,
        bins=50,
        bootstrap_samples=50,
        bootstrap_seed=1,
        baseline_observations=baseline,
    )
    assert report["ond_art_version"] == "0.1"
    assert report["metrics"]["rank"]["entropy"] >= 0.0
    assert report["bootstrap"]["samples"] == 50
    assert report["baseline"] is not None
    assert "distance" in report["baseline"]
    if not _JSONSCHEMA_AVAILABLE:
        pytest.skip("jsonschema not available")
    validate_ond_art_report(report)
