"""
Topological analysis tests using ripser.
"""

from isogenyguard.ecdsa import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    SECP256K1_ORDER,
)
from isogenyguard.topology import (
    point_cloud_from_uruz,
    calculate_persistence_diagram,
    check_betti_numbers,
    calculate_topological_entropy,
)


def _build_point_cloud(seed: int = 3):
    rows = generate_synthetic_signatures(count=200, seed=seed)
    uruz = signatures_to_uruz(rows, SECP256K1_ORDER)
    return point_cloud_from_uruz(uruz, SECP256K1_ORDER, embedding="torus")


def test_persistence_diagram_shapes():
    points = _build_point_cloud()
    diagrams = calculate_persistence_diagram(points, maxdim=2)
    assert len(diagrams) == 3


def test_check_betti_numbers_summary():
    points = _build_point_cloud(seed=5)
    result = check_betti_numbers(points, n_expected=2)
    assert result["betti_0"] == 1
    assert "betti_1" in result
    assert "betti_2" in result
    assert "topological_entropy" in result
    assert "topological_entropy_norm" in result
    assert 0.0 <= result["topological_entropy_norm"] <= 1.0


def test_entropy_normalization():
    points = _build_point_cloud(seed=11)
    raw = calculate_topological_entropy(points, bins=30, normalize=False)
    norm = calculate_topological_entropy(points, bins=30, normalize=True)
    assert raw >= 0.0
    assert 0.0 <= norm <= 1.0
