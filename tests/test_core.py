"""
Core ECDSA transform tests for IsogenyGuard.
"""

from isogenyguard.core import rsz_to_uruz
from isogenyguard.ecdsa import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    SECP256K1_ORDER,
)


def test_generate_synthetic_signatures_reproducible():
    rows_a = generate_synthetic_signatures(count=10, seed=123)
    rows_b = generate_synthetic_signatures(count=10, seed=123)
    assert rows_a == rows_b


def test_signatures_to_uruz_matches_core():
    rows = generate_synthetic_signatures(count=5, seed=99)
    uruz = signatures_to_uruz(rows, SECP256K1_ORDER)

    for row, (ur, uz) in zip(rows, uruz):
        ur2, uz2 = rsz_to_uruz(row["r"], row["s"], row["z"], SECP256K1_ORDER)
        assert (ur, uz) == (ur2, uz2)
        assert 0 <= ur < SECP256K1_ORDER
        assert 0 <= uz < SECP256K1_ORDER
