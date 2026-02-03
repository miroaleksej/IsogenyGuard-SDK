"""
ECDSA ingestion and synthetic dataset generation utilities.

This module targets secp256k1 for the prototype.
"""

from __future__ import annotations

import csv
import hashlib
import random
from typing import Iterable, List, Dict, Tuple, Optional

from ecdsa import SECP256k1, SigningKey, util

from .core import rsz_to_uruz

SECP256K1_ORDER = SECP256k1.order


def _randbytes(rng: random.Random, size: int) -> bytes:
    return rng.getrandbits(size * 8).to_bytes(size, "big")


def _parse_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.startswith("0x") or any(c in text for c in "abcdefABCDEF"):
        return int(text, 16)
    return int(text)


def generate_synthetic_signatures(
    count: int = 200,
    seed: int = 0,
    private_key: Optional[int] = None,
    digest_size: int = 32,
) -> List[Dict[str, int]]:
    """
    Generates a reproducible synthetic dataset of ECDSA signatures on secp256k1.

    Args:
        count: Number of signatures to generate.
        seed: RNG seed for reproducibility.
        private_key: Optional secret exponent to use.
        digest_size: Size of message digest in bytes.

    Returns:
        List of dicts with keys: r, s, z.
    """
    rng = random.Random(seed)
    n = SECP256K1_ORDER
    d = private_key or rng.randrange(1, n)
    sk = SigningKey.from_secret_exponent(d, curve=SECP256k1, hashfunc=hashlib.sha256)

    rows: List[Dict[str, int]] = []
    for _ in range(count):
        msg = _randbytes(rng, digest_size)
        digest = hashlib.sha256(msg).digest()
        sig = sk.sign_digest_deterministic(digest, sigencode=util.sigencode_string)
        r, s = util.sigdecode_string(sig, n)
        z = int.from_bytes(digest, "big")
        rows.append({"r": r, "s": s, "z": z})

    return rows


def signatures_to_uruz(
    signatures: Iterable[Dict[str, int]],
    n: int = SECP256K1_ORDER,
) -> List[Tuple[int, int]]:
    """
    Converts signatures (r, s, z) to (u_r, u_z).
    """
    points: List[Tuple[int, int]] = []
    for row in signatures:
        r = _parse_int(row["r"])
        s = _parse_int(row["s"])
        z = _parse_int(row["z"])
        points.append(rsz_to_uruz(r, s, z, n))
    return points


def save_signatures_csv(path: str, rows: Iterable[Dict[str, int]], as_hex: bool = True) -> None:
    """
    Saves signature rows to CSV. Columns: r, s, z.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["r", "s", "z"])
        writer.writeheader()
        for row in rows:
            r = row["r"]
            s = row["s"]
            z = row["z"]
            if as_hex:
                writer.writerow({"r": hex(r), "s": hex(s), "z": hex(z)})
            else:
                writer.writerow({"r": r, "s": s, "z": z})


def load_signatures_csv(path: str) -> List[Dict[str, int]]:
    """
    Loads signatures from a CSV with columns r, s, z.
    Accepts decimal or hex values.
    """
    rows: List[Dict[str, int]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "r": _parse_int(row["r"]),
                "s": _parse_int(row["s"]),
                "z": _parse_int(row["z"]),
            })
    return rows
