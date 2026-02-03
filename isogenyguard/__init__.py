"""
IsogenyGuard SDK — Topological Auditing of Cryptographic Keys

Based on scientific research:
- Theorem 9: Private key recovery from special points
- Theorem 21: Isogeny space ≃ (n-1)-dimensional torus
- Theorem 16: AdaptiveTDA preserves sheaf cohomologies
- Theorem 7 & Table 3: Betti numbers and topological entropy as security metrics

"Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
"""

from .core import recover_private_key, rsz_to_uruz, uruz_to_rsz
from .ecdsa import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    load_signatures_csv,
    save_signatures_csv,
    SECP256K1_ORDER,
)
from .topology import (
    point_cloud_from_uruz,
    check_betti_numbers,
    calculate_topological_entropy,
    build_report,
)
from .visualization import (
    quantize_linear_value,
    quantize_linear,
    quantize_uruz,
    heatmap_counts,
)
from .ond_art import (
    compute_ond_art_metrics,
    bootstrap_metrics,
    build_ond_art_report,
    validate_ond_art_report,
    flatten_report,
)
from .utils import validate_implementation

version = "0.2.0"

__all__ = [
    "recover_private_key",
    "rsz_to_uruz",
    "uruz_to_rsz",
    "generate_synthetic_signatures",
    "signatures_to_uruz",
    "load_signatures_csv",
    "save_signatures_csv",
    "SECP256K1_ORDER",
    "point_cloud_from_uruz",
    "check_betti_numbers",
    "calculate_topological_entropy",
    "build_report",
    "quantize_linear_value",
    "quantize_linear",
    "quantize_uruz",
    "heatmap_counts",
    "compute_ond_art_metrics",
    "bootstrap_metrics",
    "build_ond_art_report",
    "validate_ond_art_report",
    "flatten_report",
    "validate_implementation",
    "version"
]


def info():
    """
    Returns SDK version and scientific foundation.
    """
    return f"""
    IsogenyGuard SDK v{version}
    Topological auditing of cryptographic keys based on Betti numbers analysis.
    "Topology is not a hacking tool, but a microscope for vulnerability diagnostics."
    """
