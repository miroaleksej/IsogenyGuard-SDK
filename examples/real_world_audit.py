"""
Streaming-style audit example using synthetic secp256k1 signatures.
"""

import time

from isogenyguard import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    point_cloud_from_uruz,
    check_betti_numbers,
    SECP256K1_ORDER,
)

print("Starting synthetic audit...\n")

try:
    while True:
        rows = generate_synthetic_signatures(count=200, seed=int(time.time()))
        uruz = signatures_to_uruz(rows, SECP256K1_ORDER)
        points = point_cloud_from_uruz(uruz, SECP256K1_ORDER)

        result = check_betti_numbers(points, n_expected=2)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Betti: b0={result['betti_0']}, b1={result['betti_1']}, b2={result['betti_2']}")
        print(f"[{timestamp}] Secure: {result['is_secure']}  Entropy: {result['topological_entropy']:.3f}\n")

        time.sleep(5)

except KeyboardInterrupt:
    print("\nAudit stopped by user.")
