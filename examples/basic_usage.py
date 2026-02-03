"""
Basic usage example for IsogenyGuard SDK.
Generates synthetic secp256k1 signatures and runs TDA.
"""

from isogenyguard import (
    generate_synthetic_signatures,
    signatures_to_uruz,
    point_cloud_from_uruz,
    check_betti_numbers,
    build_report,
    SECP256K1_ORDER,
)

# Step 1: Generate synthetic signatures
rows = generate_synthetic_signatures(count=300, seed=7)

# Step 2: Convert signatures to (u_r, u_z)
uruz = signatures_to_uruz(rows, SECP256K1_ORDER)

# Step 3: Build a torus-embedded point cloud
points = point_cloud_from_uruz(uruz, SECP256K1_ORDER, embedding="torus")

# Step 4: Run TDA
result = check_betti_numbers(points, n_expected=2)
report = build_report(points, n_expected=2)

print("TDA summary:")
print(result)
print("Report:")
print(report)
