"""Command-line interface for IsogenyGuard prototype."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .ecdsa import (
    generate_synthetic_signatures,
    load_signatures_csv,
    save_signatures_csv,
    signatures_to_uruz,
    SECP256K1_ORDER,
)
from .topology import point_cloud_from_uruz, build_report
from .ond_art import build_ond_art_report, validate_ond_art_report, flatten_report


def _cmd_generate(args: argparse.Namespace) -> int:
    rows = generate_synthetic_signatures(
        count=args.count,
        seed=args.seed,
        private_key=args.private_key,
    )
    save_signatures_csv(args.out, rows, as_hex=True)
    print(f"Saved {len(rows)} signatures to {args.out}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    rows = load_signatures_csv(args.input)
    uruz = signatures_to_uruz(rows, SECP256K1_ORDER)
    points = point_cloud_from_uruz(uruz, SECP256K1_ORDER, embedding=args.embedding)

    report = build_report(
        points,
        n_expected=args.n_expected,
        maxdim=args.maxdim,
        persistence_threshold=args.threshold,
        entropy_bins=args.entropy_bins,
    )

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"Report saved to {args.out}")

    print(json.dumps(report, indent=2))
    return 0


def _cmd_ond_art(args: argparse.Namespace) -> int:
    rows = load_signatures_csv(args.input)
    uruz = signatures_to_uruz(rows, SECP256K1_ORDER)
    args.validate = not args.no_validate

    baseline_uruz = None
    if args.baseline and args.baseline_synthetic is not None:
        raise ValueError("Use either --baseline or --baseline-synthetic, not both")
    if args.baseline:
        baseline_rows = load_signatures_csv(args.baseline)
        baseline_uruz = signatures_to_uruz(baseline_rows, SECP256K1_ORDER)
    elif args.baseline_synthetic is not None:
        if args.baseline_synthetic == "auto":
            baseline_count = len(rows)
        else:
            try:
                baseline_count = int(args.baseline_synthetic)
            except Exception as exc:
                raise ValueError("baseline-synthetic must be an integer or 'auto'") from exc
            if baseline_count <= 0:
                baseline_count = len(rows)
        baseline_seed = args.baseline_seed if args.baseline_seed is not None else args.seed + 1
        baseline_rows = generate_synthetic_signatures(
            count=baseline_count,
            seed=baseline_seed,
            private_key=None,
        )
        baseline_uruz = signatures_to_uruz(baseline_rows, SECP256K1_ORDER)

    try:
        percentiles = tuple(float(x.strip()) for x in args.baseline_percentiles.split(","))
    except Exception as exc:
        raise ValueError("baseline-percentiles must be a comma-separated list of three numbers") from exc
    if len(percentiles) != 3:
        raise ValueError("baseline-percentiles must have three values: green,yellow,red")

    report = build_ond_art_report(
        uruz,
        modulus=SECP256K1_ORDER,
        bins=args.bins,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.seed,
        baseline_observations=baseline_uruz,
        baseline_percentiles=percentiles,
    )

    if args.validate:
        validate_ond_art_report(report)

    out_format = args.out_format
    if out_format == "json":
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
            print(f"Report saved to {args.out}")
        print(json.dumps(report, indent=2))
        return 0

    if out_format == "jsonl":
        line = json.dumps(report)
        if args.out:
            mode = "a" if args.append else "w"
            with open(args.out, mode) as handle:
                handle.write(line + "\n")
            print(f"Report saved to {args.out}")
        print(line)
        return 0

    if out_format == "csv":
        row = flatten_report(report)
        if args.out:
            mode = "a" if args.append else "w"
            with open(args.out, mode, newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(row.keys()))
                if mode == "w":
                    writer.writeheader()
                writer.writerow(row)
            print(f"Report saved to {args.out}")
        else:
            writer = csv.DictWriter(sys.stdout, fieldnames=sorted(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return 0

    raise ValueError(f"Unsupported out-format: {out_format}")


def _cmd_ond_art_fixture(args: argparse.Namespace) -> int:
    args.validate = not args.no_validate
    rows = generate_synthetic_signatures(
        count=args.count,
        seed=args.seed,
        private_key=None,
    )
    uruz = signatures_to_uruz(rows, SECP256K1_ORDER)

    baseline_rows = generate_synthetic_signatures(
        count=args.baseline_count,
        seed=args.baseline_seed,
        private_key=None,
    )
    baseline_uruz = signatures_to_uruz(baseline_rows, SECP256K1_ORDER)

    report = build_ond_art_report(
        uruz,
        modulus=SECP256K1_ORDER,
        bins=args.bins,
        bootstrap_samples=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        baseline_observations=baseline_uruz,
    )

    if args.validate:
        validate_ond_art_report(report)

    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out, "r") as handle:
            existing = json.load(handle)
        if existing == report:
            print(f"Fixture matches: {args.out}")
            return 0
        raise FileExistsError(f"Fixture differs: {args.out} (use --overwrite to update)")

    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"Fixture report saved to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isogenyguard")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic secp256k1 signatures")
    gen.add_argument("--count", type=int, default=200, help="Number of signatures")
    gen.add_argument("--seed", type=int, default=0, help="RNG seed")
    gen.add_argument("--private-key", type=int, default=None, help="Secret exponent")
    gen.add_argument("--out", required=True, help="Output CSV path")
    gen.set_defaults(func=_cmd_generate)

    analyze = sub.add_parser("analyze", help="Analyze signatures with TDA")
    analyze.add_argument("--input", required=True, help="Input CSV path")
    analyze.add_argument("--out", help="Output JSON path")
    analyze.add_argument("--embedding", default="torus", choices=["torus", "square"])
    analyze.add_argument("--n-expected", type=int, default=2, help="Expected Betti_1")
    analyze.add_argument("--maxdim", type=int, default=2)
    analyze.add_argument("--threshold", type=float, default=None)
    analyze.add_argument("--entropy-bins", type=int, default=30, help="Bins for entropy histogram")
    analyze.set_defaults(func=_cmd_analyze)

    ond = sub.add_parser("ond-art", help="Generate OND-ART JSON report")
    ond.add_argument("--input", required=True, help="Input CSV path")
    ond.add_argument("--baseline", help="Baseline CSV path")
    ond.add_argument(
        "--baseline-synthetic",
        nargs="?",
        const="auto",
        default=None,
        help="Generate synthetic baseline (count or 'auto')",
    )
    ond.add_argument("--baseline-seed", type=int, default=None, help="Seed for synthetic baseline")
    ond.add_argument("--baseline-percentiles", type=str, default="95,99,99.5", help="Percentiles for green,yellow,red thresholds")
    ond.add_argument("--out", help="Output JSON path")
    ond.add_argument("--bins", type=int, default=100, help="Quantization bins")
    ond.add_argument("--bootstrap", type=int, default=200, help="Bootstrap samples")
    ond.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    ond.add_argument("--out-format", choices=["json", "jsonl", "csv"], default="json", help="Output format")
    ond.add_argument("--append", action="store_true", help="Append to output file for jsonl/csv")
    ond.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    ond.set_defaults(func=_cmd_ond_art)

    fixture = sub.add_parser("ond-art-fixture", help="Generate a deterministic OND-ART fixture report")
    fixture.add_argument("--out", required=True, help="Output JSON path")
    fixture.add_argument("--count", type=int, default=80, help="Synthetic sample count")
    fixture.add_argument("--seed", type=int, default=123, help="Synthetic seed")
    fixture.add_argument("--baseline-count", type=int, default=80, help="Baseline sample count")
    fixture.add_argument("--baseline-seed", type=int, default=124, help="Baseline seed")
    fixture.add_argument("--bins", type=int, default=40, help="Quantization bins")
    fixture.add_argument("--bootstrap", type=int, default=30, help="Bootstrap samples")
    fixture.add_argument("--bootstrap-seed", type=int, default=7, help="Bootstrap seed")
    fixture.add_argument("--no-validate", action="store_true", help="Disable schema validation")
    fixture.add_argument("--overwrite", action="store_true", help="Overwrite existing fixture")
    fixture.set_defaults(func=_cmd_ond_art_fixture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
