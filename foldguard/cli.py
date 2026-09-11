"""Command line interface for FoldGuard."""

from __future__ import annotations

import argparse
import json
import sys

from .core import ParseError, attach_pae, parse_site, parse_structure
from .verdict import TASK_RULES, Level, assess

BAR_WIDTH = 40


def _bar(fraction: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def _print_human(report) -> None:
    s = report.structure
    counts = s.band_counts()
    n = s.n_residues

    print()
    print("=" * 66)
    print(f"  FoldGuard  |  {s.source}")
    print("=" * 66)
    print(f"  Residues: {n}    Mean pLDDT: {s.mean_plddt:.1f}    Task: {report.task}")
    if report.site:
        print(f"  Site: {len(report.site)} residues specified")
    print()

    print("  Confidence distribution")
    labels = [
        ("very_high", "very high (90+)"),
        ("confident", "confident (70-90)"),
        ("low", "low (50-70)"),
        ("very_low", "very low (<50)"),
    ]
    for key, label in labels:
        frac = counts[key] / n
        print(f"    {label:<20} {_bar(frac)} {counts[key]:>5} ({frac:>4.0%})")
    print()

    print("  Findings")
    for f in report.findings:
        print(f"    {f.marker} {f.title}")
        for line in _wrap(f.detail, 58):
            print(f"          {line}")
    print()

    verdict = report.verdict
    banner = {
        Level.PASS: "VERDICT: PASS - model is adequate for this task",
        Level.WARN: "VERDICT: WARN - usable with stated caveats",
        Level.FAIL: "VERDICT: FAIL - not adequate for this task",
    }[verdict]
    print("=" * 66)
    print(f"  {banner}")
    print("=" * 66)
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _print_json(report) -> None:
    s = report.structure
    payload = {
        "source": s.source,
        "task": report.task,
        "n_residues": s.n_residues,
        "mean_plddt": round(s.mean_plddt, 2),
        "fraction_trustworthy": round(s.fraction_trustworthy(), 4),
        "band_counts": s.band_counts(),
        "site": report.site,
        "disordered_regions": [
            {"start": r.start, "end": r.end, "length": r.length,
             "mean_plddt": round(r.mean_plddt, 2)}
            for r in s.disordered_regions()
        ],
        "findings": [
            {"level": f.level.value, "title": f.title, "detail": f.detail}
            for f in report.findings
        ],
        "verdict": report.verdict.value,
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foldguard",
        description=(
            "Decide whether a predicted protein structure is good enough for "
            "what you are about to do with it."
        ),
        epilog=(
            "Exit codes: 0 pass, 1 warn, 2 fail. Suitable as a gate in a pipeline."
        ),
    )
    parser.add_argument("structure", help="Predicted structure (.pdb or .cif)")
    parser.add_argument(
        "--task",
        default="fold",
        choices=sorted(TASK_RULES),
        help="What you intend to do with the model (default: fold)",
    )
    parser.add_argument(
        "--site",
        help="Residues you care about, e.g. '45-52,88,120-124'",
    )
    parser.add_argument("--pae", help="PAE JSON from AlphaFold")
    parser.add_argument(
        "--pae-cutoff",
        type=float,
        default=5.0,
        help="Mean PAE (A) above which relative placement is flagged (default: 5.0)",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit 2)",
    )

    args = parser.parse_args(argv)

    try:
        structure = parse_structure(args.structure)
        if args.pae:
            attach_pae(structure, args.pae)
        site = parse_site(args.site) if args.site else None
        report = assess(
            structure, task=args.task, site=site, pae_cutoff=args.pae_cutoff
        )
    except ParseError as exc:
        print(f"foldguard: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"foldguard: {exc}", file=sys.stderr)
        return 3

    if args.json:
        _print_json(report)
    else:
        _print_human(report)

    code = report.exit_code
    if args.strict and code == 1:
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
