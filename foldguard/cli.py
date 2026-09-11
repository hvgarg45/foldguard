"""Command line interface for FoldGuard."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core import ParseError, attach_pae, parse_site, parse_structure
from .verdict import MIN_PAE_CORE, PAE_MIN_RUN, TASK_RULES, Level, assess

BAR_WIDTH = 40

# Reserved for "foldguard itself failed", outside the 0/1/2 verdict range.
EXIT_INTERNAL = 4
EXIT_BAD_INPUT = 3


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that exits 3, not 2, on a usage error.

    argparse's default is sys.exit(2), and 2 is this tool's FAIL code - so a
    mistyped flag would report the model as inadequate for the task.
    """

    def error(self, message: str):  # noqa: D102
        self.print_usage(sys.stderr)
        print(f"foldguard: {message}", file=sys.stderr)
        raise SystemExit(EXIT_BAD_INPUT)


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
        if not cur:
            # A token wider than `width` still starts a line; flushing an empty
            # accumulator first would emit a blank line into the findings.
            cur = w
        elif len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}"
    if cur:
        lines.append(cur)
    return lines


def _print_json(report, thresholds: dict | None = None) -> None:
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
        # Recorded so a logged verdict can be reproduced: without them a run in
        # a pipeline log cannot be checked against the thresholds it used.
        "thresholds": thresholds or {},
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(
        prog="foldguard",
        description=(
            "Decide whether a predicted protein structure is good enough for "
            "what you are about to do with it."
        ),
        epilog=(
            "Exit codes: 0 pass, 1 warn, 2 fail, 3 bad input, 4 foldguard failed. "
            "Only 0/1/2 are claims about the model, so a broken run can never "
            "be mistaken for a verdict. Suitable as a gate in a pipeline."
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
    parser.add_argument(
        "--pae-min-run",
        type=int,
        default=PAE_MIN_RUN,
        metavar="N",
        help=(
            "Consecutive residues that must exceed --pae-cutoff before a region "
            f"is flagged; shorter runs are treated as noise (default: {PAE_MIN_RUN})"
        ),
    )
    parser.add_argument(
        "--pae-min-core",
        type=int,
        default=MIN_PAE_CORE,
        metavar="N",
        help=(
            "Confident residues required outside the site before its placement "
            f"can be judged at all (default: {MIN_PAE_CORE})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit 2)",
    )

    args = parser.parse_args(argv)

    for name, value in (
        ("--pae-cutoff", args.pae_cutoff),
        ("--pae-min-run", args.pae_min_run),
        ("--pae-min-core", args.pae_min_core),
    ):
        if value <= 0:
            print(f"foldguard: {name} must be positive (got {value})", file=sys.stderr)
            return EXIT_BAD_INPUT

    try:
        structure = parse_structure(args.structure)
        if args.pae:
            attach_pae(structure, args.pae)
        site = parse_site(args.site) if args.site else None
        report = assess(
            structure,
            task=args.task,
            site=site,
            pae_cutoff=args.pae_cutoff,
            pae_min_run=args.pae_min_run,
            pae_min_core=args.pae_min_core,
        )
    except ParseError as exc:
        print(f"foldguard: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"foldguard: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all
        # Exit codes 0/1/2 are claims about the model. An unexpected failure is
        # not a claim about anything, and must not be mistakable for one:
        # Python's default exit code for an uncaught exception is 1, which this
        # tool defines as WARN.
        print(
            f"foldguard: internal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL

    try:
        if args.json:
            _print_json(
                report,
                {
                    "pae_cutoff": args.pae_cutoff,
                    "pae_min_run": args.pae_min_run,
                    "pae_min_core": args.pae_min_core,
                    "site_mean": TASK_RULES[args.task]["site_mean"],
                    "global_trustworthy": TASK_RULES[args.task]["global_trustworthy"],
                },
            )
        else:
            _print_human(report)
        # Flush inside the guard: buffered output that fails at interpreter
        # shutdown instead reports exit 120, overriding whatever we return.
        sys.stdout.flush()
    except BrokenPipeError:
        # The consumer stopped reading (`foldguard ... | head`). Point stdout at
        # devnull so the shutdown flush cannot fail as well.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_INTERNAL
    except Exception as exc:  # noqa: BLE001 - output must not fake a verdict
        print(
            f"foldguard: internal error while writing output: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL

    code = report.exit_code
    if args.strict and code == 1:
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
