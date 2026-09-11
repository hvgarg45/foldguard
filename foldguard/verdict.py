"""
Turn confidence numbers into a decision about a specific downstream task.

The point of this module: a model is not "good" or "bad" in the abstract. It is
adequate or inadequate *for something*. A model with a well-resolved core and a
floppy terminal tail is fine for describing the fold and useless for docking into
a pocket that sits in the tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .core import CONFIDENT, VERY_HIGH, Structure, Region


class Level(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Finding:
    level: Level
    title: str
    detail: str

    @property
    def marker(self) -> str:
        return {Level.PASS: "[ok]", Level.WARN: "[warn]", Level.FAIL: "[FAIL]"}[self.level]


@dataclass
class Report:
    structure: Structure
    findings: list[Finding]
    task: str
    site: list[int] | None = None

    @property
    def verdict(self) -> Level:
        if any(f.level is Level.FAIL for f in self.findings):
            return Level.FAIL
        if any(f.level is Level.WARN for f in self.findings):
            return Level.WARN
        return Level.PASS

    @property
    def exit_code(self) -> int:
        """0 pass, 1 warn, 2 fail. Lets this run as a gate in a pipeline."""
        return {Level.PASS: 0, Level.WARN: 1, Level.FAIL: 2}[self.verdict]


# Task-specific thresholds. These are deliberately explicit and easy to argue
# with: the defaults follow common practice, and every one can be overridden.
TASK_RULES = {
    "fold": {
        "site_mean": 60.0,
        "global_trustworthy": 0.50,
        "label": "describing the overall fold",
    },
    "docking": {
        "site_mean": 80.0,
        "global_trustworthy": 0.70,
        "label": "docking into a defined site",
    },
    "md": {
        "site_mean": 70.0,
        "global_trustworthy": 0.70,
        "label": "molecular dynamics simulation",
    },
    "mutation": {
        "site_mean": 90.0,
        "global_trustworthy": 0.70,
        "label": "interpreting a point mutation",
    },
}


def assess(
    structure: Structure,
    task: str = "fold",
    site: list[int] | None = None,
    pae_cutoff: float = 5.0,
) -> Report:
    """Produce a task-specific verdict on a predicted structure."""
    if task not in TASK_RULES:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(TASK_RULES)}.")

    rules = TASK_RULES[task]
    findings: list[Finding] = []
    by_num = structure.by_number()

    # --- Global confidence ---
    frac = structure.fraction_trustworthy()
    if frac < rules["global_trustworthy"]:
        findings.append(
            Finding(
                Level.FAIL,
                "Global confidence below task threshold",
                f"Only {frac:.0%} of residues reach pLDDT {CONFIDENT:.0f}+. "
                f"{rules['label'].capitalize()} needs at least "
                f"{rules['global_trustworthy']:.0%}.",
            )
        )
    else:
        findings.append(
            Finding(
                Level.PASS,
                "Global confidence adequate",
                f"{frac:.0%} of residues at pLDDT {CONFIDENT:.0f}+ "
                f"(mean {structure.mean_plddt:.1f}).",
            )
        )

    # --- The check people actually skip: site-specific confidence ---
    if site:
        present = [n for n in site if n in by_num]
        missing = [n for n in site if n not in by_num]

        if not present:
            findings.append(
                Finding(
                    Level.FAIL,
                    "Site residues not found in model",
                    f"None of the {len(site)} requested residues are present. "
                    "Check numbering: predicted models are often renumbered from 1.",
                )
            )
        else:
            if missing:
                findings.append(
                    Finding(
                        Level.WARN,
                        "Some site residues missing from model",
                        f"{len(missing)} of {len(site)} not found "
                        f"(e.g. {missing[:5]}). Numbering may be offset.",
                    )
                )

            site_res = [by_num[n] for n in present]
            site_mean = sum(r.plddt for r in site_res) / len(site_res)
            weakest = min(site_res, key=lambda r: r.plddt)
            n_untrusted = sum(1 for r in site_res if not r.trustworthy)

            if site_mean < rules["site_mean"]:
                findings.append(
                    Finding(
                        Level.FAIL,
                        "Site confidence below task threshold",
                        f"Mean pLDDT across the site is {site_mean:.1f}; "
                        f"{rules['label']} needs {rules['site_mean']:.0f}+. "
                        f"Weakest residue {weakest.name}{weakest.number} "
                        f"at {weakest.plddt:.1f}.",
                    )
                )
            elif n_untrusted:
                findings.append(
                    Finding(
                        Level.WARN,
                        "Site mean acceptable but individual residues are weak",
                        f"Mean {site_mean:.1f} passes, but {n_untrusted} of "
                        f"{len(site_res)} site residues sit below pLDDT "
                        f"{CONFIDENT:.0f}. Weakest is {weakest.name}"
                        f"{weakest.number} at {weakest.plddt:.1f}.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        Level.PASS,
                        "Site well resolved",
                        f"Mean pLDDT {site_mean:.1f} across {len(site_res)} "
                        f"residues; weakest {weakest.plddt:.1f}.",
                    )
                )

            # --- Site vs core: the multi-domain trap ---
            if structure.has_pae():
                core = [
                    r.number
                    for r in structure.residues
                    if r.plddt >= VERY_HIGH and r.number not in set(present)
                ]
                if core:
                    inter = structure.mean_pae_between(present, core)
                    if inter is not None and inter > pae_cutoff:
                        findings.append(
                            Finding(
                                Level.WARN,
                                "Site position relative to the rest of the model is uncertain",
                                f"Mean PAE between the site and the confident core "
                                f"is {inter:.1f} A (cutoff {pae_cutoff:.1f}). "
                                "Each part may be well predicted while their "
                                "relative arrangement is not. Treat cross-domain "
                                "geometry with caution.",
                            )
                        )
                    elif inter is not None:
                        findings.append(
                            Finding(
                                Level.PASS,
                                "Site placement relative to core is consistent",
                                f"Mean PAE {inter:.1f} A.",
                            )
                        )
    else:
        findings.append(
            Finding(
                Level.WARN,
                "No site specified",
                "Global confidence says little about the region you care about. "
                "Re-run with --site to check the pocket, interface or mutation "
                "position directly.",
            )
        )

    # --- Disorder ---
    disordered = structure.disordered_regions(min_length=5)
    if disordered:
        total = sum(d.length for d in disordered)
        pct = total / structure.n_residues
        level = Level.WARN if pct < 0.30 else Level.FAIL
        listed = "; ".join(str(d) for d in disordered[:4])
        more = f" (+{len(disordered) - 4} more)" if len(disordered) > 4 else ""
        findings.append(
            Finding(
                level,
                "Probable disordered regions",
                f"{len(disordered)} run(s) below pLDDT 50 covering {pct:.0%} "
                f"of the model: {listed}{more}. These coordinates are not "
                "meaningful and should not be interpreted structurally.",
            )
        )

    if not structure.has_pae():
        findings.append(
            Finding(
                Level.WARN,
                "No PAE supplied",
                "Without PAE, relative domain placement cannot be checked. "
                "A model can be confidently wrong about how two well-predicted "
                "domains sit against each other. Supply --pae to test this.",
            )
        )

    return Report(structure=structure, findings=findings, task=task, site=site)
