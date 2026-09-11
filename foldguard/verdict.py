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

from .core import CONFIDENT, DISORDER_THRESHOLD, VERY_HIGH, Structure, Region


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
            "disorder_in_site": Level.WARN,
    },
    "docking": {
        "site_mean": 80.0,
        "global_trustworthy": 0.70,
        "label": "docking into a defined site",
            "disorder_in_site": Level.FAIL,
    },
    "md": {
        "site_mean": 70.0,
        "global_trustworthy": 0.70,
        "label": "molecular dynamics simulation",
            "disorder_in_site": Level.FAIL,
    },
    "mutation": {
        "site_mean": 90.0,
        "global_trustworthy": 0.70,
        "label": "interpreting a point mutation",
            "disorder_in_site": Level.FAIL,
    },
}


# Above this fraction, disorder away from the site is worth a caveat rather
# than a footnote. It is never itself a failure: where you are not working,
# a floppy region is biology, not a defect.
DISORDER_EXTENSIVE = 0.30

# Below this many residues outside the site, there is not enough of the model
# left to say anything about where the site sits in it. Both of these are
# judgement calls rather than facts, so both are exposed on the command line.
MIN_PAE_CORE = 10

# Runs shorter than this are treated as noise rather than a detached region.
PAE_MIN_RUN = 3


def _disorder_findings(
    structure: Structure, present: list[int], rules: dict
) -> list[Finding]:
    """Judge disorder relative to the site and the task.

    Failing a model because 35% of it is a floppy tail, when the pocket you
    intend to dock into is 50 residues away and perfectly resolved, is exactly
    the task-blind verdict this tool exists to replace. Disorder is decisive
    where you intend to work and contextual everywhere else.
    """
    regions = structure.disordered_regions(min_length=5)
    if not regions:
        return []

    site_set = set(present)
    inside, outside = [], []
    for region in regions:
        span = set(range(region.start, region.end + 1))
        (inside if site_set & span else outside).append(region)

    findings: list[Finding] = []

    if inside:
        n_hit = sum(len(site_set & set(range(r.start, r.end + 1))) for r in inside)
        findings.append(
            Finding(
                rules["disorder_in_site"],
                "Disordered residues inside the site",
                f"{n_hit} of the {len(site_set)} site residues fall in "
                f"{len(inside)} run(s) below pLDDT {DISORDER_THRESHOLD:.0f}: "
                f"{'; '.join(str(r) for r in inside[:3])}. Those coordinates are "
                f"arbitrary, so any pose or measurement that depends on them "
                f"means nothing - and this is the region you said you care about.",
            )
        )

    if outside:
        total = sum(r.length for r in outside)
        pct = total / structure.n_residues
        extensive = pct >= DISORDER_EXTENSIVE
        listed = "; ".join(str(r) for r in outside[:3])
        more = f" (+{len(outside) - 3} more)" if len(outside) > 3 else ""
        tail = (
            "That is a large share of the model; treat any interpretation "
            "beyond the site with care."
            if extensive
            else "None of it overlaps the region you asked about."
        )
        findings.append(
            Finding(
                Level.WARN if extensive else Level.PASS,
                "Disordered regions elsewhere in the model",
                f"{len(outside)} run(s) below pLDDT {DISORDER_THRESHOLD:.0f} "
                f"covering {pct:.0%} of the model: {listed}{more}. {tail}",
            )
        )

    return findings


def _pae_finding(
    structure: Structure,
    present: list[int],
    pae_cutoff: float,
    min_run: int = PAE_MIN_RUN,
    min_core: int = MIN_PAE_CORE,
) -> Finding:
    """Judge the site's placement against the rest of the model.

    This always returns a finding. The previous version skipped silently when
    no confident core existed, which produced a clean PASS on a model whose
    PAE was never looked at - the exact class of unchecked-but-confident result
    FoldGuard exists to prevent.
    """
    site_set = set(present)
    core = [r.number for r in structure.residues
            if r.plddt >= VERY_HIGH and r.number not in site_set]
    core_bar = f"pLDDT {VERY_HIGH:.0f}+"

    if not core:
        # No very-high core to anchor against. Fall back to merely confident
        # residues rather than skipping the check entirely.
        core = [r.number for r in structure.residues
                if r.trustworthy and r.number not in site_set]
        core_bar = f"pLDDT {CONFIDENT:.0f}+"

    if not core:
        return Finding(
            Level.WARN,
            "PAE supplied but no confident core to compare the site against",
            f"No residue outside the site reaches pLDDT {CONFIDENT:.0f}, so the "
            "site's placement relative to the model cannot be tested. The PAE "
            "matrix was read but could not be used. Treat every cross-region "
            "distance in this model as unverified.",
        )

    profile = structure.pae_profile(present, core)
    if not profile:
        return Finding(
            Level.WARN,
            "PAE supplied but could not be matched to the site",
            "No site/core residue pair mapped into the PAE matrix. The matrix "
            "and the model may describe different residue ranges.",
        )

    if len(core) < min_core:
        return Finding(
            Level.WARN,
            "Site covers too much of the model to check its placement",
            f"Only {len(core)} residue(s) at {core_bar} sit outside the site - "
            f"fewer than the {min_core} needed to judge the site against the "
            "rest of the model. There is nothing here to place the site relative "
            "to, so its arrangement is unverified rather than confirmed.",
        )

    segments = structure.pae_segments(present, core, pae_cutoff, min_run)
    worst_res = max(profile, key=lambda j: profile[j])
    pooled = sum(profile.values()) / len(profile)

    if segments:
        covered = sum(s.length for s in segments)
        if pooled > pae_cutoff:
            # The problem is model-wide; naming individual runs adds noise.
            where = (
                f"{covered} of {len(profile)} {core_bar} core residues sit above "
                f"{pae_cutoff:.1f} A PAE from the site, and the pooled mean is "
                f"{pooled:.1f} A. The site is poorly placed relative to the model "
                "as a whole."
            )
        else:
            listed = "; ".join(str(s) for s in segments[:3])
            more = f" (+{len(segments) - 3} more)" if len(segments) > 3 else ""
            where = (
                f"{covered} of {len(profile)} {core_bar} core residues sit above "
                f"{pae_cutoff:.1f} A PAE from the site: {listed}{more}. The pooled "
                f"mean is only {pooled:.1f} A, so an averaged check would have "
                "missed this."
            )
        return Finding(
            Level.WARN,
            "Site position relative to part of the model is uncertain",
            f"{where} Each part may be well predicted while their relative "
            "arrangement is not; treat cross-region geometry as unreliable.",
        )

    if pooled > pae_cutoff:
        # No run survived the noise filter, but the average is still over the
        # line. Reporting PASS here printed a pooled mean above the cutoff
        # inside a finding marked [ok].
        return Finding(
            Level.WARN,
            "Site position relative to the model is uncertain",
            f"No run of consecutive residues clears the noise filter, but the "
            f"pooled mean across the {core_bar} core is {pooled:.1f} A, above "
            f"the {pae_cutoff:.1f} A cutoff. Worst single residue is "
            f"{worst_res} at {profile[worst_res]:.1f} A. The uncertainty is "
            "scattered rather than localised, which is harder to reason about, "
            "not safer.",
        )

    return Finding(
        Level.PASS,
        "Site placement relative to core is consistent",
        f"No run of {core_bar} core residues exceeds {pae_cutoff:.1f} A PAE from "
        f"the site. Worst single residue is {worst_res} at "
        f"{profile[worst_res]:.1f} A; pooled mean {pooled:.1f} A.",
    )


def assess(
    structure: Structure,
    task: str = "fold",
    site: list[int] | None = None,
    pae_cutoff: float = 5.0,
    pae_min_run: int = PAE_MIN_RUN,
    pae_min_core: int = MIN_PAE_CORE,
) -> Report:
    """Produce a task-specific verdict on a predicted structure."""
    if task not in TASK_RULES:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(TASK_RULES)}.")

    rules = TASK_RULES[task]
    findings: list[Finding] = []
    by_num = structure.by_number()
    present: list[int] = [n for n in site if n in by_num] if site else []

    # Whether the site clears its own bar decides how much weight the global
    # score carries, so it has to be known before the global finding is written.
    site_res = [by_num[n] for n in present]
    site_mean = (sum(r.plddt for r in site_res) / len(site_res)) if site_res else None
    site_clears_bar = site_mean is not None and site_mean >= rules["site_mean"]

    # --- Global confidence ---
    frac = structure.fraction_trustworthy()
    if frac < rules["global_trustworthy"]:
        if site_clears_bar:
            # Most human proteins carry disordered regions. Refusing a
            # well-resolved pocket because a tail 200 residues away is floppy is
            # the task-blind verdict this tool exists to replace - real
            # AlphaFold models of p53 fail this way with a site at pLDDT 98.
            findings.append(
                Finding(
                    Level.WARN,
                    "Global confidence below task threshold",
                    f"Only {frac:.0%} of residues reach pLDDT {CONFIDENT:.0f}+, "
                    f"below the {rules['global_trustworthy']:.0%} usual for "
                    f"{rules['label']}. The site you asked about clears its own "
                    f"bar (mean {site_mean:.1f}), so this is context rather than "
                    "a veto: the model is weak elsewhere, not where you intend "
                    "to work.",
                )
            )
        else:
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
                findings.append(
                    _pae_finding(
                        structure, present, pae_cutoff,
                        pae_min_run, pae_min_core,
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

    findings.extend(_disorder_findings(structure, present, rules))

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
