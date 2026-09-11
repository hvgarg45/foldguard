# FoldGuard

**Decide whether a predicted protein structure is good enough for what you are about to do with it.**

AlphaFold hands you a structure and a confidence score. It does not tell you whether the model is adequate for *your* purpose. FoldGuard does.

```bash
foldguard model.pdb --task docking --site 96-108 --pae model_pae.json
```

```
==================================================================
  FoldGuard  |  model.pdb
==================================================================
  Residues: 180    Mean pLDDT: 81.9    Task: docking
  Site: 13 residues specified

  Confidence distribution
    very high (90+)      ########################................   106 ( 59%)
    confident (70-90)    ########................................    38 ( 21%)
    low (50-70)          #.......................................     3 (  2%)
    very low (<50)       #######.................................    33 ( 18%)

  Findings
    [ok] Global confidence adequate
          80% of residues at pLDDT 70+ (mean 81.9).
    [FAIL] Site confidence below task threshold
          Mean pLDDT across the site is 48.4; docking into a defined
          site needs 80+. Weakest residue ALA107 at 42.1.
    [warn] Site position relative to the rest of the model is uncertain
          Mean PAE between the site and the confident core is 12.5 A
          (cutoff 5.0). Each part may be well predicted while their
          relative arrangement is not.

==================================================================
  VERDICT: FAIL - not adequate for this task
==================================================================
```

That model is 80% confident. It is also useless for docking, because the pocket sits in a disordered loop. Nothing in the AlphaFold output says so.

## The problem

A predicted structure is a hypothesis with a confidence attached. In practice the confidence gets checked once, globally, and then forgotten:

- **Global pLDDT hides local failure.** A model can be 85% confident overall while the binding site — the only part you care about — sits at 45. Docking into it produces poses that look plausible and mean nothing.
- **PAE gets ignored entirely.** Two domains can each be predicted beautifully while their *relative arrangement* is guesswork. Every cross-domain distance you measure is then fiction. pLDDT cannot tell you this; only PAE can.
- **Disordered regions get modelled anyway.** AlphaFold returns coordinates for disordered loops. They are not wrong exactly — they are meaningless, and they look identical to real coordinates in PyMOL.
- **Residue numbering silently shifts.** Predicted models are often renumbered from 1. Selecting "residues 340-350" on a renumbered model selects the wrong residues, quietly.

Each of these is well known and routinely skipped, because checking means opening a JSON, cross-referencing residue numbers, and making a judgement call. So it does not happen, and the cost lands weeks later as a docking campaign or an MD run that was never going to work.

## What FoldGuard does

It makes the check take five seconds, and it makes the answer task-specific.

**Task-aware thresholds.** A model good enough to describe a fold is not good enough to interpret a point mutation. The bar moves with the question:

| Task | Site mean pLDDT | Global trustworthy |
|------|-----------------|--------------------|
| `fold` | 60 | 50% |
| `md` | 70 | 70% |
| `docking` | 80 | 70% |
| `mutation` | 90 | 70% |

**Site-specific checking.** `--site 96-108` evaluates the residues you actually intend to use, not the average of the whole chain.

**PAE cross-checking.** With `--pae`, FoldGuard compares your site against the confident core and flags when their relative placement is unreliable — the failure mode pLDDT structurally cannot detect.

**Honest defaults.** No site specified? That is a warning, not a pass — global confidence says little about your region. No PAE supplied? Also a warning. A clean `PASS` requires you to have actually checked.

**Pipeline-ready.** Exit codes are `0` pass, `1` warn, `2` fail, so it gates a workflow:

```bash
foldguard model.pdb --task docking --site 96-108 --strict || exit 1
```

`--json` gives machine-readable output for logging or downstream tooling.

## Install

```bash
git clone https://github.com/<user>/foldguard
cd foldguard
pip install -e .
```

Pure standard library. No dependencies.

## Usage

```bash
# Is this model usable at all?
foldguard model.pdb

# Can I dock into this pocket?
foldguard model.pdb --task docking --site 45-52,88,120-124

# Can I trust this domain arrangement?
foldguard model.cif --task md --site 200-260 --pae pae.json

# Gate a pipeline
foldguard model.pdb --task docking --site 96-108 --strict --json
```

Accepts `.pdb` and `.cif` from AlphaFold2, AlphaFold3, ColabFold, ESMFold, or anything else that writes per-residue confidence into the B-factor column.

## Thresholds are arguments, not facts

The defaults follow common practice, not consensus — there is no universally agreed pLDDT cutoff for docking. They are deliberately visible in `verdict.py:TASK_RULES` so you can disagree with them, and `--pae-cutoff` is exposed on the command line.

The tool's value is not the specific numbers. It is that the check happens at all, against the region you care about, before the compute gets spent.

## Tests

```bash
pytest tests/ -q
```

29 tests. The ones that matter are at the bottom of `tests/test_foldguard.py`: they encode the failure mode this exists to catch — a globally confident model with a weak pocket must fail for docking and pass for fold description.

## Limitations

- pLDDT and PAE measure the predictor's confidence, not correctness. A confidently wrong prediction stays wrong. This narrows the failure space; it does not eliminate it.
- Single chain only. Multimer interface confidence (ipTM) is not yet handled.
- The disorder call is a pLDDT heuristic, not a dedicated disorder predictor.
- Numbering-mismatch detection is a heuristic: it catches selections that miss entirely, not ones that are offset by a few residues.

## License

MIT
