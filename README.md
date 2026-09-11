# FoldGuard

**Decide whether a predicted protein structure is good enough for what you are about to do with it.**

AlphaFold hands you a structure and a confidence score. It does not tell you whether the model is adequate for *your* purpose. FoldGuard does.

```bash
foldguard model.pdb --task docking --site 96-108 --pae model_pae.json
```

```
==================================================================
  FoldGuard  |  examples/model.pdb
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
    [warn] Site position relative to part of the model is uncertain
          89 of 106 pLDDT 90+ core residues sit above 5.0 A PAE from
          the site, and the pooled mean is 13.7 A. The site is
          poorly placed relative to the model as a whole. Each part
          may be well predicted while their relative arrangement is
          not; treat cross-region geometry as unreliable.
    [FAIL] Disordered residues inside the site
          5 of the 13 site residues fall in 1 run(s) below pLDDT 50:
          100-104 (5 res, mean pLDDT 46.1). Those coordinates are
          arbitrary, so any pose or measurement that depends on them
          means nothing - and this is the region you said you care
          about.
    [ok] Disordered regions elsewhere in the model
          2 run(s) below pLDDT 50 covering 13% of the model: 1-14
          (14 res, mean pLDDT 33.2); 172-180 (9 res, mean pLDDT
          40.7). None of it overlaps the region you asked about.

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

| Task | Site mean pLDDT | Global trustworthy | Disorder in the site |
|------|-----------------|--------------------|----------------------|
| `fold` | 60 | 50% | warn |
| `md` | 70 | 70% | fail |
| `docking` | 80 | 70% | fail |
| `mutation` | 90 | 70% | fail |

**Site-specific checking.** `--site 96-108` evaluates the residues you actually intend to use, not the average of the whole chain.

**PAE cross-checking.** With `--pae`, FoldGuard compares your site against the confident core and flags when their relative placement is unreliable — the failure mode pLDDT structurally cannot detect. The comparison is made per residue and then grouped into contiguous runs, not pooled into one average: a 20-residue domain floating at 28 A disappears into the mean of a 200-residue model, so the finding names the residues your site is actually uncertain against. PAE is asymmetric, and the worse of the two directions is used.

**Disorder judged where it matters.** A disordered run overlapping your site is decisive, and graded by task. Disorder elsewhere is reported and never fails the model — a floppy terminal tail is real biology, and failing a docking run because of one 50 residues from the pocket is exactly the task-blind verdict this tool exists to replace.

**Global confidence is context, not a veto.** If you name a site and it clears its task threshold, a weak global score becomes a warning rather than a failure. Most human proteins carry disordered regions; refusing a pocket at pLDDT 98 because a tail 200 residues away is floppy is the same task-blind judgement this tool exists to replace.

**Honest defaults.** No site specified? That is a warning, not a pass — global confidence says little about your region. No PAE supplied? Also a warning. A clean `PASS` requires you to have actually checked.

**Pipeline-ready.** Exit codes are `0` pass, `1` warn, `2` fail — and, deliberately outside that range, `3` bad input (including a mistyped flag) and `4` foldguard itself failed. Only `0/1/2` are claims about the model: a crash can never be read as "usable with caveats". So it gates a workflow:

```bash
foldguard model.pdb --task docking --site 96-108 --strict || exit 1
```

`--json` gives machine-readable output for logging or downstream tooling.

## Install

```bash
git clone https://github.com/hvgarg45/foldguard
cd foldguard
pip install -e .
```

Pure standard library. No dependencies. Tested on Python 3.9 and 3.13.

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

**It refuses what it cannot read honestly.** Every check FoldGuard makes assumes that a residue number identifies one residue and that the B-factor column really holds pLDDT. Where that assumption breaks, the tool stops with exit 3 and says why, rather than producing a confident verdict from misread numbers:

| Input | Why it is refused |
|-------|-------------------|
| More than one chain | Residue number is not unique, so every per-residue check would mix the chains |
| More than one model | Keeping the first silently discards the rest |
| Insertion codes (`100`, `100A`) | Three residues would collapse into one |
| Every pLDDT `0.00` | The signature of a misaligned B-factor column |
| Every pLDDT `≤ 1.0` | pLDDT written on a 0–1 scale, not 0–100 |
| Any pLDDT outside `0–100` | Whatever that column holds, it is not pLDDT |

For `.cif`, residue numbers come from `auth_seq_id` — the numbering you read off a paper or a PDB entry — falling back to `label_seq_id` only when it is absent.

## Thresholds are arguments, not facts

The defaults follow common practice, not consensus — there is no universally agreed pLDDT cutoff for docking. The task thresholds are deliberately visible in `verdict.py:TASK_RULES` so you can disagree with them, and every PAE judgement call is on the command line:

| Flag | Default | What it decides |
|------|---------|-----------------|
| `--pae-cutoff` | 5.0 | PAE (Å) above which a residue counts as badly placed relative to the site |
| `--pae-min-run` | 3 | consecutive such residues before a region is flagged; shorter runs are noise |
| `--pae-min-core` | 10 | confident residues needed outside the site before its placement can be judged at all |

Three independent guards decide whether placement is reported as consistent: a contiguous run above the cutoff, the pooled mean above the cutoff, and the *share* of core residues above it (15%). The third exists because uncertainty that is scattered rather than clustered clears the first, and if it is mild it never moves the second — half the core could sit above the cutoff and still be called consistent. All three values must be positive and finite: `--pae-cutoff nan` would otherwise make every comparison false, disabling the check while reporting it as passed.

**Loosening them has a cost, and it is worth seeing.** On the real p53 model, the tetramerization domain sits 19 Å from the DNA-binding domain — a 21-residue run:

```bash
foldguard p53.pdb --task mutation --site 273 --pae pae.json --pae-min-run 21
#   [warn] Site position relative to part of the model is uncertain
foldguard p53.pdb --task mutation --site 273 --pae pae.json --pae-min-run 22
#   [ok]   Site placement relative to core is consistent
```

One residue past the length of the thing you wanted to detect, and it is gone — the pooled mean is 4.2 Å, comfortably under any cutoff. Raise `--pae-min-run` only above the size of a region you would be willing to miss.

`--json` records every threshold a run used, so a logged verdict can be checked against the numbers that produced it.

The tool's value is not the specific numbers. It is that the check happens at all, against the region you care about, before the compute gets spent.

## Tests

```bash
pytest tests/ -q
```

111 tests. The ones that matter are at the bottom of `tests/test_foldguard.py`: they encode the failure modes this exists to catch — a globally confident model with a weak pocket must fail for docking and pass for fold description, and a PAE check that cannot run must warn rather than pass silently.

## Validated against real models

Checked against AlphaFold DB v6 downloads rather than only synthetic fixtures: **p53** (`P04637`, 393 residues, 30% very-low pLDDT — a well-folded DNA-binding domain between disordered termini) and **lysozyme** (`P00698`, 147 residues, single compact domain). Both formats parse identically, `.pdb` and `.cif` agree on every finding, and no validation rule fires spuriously on genuine input.

The multi-domain check earns its keep there. Asked about the R273 hotspot mutation, FoldGuard reports:

> 26 of 206 pLDDT 90+ core residues sit above 5.0 Å PAE from the site: **328-348 (21 res, mean PAE 19.0 Å)**; 224-228 (5 res, mean PAE 7.2 Å). The pooled mean is only 4.2 Å, so an averaged check would have missed this.

Residues 328–348 are the tetramerization domain; R273 is in the DNA-binding domain. AlphaFold genuinely cannot place the two relative to each other, and the pooled mean — 4.2 Å, under the cutoff — hides it completely.

## Limitations

- pLDDT and PAE measure the predictor's confidence, not correctness. A confidently wrong prediction stays wrong. This narrows the failure space; it does not eliminate it.
- Single chain only, and enforced: a model with more than one chain is refused with exit 3 rather than analysed. Residue number alone is not a unique identity in a multimer, so every per-residue check would silently mix the chains. Multimer interface confidence (ipTM) is not handled either.
- The PAE check needs at least `--pae-min-core` confident residues outside your site to judge placement against. If your site covers nearly the whole model there is nothing left to compare it to, and FoldGuard says so rather than reporting a placement it never checked.
- The disorder call is a pLDDT heuristic, not a dedicated disorder predictor. It is judged against your site: overlapping runs are graded by task, and runs elsewhere are reported without failing the model.
- Numbering-mismatch detection is a heuristic: it catches selections that miss entirely, not ones that are offset by a few residues.
- The pLDDT sanity checks catch misread columns and wrong scales, but they cannot catch a real crystal structure's B-factors: those sit inside 0–100 and are indistinguishable from pLDDT by value alone. FoldGuard trusts you that the file is a prediction.
- Alternate conformations (altLoc) are not refused; the first is taken, as before.
- `--site` selections are capped at 100,000 residues, so a mistyped range is rejected instead of allocating gigabytes.
- A supplied PAE matrix must match the model's residue count exactly, and is rejected otherwise: a matrix from a different prediction is valid JSON and produces confident nonsense. The check is on dimensions only, so it cannot tell two same-sized models apart.

## License

MIT
