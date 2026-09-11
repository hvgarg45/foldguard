"""
Core parsing and confidence analysis for predicted protein structures.

AlphaFold and related predictors write per-residue confidence (pLDDT) into the
B-factor column of the output PDB. PAE (predicted aligned error) is written to a
separate JSON. This module reads both and turns them into judgements about
whether a model is safe to use for a given downstream task.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# pLDDT bands as defined by DeepMind.
# https://alphafold.ebi.ac.uk/faq
VERY_HIGH = 90.0
CONFIDENT = 70.0
LOW = 50.0

# Below this, a region is more likely disordered than merely uncertain.
DISORDER_THRESHOLD = 50.0

# pLDDT is a percentage. Anything outside this did not come from the column we
# think we read.
PLDDT_MIN = 0.0
PLDDT_MAX = 100.0

# The largest known protein (titin) is about 35,000 residues. A selection an
# order of magnitude beyond that is a typo, and expanding it costs gigabytes.
MAX_SITE_RESIDUES = 100_000


class ParseError(Exception):
    """Raised when an input file cannot be read as expected."""


@dataclass
class Residue:
    """A single residue with its predicted confidence."""

    number: int
    name: str
    chain: str
    plddt: float

    @property
    def band(self) -> str:
        if self.plddt >= VERY_HIGH:
            return "very_high"
        if self.plddt >= CONFIDENT:
            return "confident"
        if self.plddt >= LOW:
            return "low"
        return "very_low"

    @property
    def trustworthy(self) -> bool:
        """Confident or better. The usual bar for structural interpretation."""
        return self.plddt >= CONFIDENT


@dataclass
class Region:
    """A contiguous run of residues sharing a confidence band."""

    start: int
    end: int
    band: str
    mean_plddt: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end} ({self.length} res, mean pLDDT {self.mean_plddt:.1f})"


def _validate_plddt(residues: list[Residue], source: str) -> None:
    """Refuse values that cannot be pLDDT.

    The B-factor column is read by fixed offset, so a file whose columns are
    even slightly off still parses - it just yields the wrong numbers, and a
    confident verdict computed from them. These checks catch the three shapes
    that failure actually takes. They cannot catch everything: a real crystal
    structure's B-factors sit inside 0-100 and are indistinguishable from pLDDT
    by value alone.
    """
    values = [r.plddt for r in residues]

    non_finite = [v for v in values if not math.isfinite(v)]
    if non_finite:
        raise ParseError(
            f"{source} has confidence values that are not a number (NaN or "
            "infinity). Comparisons against them are all false, so every "
            "threshold check would silently pass."
        )

    out_of_range = [v for v in values if v < PLDDT_MIN or v > PLDDT_MAX]
    if out_of_range:
        raise ParseError(
            f"{source} has confidence values outside {PLDDT_MIN:.0f}-"
            f"{PLDDT_MAX:.0f} (e.g. {out_of_range[0]:.2f}). pLDDT is a "
            "percentage, so this column is not pLDDT - check the file really "
            "carries per-residue confidence in the B-factor column."
        )

    if all(v == 0.0 for v in values):
        raise ParseError(
            f"Every residue in {source} has a confidence of 0.00. That is the "
            "signature of a misaligned B-factor column: FoldGuard reads columns "
            "61-66 of each ATOM record, and a file written a few characters "
            "narrow yields zeros instead of failing. Check the file's column "
            "alignment against the PDB format."
        )

    if all(v <= 1.0 for v in values):
        raise ParseError(
            f"Every residue in {source} has a confidence of 1.00 or less. This "
            "looks like pLDDT written on a 0-1 scale, which some ColabFold and "
            "ESMFold writers emit. FoldGuard expects the 0-100 scale; multiply "
            "the column by 100, or re-export with 0-100 confidence."
        )


@dataclass
class PaeSegment:
    """Mean PAE between a residue set and one contiguous run of the model.

    Kept per-run rather than pooled so that a single detached domain stays
    visible instead of being averaged into the bulk of a large model.
    """

    start: int
    end: int
    mean_pae: float
    n_pairs: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.start}-{self.end} ({self.length} res, mean PAE {self.mean_pae:.1f} A)"


@dataclass
class Structure:
    """A parsed predicted structure."""

    residues: list[Residue]
    source: str
    pae: list[list[float]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.residues:
            raise ParseError(f"No residues with confidence values found in {self.source}")

        _validate_plddt(self.residues, self.source)

        chains = sorted({r.chain for r in self.residues})
        if len(chains) > 1:
            # Residue number alone is not an identity in a multimer. Allowing
            # this through would drop one chain from by_number(), interleave the
            # chains in _runs() so disorder detection finds nothing, and index
            # the PAE matrix meaninglessly. Those are wrong answers, not
            # uncertain ones, so they are refused rather than warned about.
            raise ParseError(
                f"{self.source} contains more than one chain "
                f"({', '.join(chains)}). FoldGuard analyses a single chain at a "
                "time; on a multimer every per-residue check would silently mix "
                "the chains together. Extract the chain you care about and run "
                "FoldGuard on that."
            )

        if self.pae is not None:
            # Validate here as well as in attach_pae, so that a Structure holding
            # a PAE matrix always holds a usable one however it was built.
            self.pae = _validate_pae(self.pae, len(self.residues), self.source)

    @property
    def mean_plddt(self) -> float:
        return sum(r.plddt for r in self.residues) / len(self.residues)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def by_number(self) -> dict[int, Residue]:
        return {r.number: r for r in self.residues}

    def band_counts(self) -> dict[str, int]:
        counts = {"very_high": 0, "confident": 0, "low": 0, "very_low": 0}
        for r in self.residues:
            counts[r.band] += 1
        return counts

    def fraction_trustworthy(self) -> float:
        return sum(1 for r in self.residues if r.trustworthy) / len(self.residues)

    def disordered_regions(self, min_length: int = 5) -> list[Region]:
        """Runs of >=min_length residues below the disorder threshold."""
        return self._runs(lambda r: r.plddt < DISORDER_THRESHOLD, min_length)

    def _runs(self, predicate, min_length: int) -> list[Region]:
        regions: list[Region] = []
        current: list[Residue] = []

        def flush() -> None:
            if len(current) >= min_length:
                mean = sum(r.plddt for r in current) / len(current)
                regions.append(
                    Region(
                        start=current[0].number,
                        end=current[-1].number,
                        band=current[0].band,
                        mean_plddt=mean,
                    )
                )

        prev_num = None
        for r in sorted(self.residues, key=lambda x: x.number):
            if predicate(r) and (prev_num is None or r.number == prev_num + 1 or not current):
                current.append(r)
            elif predicate(r):
                flush()
                current = [r]
            else:
                flush()
                current = []
            prev_num = r.number
        flush()
        return regions

    # ---- PAE ----

    def has_pae(self) -> bool:
        return self.pae is not None

    def _pae_index(self) -> dict[int, int]:
        """Residue number -> row/column in the PAE matrix.

        The matrix is indexed by position in the modelled sequence, so this is
        only meaningful once the matrix has been validated against the
        structure in attach_pae().
        """
        return {r.number: i for i, r in enumerate(sorted(self.residues, key=lambda x: x.number))}

    def _pair_pae(self, ii: int, jj: int) -> float:
        """PAE for one residue pair, taking the worse of the two directions.

        PAE is asymmetric: pae[i][j] is the expected error in residue i's
        position when the prediction is aligned on residue j, and the two
        directions can differ sharply. For deciding whether a relative
        arrangement is trustworthy, the pessimistic direction is the honest one.
        """
        return max(self.pae[ii][jj], self.pae[jj][ii])

    def mean_pae_between(self, a: list[int], b: list[int]) -> float | None:
        """Mean PAE between two residue sets, over the worse direction per pair.

        High values mean the relative position of the two sets is unreliable,
        even when each set is individually well predicted. This is the single
        most misread signal in AlphaFold output.

        Note that a mean dilutes: a genuinely detached 20-residue domain
        averaged against 200 well-packed ones disappears. Prefer
        pae_segments() when deciding whether a model is usable.
        """
        if self.pae is None:
            return None
        index = self._pae_index()
        vals = [
            self._pair_pae(index[i], index[j])
            for i in a if i in index
            for j in b if j in index
        ]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def pae_profile(self, a: list[int], b: list[int]) -> dict[int, float]:
        """Mean PAE from residue set a to each residue of b, taken one at a time.

        Keeping the per-residue resolution is the whole point. Pooling first is
        what lets a genuinely detached domain vanish into a large model's
        average.
        """
        if self.pae is None:
            return {}
        index = self._pae_index()
        rows = [index[i] for i in sorted(set(a)) if i in index]
        if not rows:
            return {}
        return {
            j: sum(self._pair_pae(ii, index[j]) for ii in rows) / len(rows)
            for j in sorted(set(b)) if j in index
        }

    def pae_segments(
        self, a: list[int], b: list[int], cutoff: float, min_length: int = 3
    ) -> list[PaeSegment]:
        """Contiguous runs of b that sit above `cutoff` PAE from a, worst first.

        Segmenting by the PAE signal rather than by the residue selection is
        deliberate: contiguity in numbering is not domain membership, so
        splitting the selection into runs and averaging each one reintroduces
        the same dilution at a smaller scale. Grouping consecutive residues
        that individually exceed the cutoff puts the boundaries where the
        uncertainty actually changes, and lets the finding name the residues.

        `min_length` suppresses isolated residues, which are usually noise
        rather than a detached region.
        """
        profile = self.pae_profile(a, b)
        index = self._pae_index()
        n_site = len({i for i in a if i in index})
        segments: list[PaeSegment] = []
        for run in _contiguous([j for j, v in profile.items() if v > cutoff]):
            if len(run) < min_length:
                continue
            segments.append(
                PaeSegment(
                    start=run[0],
                    end=run[-1],
                    mean_pae=sum(profile[j] for j in run) / len(run),
                    n_pairs=len(run) * n_site,
                )
            )
        segments.sort(key=lambda s: s.mean_pae, reverse=True)
        return segments


def _contiguous(numbers: list[int]) -> list[list[int]]:
    """Split a sorted list of residue numbers into consecutive runs."""
    runs: list[list[int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    return runs


def parse_structure(path: str | Path) -> Structure:
    """Read pLDDT from the B-factor column of a PDB or mmCIF file."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"File not found: {path}")
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        # A path can exist and still be unreadable: a directory, a bad mode, a
        # dead symlink. That is a problem with the input, not with FoldGuard.
        raise ParseError(f"Could not read {path}: {exc.strerror}") from exc
    if path.suffix.lower() in {".cif", ".mmcif"}:
        residues = _parse_cif(text, str(path))
    else:
        residues = _parse_pdb(text, str(path))
    return Structure(residues=residues, source=str(path))


def _parse_pdb(text: str, source: str) -> list[Residue]:
    seen: dict[tuple[str, int], Residue] = {}
    insertion_codes: list[str] = []
    n_models = 0

    for line in text.splitlines():
        if line.startswith("MODEL "):
            n_models += 1
            if n_models > 1:
                # Ranked models concatenated, or an NMR ensemble. Keeping the
                # first silently discards the rest and reports on a model the
                # user may not have meant.
                raise ParseError(
                    f"{source} contains more than one model. FoldGuard analyses "
                    "a single model; keeping only the first would silently "
                    "discard the others. Split the file and pass the model you "
                    "want."
                )
            continue
        if not line.startswith("ATOM"):
            continue
        # Only take CA atoms: one confidence value per residue.
        if line[12:16].strip() != "CA":
            continue
        try:
            name = line[17:20].strip()
            chain = line[21].strip() or "A"
            number = int(line[22:26])
            plddt = float(line[60:66])
        except (ValueError, IndexError):
            continue
        icode = line[26:27].strip()
        if icode:
            insertion_codes.append(f"{chain}/{number}{icode}")
        key = (chain, number)
        if key not in seen:
            seen[key] = Residue(number=number, name=name, chain=chain, plddt=plddt)

    if insertion_codes:
        raise ParseError(
            f"{source} uses insertion codes (e.g. "
            f"{', '.join(insertion_codes[:3])}). Residue number alone is then "
            "not a unique identity - 100, 100A and 100B are three residues that "
            "would collapse into one - so FoldGuard refuses rather than "
            "silently dropping them. Renumber sequentially first."
        )
    return list(seen.values())


def _cif_tokens(line: str) -> list[str]:
    """Split one mmCIF data line, honouring quoted values.

    A plain str.split() breaks any value containing a space - "ALA X",
    'HIS A' - which shortens the row and makes the residue get skipped
    silently. In CIF a quote closes only when followed by whitespace or end of
    line, so an apostrophe inside a value does not terminate it.
    """
    tokens: list[str] = []
    i, n = 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line[i] in "'\"":
            quote = line[i]
            i += 1
            start = i
            while i < n and not (
                line[i] == quote and (i + 1 >= n or line[i + 1].isspace())
            ):
                i += 1
            tokens.append(line[start:i])
            i += 1
        else:
            start = i
            while i < n and not line[i].isspace():
                i += 1
            tokens.append(line[start:i])
    return tokens


def _cif_value(row: dict, *keys: str) -> str | None:
    """First present, non-placeholder value among keys. '.' and '?' mean absent."""
    for key in keys:
        value = row.get(key)
        if value is not None and value not in (".", "?"):
            return value
    return None


def _parse_cif(text: str, source: str) -> list[Residue]:
    """Minimal mmCIF atom_site reader. Handles AlphaFold DB output."""
    residues: dict[tuple[str, int], Residue] = {}
    headers: list[str] = []
    insertion_codes: list[str] = []
    models: set[str] = set()
    in_loop = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith("#") or not stripped:
            break
        parts = _cif_tokens(stripped)
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        # A calcium ion is label_atom_id 'CA', identical to an alpha carbon.
        # Without this the ion becomes a residue - and when AlphaFold3 writes a
        # ligand on its own chain, a legitimate monomer gets refused as a
        # multimer. The PDB reader filters on the ATOM record for the same
        # reason.
        if row.get("group_PDB", "ATOM") != "ATOM":
            continue
        if row.get("label_atom_id") != "CA":
            continue

        model = _cif_value(row, "pdbx_PDB_model_num")
        if model is not None:
            models.add(model)
            if len(models) > 1:
                raise ParseError(
                    f"{source} contains more than one model. FoldGuard analyses "
                    "a single model; keeping only the first would silently "
                    "discard the others. Split the file and pass the model you "
                    "want."
                )

        # Author numbering is what a user reads off a paper or a PDB entry and
        # types into --site. label_seq_id is an internal sequential index, and
        # on a model carrying UniProt numbering the two differ - selecting
        # 341-343 would then miss entirely.
        number_text = _cif_value(row, "auth_seq_id", "label_seq_id")
        chain = _cif_value(row, "auth_asym_id", "label_asym_id") or "A"
        try:
            number = int(number_text)
            plddt = float(row.get("B_iso_or_equiv", "0"))
        except (TypeError, ValueError):
            continue
        name = row.get("label_comp_id", "UNK")

        icode = _cif_value(row, "pdbx_PDB_ins_code")
        if icode:
            insertion_codes.append(f"{chain}/{number}{icode}")

        key = (chain, number)
        if key not in residues:
            residues[key] = Residue(number=number, name=name, chain=chain, plddt=plddt)

    if insertion_codes:
        raise ParseError(
            f"{source} uses insertion codes (e.g. "
            f"{', '.join(insertion_codes[:3])}). Residue number alone is then "
            "not a unique identity, so FoldGuard refuses rather than silently "
            "dropping residues. Renumber sequentially first."
        )
    return list(residues.values())


# AlphaFold caps PAE at 31.75 A; allow headroom for other predictors' conventions.
MAX_PLAUSIBLE_PAE = 100.0


def _validate_pae(matrix, n_residues: int, source: str) -> list[list[float]]:
    """Check a PAE matrix really describes this structure, before anything uses it.

    A PAE matrix from a different model is the most dangerous input this tool
    can receive: it is structurally valid, so every downstream check runs
    normally and reports numbers about the wrong protein. Validating here means
    a mismatch fails once, loudly, instead of silently degrading into a PASS.
    """
    if not isinstance(matrix, list) or not matrix:
        raise ParseError(f"PAE value in {source} is not a non-empty matrix.")

    n = len(matrix)
    for k, row in enumerate(matrix):
        if not isinstance(row, list):
            raise ParseError(f"PAE row {k} in {source} is not a list.")
        if len(row) != n:
            raise ParseError(
                f"PAE matrix in {source} is not square: it has {n} rows but "
                f"row {k} has {len(row)} columns."
            )
        try:
            # fsum rejects non-numeric entries at C speed and propagates NaN.
            total = math.fsum(row)
        except TypeError as exc:
            raise ParseError(f"PAE row {k} in {source} contains non-numeric values.") from exc
        if total != total:
            raise ParseError(f"PAE row {k} in {source} contains NaN.")
        if min(row) < 0.0 or max(row) > MAX_PLAUSIBLE_PAE:
            raise ParseError(
                f"PAE row {k} in {source} has values outside 0-{MAX_PLAUSIBLE_PAE:.0f} A "
                f"(min {min(row):.2f}, max {max(row):.2f}). This does not look like PAE."
            )

    if n != n_residues:
        raise ParseError(
            f"PAE matrix is {n}x{n} but the structure has {n_residues} residues. "
            "The matrix does not describe this model - check you have the PAE "
            "file matching this prediction (and note that multimer PAE covers "
            "all chains, while FoldGuard reads one chain)."
        )
    return matrix


def attach_pae(structure: Structure, path: str | Path) -> Structure:
    """Load a PAE matrix from AlphaFold JSON, validate it, and attach it."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"PAE file not found: {path}")
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ParseError(f"Could not read PAE file {path}: {exc.strerror}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"Could not parse PAE JSON: {exc}") from exc

    # AlphaFold DB uses a list containing one dict; ColabFold writes the dict directly.
    if isinstance(data, list):
        if not data:
            raise ParseError("PAE JSON is an empty list")
        data = data[0]
    if not isinstance(data, dict):
        raise ParseError(f"PAE JSON in {path} is not an object.")

    for key in ("predicted_aligned_error", "pae"):
        if key in data:
            structure.pae = _validate_pae(data[key], structure.n_residues, str(path))
            return structure

    if {"residue1", "residue2", "distance"} <= set(data):
        raise ParseError(
            "This is the legacy AlphaFold DB PAE format (residue1/residue2/"
            "distance), which FoldGuard does not yet read. Re-download the "
            "current format, which stores 'predicted_aligned_error' directly."
        )

    raise ParseError(
        "PAE JSON contained no 'predicted_aligned_error' or 'pae' key. "
        f"Keys present: {sorted(data)[:6]}"
    )


def parse_site(spec: str) -> list[int]:
    """Parse a residue selection like '45-52,88,120-124' into residue numbers."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError as exc:
                raise ParseError(f"Could not parse residue range '{chunk}'") from exc
            if hi_i < lo_i:
                raise ParseError(f"Range '{chunk}' runs backwards")
            # Check the span before expanding it: `--site 1-20000000` would
            # otherwise allocate 20 million integers before anything looked at
            # the model.
            if hi_i - lo_i + 1 > MAX_SITE_RESIDUES - len(out):
                raise ParseError(
                    f"Selection '{spec}' covers too many residues "
                    f"(limit {MAX_SITE_RESIDUES:,}). Range '{chunk}' spans "
                    f"{hi_i - lo_i + 1:,}. Check for a typo in the residue "
                    "numbers."
                )
            out.extend(range(lo_i, hi_i + 1))
        else:
            try:
                out.append(int(chunk))
            except ValueError as exc:
                raise ParseError(f"Could not parse residue '{chunk}'") from exc
            if len(out) > MAX_SITE_RESIDUES:
                raise ParseError(
                    f"Selection '{spec}' covers too many residues "
                    f"(limit {MAX_SITE_RESIDUES:,})."
                )
    if not out:
        raise ParseError(f"No residues parsed from '{spec}'")
    return sorted(set(out))
