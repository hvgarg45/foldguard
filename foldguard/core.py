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


@dataclass
class Structure:
    """A parsed predicted structure."""

    residues: list[Residue]
    source: str
    pae: list[list[float]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.residues:
            raise ParseError(f"No residues with confidence values found in {self.source}")

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

    def low_confidence_regions(self, min_length: int = 5) -> list[Region]:
        """Runs of >=min_length residues below the 'confident' bar."""
        return self._runs(lambda r: r.plddt < CONFIDENT, min_length)

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

    def mean_pae_between(self, a: list[int], b: list[int]) -> float | None:
        """Mean predicted aligned error between two residue sets.

        High values mean the relative position of the two sets is unreliable,
        even when each set is individually well predicted. This is the single
        most misread signal in AlphaFold output.
        """
        if self.pae is None:
            return None
        index = {r.number: i for i, r in enumerate(sorted(self.residues, key=lambda x: x.number))}
        vals = []
        for i in a:
            for j in b:
                ii, jj = index.get(i), index.get(j)
                if ii is None or jj is None:
                    continue
                if ii < len(self.pae) and jj < len(self.pae[ii]):
                    vals.append(self.pae[ii][jj])
        if not vals:
            return None
        return sum(vals) / len(vals)


def parse_structure(path: str | Path) -> Structure:
    """Read pLDDT from the B-factor column of a PDB or mmCIF file."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"File not found: {path}")
    text = path.read_text(errors="replace")
    if path.suffix.lower() in {".cif", ".mmcif"}:
        residues = _parse_cif(text, str(path))
    else:
        residues = _parse_pdb(text, str(path))
    return Structure(residues=residues, source=str(path))


def _parse_pdb(text: str, source: str) -> list[Residue]:
    seen: dict[tuple[str, int], Residue] = {}
    for line in text.splitlines():
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
        key = (chain, number)
        if key not in seen:
            seen[key] = Residue(number=number, name=name, chain=chain, plddt=plddt)
    return list(seen.values())


def _parse_cif(text: str, source: str) -> list[Residue]:
    """Minimal mmCIF atom_site reader. Handles AlphaFold DB output."""
    lines = text.splitlines()
    residues: dict[tuple[str, int], Residue] = {}
    headers: list[str] = []
    in_loop = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("_atom_site."):
            headers.append(stripped.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if stripped.startswith("#") or not stripped:
                break
            parts = stripped.split()
            if len(parts) < len(headers):
                continue
            row = dict(zip(headers, parts))
            if row.get("label_atom_id") != "CA":
                continue
            try:
                number = int(row.get("label_seq_id", "0"))
                plddt = float(row.get("B_iso_or_equiv", "0"))
                chain = row.get("label_asym_id", "A")
                name = row.get("label_comp_id", "UNK")
            except ValueError:
                continue
            key = (chain, number)
            if key not in residues:
                residues[key] = Residue(number=number, name=name, chain=chain, plddt=plddt)
    return list(residues.values())


def attach_pae(structure: Structure, path: str | Path) -> Structure:
    """Load a PAE matrix from AlphaFold JSON and attach it."""
    path = Path(path)
    if not path.exists():
        raise ParseError(f"PAE file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ParseError(f"Could not parse PAE JSON: {exc}") from exc

    # AlphaFold DB uses a list containing one dict; ColabFold writes the dict directly.
    if isinstance(data, list):
        if not data:
            raise ParseError("PAE JSON is an empty list")
        data = data[0]

    for key in ("predicted_aligned_error", "pae"):
        if key in data:
            structure.pae = data[key]
            return structure

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
            out.extend(range(lo_i, hi_i + 1))
        else:
            try:
                out.append(int(chunk))
            except ValueError as exc:
                raise ParseError(f"Could not parse residue '{chunk}'") from exc
    if not out:
        raise ParseError(f"No residues parsed from '{spec}'")
    return sorted(set(out))
