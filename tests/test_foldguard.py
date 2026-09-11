"""Tests for FoldGuard.

The important ones are the behavioural tests at the bottom: they encode the
failure mode this tool exists to catch.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foldguard.core import (  # noqa: E402
    ParseError,
    Residue,
    Structure,
    attach_pae,
    parse_site,
    parse_structure,
)
from foldguard.verdict import Level, assess  # noqa: E402


# ---------- helpers ----------

def make_pdb(tmp_path: Path, plddts: list[float]) -> Path:
    lines = []
    for i, p in enumerate(plddts, start=1):
        lines.append(
            f"ATOM  {i*5:5d}  CA  ALA A{i:4d}    "
            f"{i*1.2:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{p:6.2f}           C"
        )
    lines.append("END")
    path = tmp_path / "m.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------- parsing ----------

def test_parses_plddt_from_bfactor(tmp_path):
    s = parse_structure(make_pdb(tmp_path, [95.0, 80.0, 45.0]))
    assert s.n_residues == 3
    assert s.residues[0].plddt == pytest.approx(95.0)
    assert s.residues[2].plddt == pytest.approx(45.0)


def test_only_one_value_per_residue(tmp_path):
    """Non-CA atoms must not create duplicate residues."""
    path = tmp_path / "multi.pdb"
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 90.00           N\n"
        "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 90.00           C\n"
        "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 90.00           C\n"
        "END\n"
    )
    assert parse_structure(path).n_residues == 1


def test_missing_file_raises():
    with pytest.raises(ParseError):
        parse_structure("/no/such/file.pdb")


def test_empty_structure_raises(tmp_path):
    path = tmp_path / "empty.pdb"
    path.write_text("HEADER nothing\nEND\n")
    with pytest.raises(ParseError):
        parse_structure(path)


# ---------- bands ----------

@pytest.mark.parametrize(
    "plddt,expected",
    [(95.0, "very_high"), (90.0, "very_high"), (75.0, "confident"),
     (70.0, "confident"), (60.0, "low"), (50.0, "low"), (30.0, "very_low")],
)
def test_confidence_bands(plddt, expected):
    assert Residue(1, "ALA", "A", plddt).band == expected


def test_disordered_region_detection():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    for i in range(5, 13):  # 8-residue disordered stretch
        res[i - 1] = Residue(i, "ALA", "A", 30.0)
    regions = Structure(res, "t").disordered_regions(min_length=5)
    assert len(regions) == 1
    assert regions[0].start == 5 and regions[0].end == 12


def test_short_dips_are_not_reported():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    res[9] = Residue(10, "ALA", "A", 30.0)  # single weak residue
    assert Structure(res, "t").disordered_regions(min_length=5) == []


# ---------- site parsing ----------

def test_parse_site_ranges_and_singles():
    assert parse_site("1-3,7,10-11") == [1, 2, 3, 7, 10, 11]


def test_parse_site_deduplicates():
    assert parse_site("5,5,4-6") == [4, 5, 6]


def test_parse_site_rejects_backwards_range():
    with pytest.raises(ParseError):
        parse_site("10-2")


def test_parse_site_rejects_garbage():
    with pytest.raises(ParseError):
        parse_site("abc")


# ---------- PAE ----------

def test_attach_pae_accepts_both_layouts(tmp_path):
    res = [Residue(i, "ALA", "A", 90.0) for i in range(1, 4)]
    mat = [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]

    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps([{"predicted_aligned_error": mat}]))
    assert attach_pae(Structure(res, "t"), p1).has_pae()

    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"pae": mat}))
    assert attach_pae(Structure(res, "t"), p2).has_pae()


def test_attach_pae_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"something_else": []}))
    with pytest.raises(ParseError):
        attach_pae(Structure([Residue(1, "ALA", "A", 90.0)], "t"), p)


# ---------- the behaviour this tool exists for ----------

def test_confident_model_with_weak_pocket_fails_docking():
    """The core case. Global confidence is high, the site is not.

    Without a site check this model looks fine. That is the mistake.
    """
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 101)]
    for i in range(40, 51):  # pocket sits in a floppy loop
        res[i - 1] = Residue(i, "ALA", "A", 45.0)
    s = Structure(res, "t")

    assert s.fraction_trustworthy() > 0.85  # looks good globally
    assert assess(s, task="fold").verdict is not Level.FAIL
    assert assess(s, task="docking", site=list(range(40, 51))).verdict is Level.FAIL


def test_same_model_passes_when_site_is_well_resolved():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 101)]
    for i in range(40, 51):
        res[i - 1] = Residue(i, "ALA", "A", 45.0)
    report = assess(Structure(res, "t"), task="docking", site=list(range(60, 71)))
    assert report.verdict is not Level.FAIL


def test_mutation_task_is_stricter_than_docking():
    res = [Residue(i, "ALA", "A", 85.0) for i in range(1, 101)]
    site = [50]
    assert assess(Structure(res, "t"), task="docking", site=site).verdict is not Level.FAIL
    assert assess(Structure(res, "t"), task="mutation", site=site).verdict is Level.FAIL


def test_missing_site_produces_a_warning():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    report = assess(Structure(res, "t"), task="docking")
    assert any("No site specified" in f.title for f in report.findings)


def test_site_numbering_mismatch_is_caught():
    """Predicted models are often renumbered. Silent misses are dangerous."""
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    report = assess(Structure(res, "t"), task="docking", site=[900, 901, 902])
    assert report.verdict is Level.FAIL
    assert any("not found" in f.title.lower() for f in report.findings)


def test_high_interdomain_pae_is_flagged():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    mat = [[12.0] * 20 for _ in range(20)]
    s = Structure(res, "t", pae=mat)
    report = assess(s, task="docking", site=[1, 2, 3])
    assert any("relative to the rest" in f.title for f in report.findings)


def test_exit_codes_map_to_verdicts():
    good = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    bad = [Residue(i, "ALA", "A", 20.0) for i in range(1, 51)]
    clean_pae = [[0.5] * 50 for _ in range(50)]

    # A clean pass requires a site AND PAE: without PAE we cannot rule out
    # a confidently wrong domain arrangement, so WARN is the honest ceiling.
    passing = Structure(good, "t", pae=clean_pae)
    assert assess(passing, "docking", site=[10, 11]).exit_code == 0

    assert assess(Structure(good, "t"), "docking").exit_code == 1
    assert assess(Structure(bad, "t"), "docking", site=[10, 11]).exit_code == 2


def test_absent_pae_caps_verdict_at_warn():
    """No PAE means relative placement is unverified. Never a silent pass."""
    good = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    report = assess(Structure(good, "t"), "docking", site=[10, 11])
    assert report.verdict is Level.WARN
    assert any("No PAE" in f.title for f in report.findings)


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        assess(Structure([Residue(1, "ALA", "A", 90.0)], "t"), task="nonsense")


# ---------- CLI ----------

def test_cli_runs_and_returns_expected_code(tmp_path):
    path = make_pdb(tmp_path, [95.0] * 40 + [30.0] * 10)
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "foldguard.cli", str(path),
         "--task", "docking", "--site", "41-50", "--json"],
        capture_output=True, text=True, cwd=root,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "FAIL"
    assert payload["n_residues"] == 50
