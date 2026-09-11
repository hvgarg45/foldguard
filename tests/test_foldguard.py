"""Tests for FoldGuard.

The important ones are the behavioural tests at the bottom: they encode the
failure mode this tool exists to catch.
"""

import json
import os
import subprocess
import time
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
    flagged = [f for f in report.findings if "Site position relative" in f.title]
    assert flagged and flagged[0].level is Level.WARN


# ---------- PAE: the checks that must never silently skip ----------

def test_detached_domain_is_not_averaged_away():
    """The failure this tool exists to catch, at the scale it actually occurs.

    A 20-residue domain floating at 28 A, against 200 well-packed residues at
    1 A. The pooled mean is 3.6 A and sails under any sensible cutoff; only
    per-segment scoring keeps the floating domain visible.
    """
    n = 220
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i >= 200) != (j >= 200):
                mat[i][j] = 28.0
    s = Structure(res, "t", pae=mat)
    site = list(range(95, 105))

    assert s.mean_pae_between(site, [x for x in range(1, n + 1) if x not in site]) < 5.0
    report = assess(s, task="docking", site=site)
    assert report.verdict is Level.WARN
    flagged = [f for f in report.findings if "Site position relative" in f.title]
    assert flagged and "201-220" in flagged[0].detail


def test_pae_segments_isolates_the_floating_run():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 31)]
    mat = [[1.0] * 30 for _ in range(30)]
    for i in range(30):
        for j in range(30):
            if (i >= 25) != (j >= 25):
                mat[i][j] = 24.0
    segs = Structure(res, "t", pae=mat).pae_segments([1, 2, 3], list(range(4, 31)), cutoff=5.0)
    assert segs[0].start == 26 and segs[0].end == 30
    assert segs[0].mean_pae == pytest.approx(24.0)


def test_pae_uses_the_worse_direction():
    """PAE is asymmetric; reading one direction only hides half the failures."""
    n = 30                                   # core must exceed MIN_PAE_CORE
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[0.5] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i >= 3 and j < 3:
                mat[i][j] = 29.0  # core->site catastrophic, site->core clean
    report = assess(Structure(res, "t", pae=mat), task="docking", site=[1, 2, 3])
    assert report.verdict is Level.WARN
    assert any("Site position relative" in f.title for f in report.findings)


def test_pae_without_very_high_core_still_runs():
    """No residue reaches 90. The check must fall back, not vanish."""
    res = [Residue(i, "ALA", "A", 85.0) for i in range(1, 51)]
    worst = [[31.75] * 50 for _ in range(50)]
    report = assess(Structure(res, "t", pae=worst), task="md", site=[10, 11, 12])
    assert report.verdict is not Level.PASS
    assert any("relative" in f.title or "no confident core" in f.title
               for f in report.findings)


def test_pae_with_no_usable_core_warns_rather_than_passing():
    res = [Residue(i, "ALA", "A", 40.0) for i in range(1, 21)]
    report = assess(Structure(res, "t", pae=[[0.5] * 20] * 20), task="fold", site=[1, 2])
    assert any("no confident core" in f.title for f in report.findings)


def test_mismatched_pae_matrix_is_rejected(tmp_path):
    """A PAE from another prediction is valid JSON and totally wrong."""
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps({"pae": [[0.1] * 3 for _ in range(3)]}))
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 51)], "t")
    with pytest.raises(ParseError, match="3x3.*50 residues"):
        attach_pae(s, p)


@pytest.mark.parametrize("payload", [
    {"pae": 5.0},
    {"pae": None},
    {"pae": [["a", "b"], ["c", "d"]]},
    {"pae": [[0.0, 1.0], [1.0]]},
    {"pae": [[0.0, -3.0], [-3.0, 0.0]]},
    {"pae": [[0.0, float("nan")], [float("nan"), 0.0]]},
])
def test_malformed_pae_payloads_are_rejected(tmp_path, payload):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 3)], "t")
    with pytest.raises(ParseError):
        attach_pae(s, p)


def test_pae_is_validated_on_the_constructor_path_too():
    """attach_pae is not the only way a matrix gets in."""
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    with pytest.raises(ParseError, match="3x3.*50 residues"):
        Structure(res, "t", pae=[[0.1] * 3 for _ in range(3)])


def test_legacy_afdb_pae_format_gets_a_named_error(tmp_path):
    p = tmp_path / "v1.json"
    p.write_text(json.dumps([{"residue1": [1], "residue2": [1], "distance": [0.5]}]))
    s = Structure([Residue(1, "ALA", "A", 95.0)], "t")
    with pytest.raises(ParseError, match="legacy AlphaFold DB"):
        attach_pae(s, p)


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


# ---------- the tool's own failures must never look like a verdict ----------

def test_internal_error_exits_outside_the_verdict_range(tmp_path, monkeypatch):
    """A crash must not be reportable as 'WARN, usable with caveats'.

    Exit codes 0/1/2 are claims about the model. If an unexpected exception
    escapes, Python exits 1, which this tool defines as WARN - so a broken
    FoldGuard reads to a pipeline as a soft pass. This tests the invariant
    rather than any single trigger, so it still holds for exception paths
    nobody has found yet.
    """
    import foldguard.cli as cli

    def boom(*args, **kwargs):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(cli, "assess", boom)
    path = make_pdb(tmp_path, [95.0] * 10)
    assert cli.main([str(path), "--task", "fold"]) == 4


def test_directory_as_structure_is_bad_input_not_a_crash(tmp_path):
    import foldguard.cli as cli
    d = tmp_path / "somedir"
    d.mkdir()
    assert cli.main([str(d)]) == 3


def test_directory_as_pae_is_bad_input_not_a_crash(tmp_path):
    import foldguard.cli as cli
    d = tmp_path / "paedir"
    d.mkdir()
    path = make_pdb(tmp_path, [95.0] * 10)
    assert cli.main([str(path), "--pae", str(d)]) == 3


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_structure_is_bad_input_not_a_crash(tmp_path):
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 10)
    path.chmod(0o000)
    try:
        assert cli.main([str(path)]) == 3
    finally:
        path.chmod(0o644)


def test_broken_pipe_does_not_crash(tmp_path):
    """`foldguard --json | head` must not traceback into the WARN exit code.

    The read end is closed before the child starts, so every write fails with
    EPIPE - deterministic, unlike racing a real `head`.
    """
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 40)
    root = Path(__file__).resolve().parents[1]

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    proc = subprocess.Popen(
        [sys.executable, "-m", "foldguard.cli", str(path), "--json"],
        stdout=write_fd, stderr=subprocess.PIPE, cwd=root, text=True,
    )
    os.close(write_fd)
    _, err = proc.communicate()

    assert "Traceback" not in err, err
    assert proc.returncode == cli.EXIT_INTERNAL, f"got {proc.returncode}: {err}"


# ---------- multimers: refuse rather than silently mis-analyse ----------

def make_multichain_pdb(tmp_path: Path, chains: dict) -> Path:
    lines, serial = [], 0
    for chain, plddts in chains.items():
        for i, p in enumerate(plddts, start=1):
            serial += 1
            lines.append(
                f"ATOM  {serial*5:5d}  CA  ALA {chain}{i:4d}    "
                f"{i*1.2:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{p:6.2f}           C"
            )
    lines.append("END")
    path = tmp_path / "multi.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_structure_rejects_more_than_one_chain():
    """Residue number alone is not an identity in a multimer.

    by_number() would drop one chain's residues entirely, _runs() would
    interleave the chains so disorder detection finds nothing, and the PAE
    index would be meaningless. Every downstream number is wrong, not merely
    uncertain, so this is refused rather than warned about.
    """
    res = ([Residue(i, "ALA", "A", 95.0) for i in range(1, 31)]
           + [Residue(i, "ALA", "B", 20.0) for i in range(1, 31)])
    with pytest.raises(ParseError, match="more than one chain"):
        Structure(res, "t")


def test_parse_structure_rejects_a_dimer(tmp_path):
    path = make_multichain_pdb(tmp_path, {"A": [95.0] * 30, "B": [20.0] * 30})
    with pytest.raises(ParseError, match="more than one chain"):
        parse_structure(path)


def test_cli_on_a_dimer_is_bad_input_and_names_the_chains(tmp_path, capsys):
    import foldguard.cli as cli
    path = make_multichain_pdb(tmp_path, {"A": [95.0] * 30, "B": [20.0] * 30})
    assert cli.main([str(path), "--task", "docking", "--site", "10-12"]) == 3
    err = capsys.readouterr().err
    assert "(A, B)" in err, err


def test_single_chain_models_are_unaffected(tmp_path):
    path = make_multichain_pdb(tmp_path, {"A": [95.0] * 20})
    assert parse_structure(path).n_residues == 20


# ---------- input trust: values must actually look like pLDDT ----------

def test_all_zero_plddt_is_refused_as_a_column_shift(tmp_path):
    """The signature of a misaligned B-factor column.

    Reading fixed columns [60:66] on a line three characters short yields
    ".00" -> 0.0 for every residue. Nothing about the parse fails, so the
    tool would confidently FAIL a model it never actually read.
    """
    res = [Residue(i, "ALA", "A", 0.0) for i in range(1, 21)]
    with pytest.raises(ParseError, match="0.00|misalign"):
        Structure(res, "t")


def test_zero_to_one_scale_is_refused_with_a_diagnosis(tmp_path):
    """Some ColabFold/ESMFold writers emit pLDDT on a 0-1 scale."""
    res = [Residue(i, "ALA", "A", 0.95) for i in range(1, 21)]
    with pytest.raises(ParseError, match="0-1"):
        Structure(res, "t")


@pytest.mark.parametrize("bad", [101.0, 999.0, -0.1, -5.0])
def test_out_of_range_plddt_is_refused(bad):
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    res[5] = Residue(6, "ALA", "A", bad)
    with pytest.raises(ParseError, match="outside"):
        Structure(res, "t")


def test_column_shifted_pdb_is_refused_end_to_end(tmp_path):
    shifted = "\n".join(
        "ATOM      5  CA  ALA A" + f"{i:4d} "
        + f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{95.00:6.2f}"
        for i in range(1, 21)
    )
    path = tmp_path / "shift.pdb"
    path.write_text(shifted + "\nEND\n")
    with pytest.raises(ParseError):
        parse_structure(path)


def test_plausible_models_are_unaffected():
    Structure([Residue(i, "ALA", "A", p) for i, p in
               enumerate([95.0, 70.0, 30.0, 0.0, 100.0], start=1)], "t")


def test_insertion_codes_are_refused(tmp_path):
    """Antibody/Kabat numbering: 100, 100A, 100B are three different residues.

    With the iCode ignored they collapse to one and two are silently dropped,
    so residue number stops being an identity - the same failure as a multimer.
    """
    path = make_multichain_pdb(tmp_path, {"A": [95.0] * 3})
    lines = path.read_text().splitlines()
    # give all three residues number 100, distinguished only by iCode
    patched = []
    for line, icode in zip(lines[:3], (" ", "A", "B")):
        patched.append(line[:22] + f"{100:4d}" + icode + line[27:])
    path.write_text("\n".join(patched) + "\nEND\n")
    with pytest.raises(ParseError, match="insertion code"):
        parse_structure(path)


def test_multi_model_files_are_refused(tmp_path):
    """Ranked models concatenated into one file, or an NMR ensemble."""
    body = make_multichain_pdb(tmp_path, {"A": [95.0] * 5}).read_text().splitlines()[:5]
    path = tmp_path / "mm.pdb"
    path.write_text(
        "MODEL        1\n" + "\n".join(body) + "\nENDMDL\n"
        "MODEL        2\n" + "\n".join(body) + "\nENDMDL\nEND\n"
    )
    with pytest.raises(ParseError, match="more than one model"):
        parse_structure(path)


def test_a_single_model_record_is_fine(tmp_path):
    body = make_multichain_pdb(tmp_path, {"A": [95.0] * 5}).read_text().splitlines()[:5]
    path = tmp_path / "one.pdb"
    path.write_text("MODEL        1\n" + "\n".join(body) + "\nENDMDL\nEND\n")
    assert parse_structure(path).n_residues == 5


# ---------- mmCIF input trust ----------

def make_cif(tmp_path: Path, headers: list, rows: list, name="m.cif") -> Path:
    lines = ["data_test", "loop_"] + [f"_atom_site.{h}" for h in headers]
    lines += [" ".join(str(v) for v in r) for r in rows] + ["#"]
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


CIF_HEADERS = ["group_PDB", "id", "label_atom_id", "label_comp_id",
               "label_asym_id", "label_seq_id", "auth_asym_id", "auth_seq_id",
               "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv"]


def test_cif_uses_author_numbering(tmp_path):
    """--site is typed from a paper or a PDB entry, i.e. author numbering.

    An AFDB model carrying UniProt numbering 341-343 must be selectable as
    341-343, not silently renumbered to 1-3.
    """
    rows = [["ATOM", i, "CA", "ALA", "A", i, "A", 340 + i, 0.0, 0.0, 0.0, 1.0, 95.0]
            for i in (1, 2, 3)]
    s = parse_structure(make_cif(tmp_path, CIF_HEADERS, rows))
    assert sorted(r.number for r in s.residues) == [341, 342, 343]


def test_cif_falls_back_to_label_seq_id(tmp_path):
    headers = [h for h in CIF_HEADERS if not h.startswith("auth_")]
    rows = [["ATOM", i, "CA", "ALA", "A", i, 0.0, 0.0, 0.0, 1.0, 95.0]
            for i in (1, 2, 3)]
    s = parse_structure(make_cif(tmp_path, headers, rows))
    assert sorted(r.number for r in s.residues) == [1, 2, 3]


def test_cif_quoted_values_containing_spaces_are_not_dropped(tmp_path):
    rows = [["ATOM", i, "CA", "'ALA X'", "A", i, "A", 340 + i, 0.0, 0.0, 0.0, 1.0, 95.0]
            for i in (1, 2, 3)]
    assert parse_structure(make_cif(tmp_path, CIF_HEADERS, rows)).n_residues == 3


def test_cif_insertion_codes_are_refused(tmp_path):
    headers = CIF_HEADERS + ["pdbx_PDB_ins_code"]
    rows = [["ATOM", 1, "CA", "ALA", "A", 1, "A", 100, 0.0, 0.0, 0.0, 1.0, 95.0, "."],
            ["ATOM", 2, "CA", "GLY", "A", 2, "A", 100, 0.0, 0.0, 0.0, 1.0, 95.0, "A"]]
    with pytest.raises(ParseError, match="insertion code"):
        parse_structure(make_cif(tmp_path, headers, rows))


# ---------- disorder must be judged relative to the site and the task ----------

def _disorder_findings(report):
    return [f for f in report.findings if "isorder" in f.title]


def _model(disordered_range, n=100, pae=True):
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    for i in disordered_range:
        res[i - 1] = Residue(i, "ALA", "A", 25.0)
    mat = [[0.4] * n for _ in range(n)] if pae else None
    return Structure(res, "t", pae=mat)


def test_floppy_tail_far_from_the_site_does_not_fail_a_fold_task():
    """A 35% disordered C-terminal tail is real biology, not a defect.

    The site is perfect and the PAE is clean; nothing about describing this
    fold is compromised by a floppy tail 50 residues away.
    """
    report = assess(_model(range(66, 101)), task="fold", site=[10, 11, 12])
    assert report.verdict is not Level.FAIL
    assert all(f.level is not Level.FAIL for f in _disorder_findings(report))


def test_disorder_away_from_the_site_is_never_itself_a_failure():
    """Docking here fails on global confidence - but not because of disorder."""
    report = assess(_model(range(66, 101)), task="docking", site=[10, 11, 12])
    assert all(f.level is not Level.FAIL for f in _disorder_findings(report))


def test_disorder_inside_the_site_fails_docking():
    site = list(range(40, 60))          # 20 residues, 5 of them disordered
    report = assess(_model(range(40, 45)), task="docking", site=site)
    inside = [f for f in _disorder_findings(report) if "site" in f.title.lower()]
    assert inside and inside[0].level is Level.FAIL


def test_the_same_disordered_site_only_warns_for_fold():
    site = list(range(40, 60))
    report = assess(_model(range(40, 45)), task="fold", site=site)
    inside = [f for f in _disorder_findings(report) if "site" in f.title.lower()]
    assert inside and inside[0].level is Level.WARN


def test_minor_disorder_elsewhere_does_not_block_a_pass():
    report = assess(_model(range(80, 90)), task="docking", site=[10, 11, 12])
    assert report.verdict is Level.PASS


# ---------- a selection must be bounded ----------

def test_absurd_site_range_is_refused_without_allocating():
    """`--site 1-20000000` is a typo, not a request.

    Expanding it builds 20 million integers and ~1.6 GB before anything looks
    at the model. The timing assertion is the real check: it fails if the
    range is expanded before being rejected.
    """
    start = time.perf_counter()
    with pytest.raises(ParseError, match="too many residues"):
        parse_site("1-20000000")
    assert time.perf_counter() - start < 0.5


def test_site_size_cap_is_cumulative_across_chunks():
    with pytest.raises(ParseError, match="too many residues"):
        parse_site("1-60000,200000-260000")


def test_ordinary_large_selections_still_work():
    assert len(parse_site("1-1000")) == 1000
    assert parse_site("45-52,88,120-124")[:3] == [45, 46, 47]


# ---------- output formatting ----------

def test_wrap_never_emits_a_blank_line():
    """A token longer than the width flushed an empty accumulator first."""
    from foldguard.cli import _wrap
    lines = _wrap("x" * 80 + " tail", 58)
    assert all(line.strip() for line in lines), lines
    assert lines[0].startswith("x")


# ---------- CLI flags that had no coverage ----------
# Characterisation tests: these pin behaviour that already worked but was
# untested, so they pass on first run rather than red-green.

def test_strict_promotes_warn_to_a_failing_exit(tmp_path):
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 50)
    assert cli.main([str(path), "--task", "fold"]) == 1          # WARN: no site, no PAE
    assert cli.main([str(path), "--task", "fold", "--strict"]) == 2


def test_strict_leaves_pass_and_fail_alone(tmp_path):
    import foldguard.cli as cli
    good = make_pdb(tmp_path, [95.0] * 50)
    assert cli.main([str(good), "--task", "fold", "--site", "900", "--strict"]) == 2


@pytest.mark.parametrize("cutoff,expect_warn", [(5.0, True), (20.0, False)])
def test_pae_cutoff_moves_the_threshold(cutoff, expect_warn):
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    s = Structure(res, "t", pae=[[12.0] * 20 for _ in range(20)])
    report = assess(s, task="docking", site=[1, 2, 3], pae_cutoff=cutoff)
    flagged = any("Site position relative" in f.title for f in report.findings)
    assert flagged is expect_warn


# ---------- found by testing against real AlphaFold DB models ----------

def _partly_disordered(n=100, good=range(1, 41), pae_clean=True):
    """A p53-shaped model: a well-resolved domain plus a large disordered rest."""
    res = [Residue(i, "ALA", "A", 95.0 if i in good else 40.0) for i in range(1, n + 1)]
    mat = [[0.4] * n for _ in range(n)] if pae_clean else None
    return Structure(res, "t", pae=mat)


def _global_finding(report):
    return next(f for f in report.findings if "Global confidence" in f.title)


def test_strong_site_demotes_a_weak_global_score_to_a_warning():
    """Real case: p53 R273 has site pLDDT 98.6 and failed on global confidence.

    40% of p53 is disordered 200 residues from the mutation. That is context
    for the interpretation, not grounds to refuse it.
    """
    report = assess(_partly_disordered(), task="docking", site=[10, 11, 12])
    assert _global_finding(report).level is Level.WARN


def test_a_weak_site_keeps_the_global_failure():
    report = assess(_partly_disordered(), task="docking", site=[50, 51, 52])
    assert _global_finding(report).level is Level.FAIL


def test_without_a_site_global_confidence_still_fails():
    report = assess(_partly_disordered(), task="docking")
    assert _global_finding(report).level is Level.FAIL


def test_pae_check_warns_when_the_core_is_too_small_to_judge():
    """Real case: lysozyme site 19-147 left one very-high core residue.

    The check reported PASS while its own detail said the pooled mean was
    8.6 A against a 5.0 A cutoff.
    """
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    s = Structure(res, "t", pae=[[0.4] * 20 for _ in range(20)])
    report = assess(s, task="docking", site=list(range(1, 16)))
    pae = [f for f in report.findings if "placement" in f.title or "relative" in f.title]
    assert pae and pae[0].level is Level.WARN
    assert "outside the site" in pae[0].detail


def test_pae_finding_can_never_contradict_its_own_pooled_mean():
    """One 60 A outlier lifts the pooled mean over the cutoff.

    No run of 3 survives the noise filter, so the old code reported PASS with
    a pooled mean printed above the cutoff in the same sentence.
    """
    n = 15                                   # core = residues 4-15, twelve of them
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[0.5] * n for _ in range(n)]
    for i in range(n):                       # one isolated 95 A residue (number 10)
        mat[i][9] = mat[9][i] = 95.0
    s = Structure(res, "t", pae=mat)
    report = assess(s, task="docking", site=[1, 2, 3])
    pae = [f for f in report.findings if "placement" in f.title or "relative" in f.title]
    assert pae and pae[0].level is Level.WARN, pae[0].detail if pae else "no finding"


def test_a_genuinely_clean_model_still_passes():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    s = Structure(res, "t", pae=[[0.5] * 50 for _ in range(50)])
    assert assess(s, task="docking", site=[10, 11, 12]).verdict is Level.PASS


# ---------- the PAE noise filters are arguments, not facts ----------

def _two_residue_blip(n=40, high=(20, 21), value=30.0):
    """A 2-residue run above the cutoff, too small to lift the pooled mean."""
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[0.5] * n for _ in range(n)]
    for j in high:
        for i in range(n):
            mat[i][j - 1] = mat[j - 1][i] = value
    return Structure(res, "t", pae=mat)


def test_default_min_run_suppresses_a_two_residue_blip():
    assert assess(_two_residue_blip(), task="docking", site=[1, 2, 3]).verdict is Level.PASS


def test_lowering_min_run_surfaces_it():
    report = assess(_two_residue_blip(), task="docking", site=[1, 2, 3], pae_min_run=2)
    assert report.verdict is Level.WARN
    assert any("relative" in f.title for f in report.findings)


def test_default_min_core_refuses_to_judge_a_small_core():
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 21)], "t",
                  pae=[[0.4] * 20 for _ in range(20)])
    report = assess(s, task="docking", site=list(range(1, 16)))
    assert any("too much of the model" in f.title for f in report.findings)


def test_lowering_min_core_lets_the_check_run():
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 21)], "t",
                  pae=[[0.4] * 20 for _ in range(20)])
    report = assess(s, task="docking", site=list(range(1, 16)), pae_min_core=3)
    assert report.verdict is Level.PASS


def test_cli_threads_both_flags_through(tmp_path):
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 20)
    pae = tmp_path / "p.json"
    pae.write_text(json.dumps({"pae": [[0.4] * 20 for _ in range(20)]}))
    args = [str(path), "--task", "docking", "--site", "1-15", "--pae", str(pae)]
    assert cli.main(args) == 1                              # core of 5 < default 10
    assert cli.main(args + ["--pae-min-core", "3"]) == 0     # now judgeable, and clean


def test_json_records_the_thresholds_used(tmp_path, capsys):
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 20)
    cli.main([str(path), "--task", "docking", "--site", "1-5", "--json",
              "--pae-cutoff", "7.5", "--pae-min-run", "2", "--pae-min-core", "4"])
    payload = json.loads(capsys.readouterr().out)
    t = payload["thresholds"]
    assert t["pae_cutoff"] == 7.5
    assert t["pae_min_run"] == 2
    assert t["pae_min_core"] == 4
    assert t["site_mean"] == 80.0 and t["global_trustworthy"] == 0.70


@pytest.mark.parametrize("flag,value", [("--pae-min-run", "0"), ("--pae-min-run", "-1"),
                                        ("--pae-min-core", "0"), ("--pae-cutoff", "-2"),
                                        ("--pae-cutoff", "nan"), ("--pae-cutoff", "inf"),
                                        ("--pae-cutoff", "-inf")])
def test_nonsense_threshold_values_are_rejected(tmp_path, flag, value):
    """Rejected either by our own check or by argparse; both must give exit 3.

    "-inf" parses as an option rather than a value, so argparse intercepts it
    before our validation runs and raises SystemExit(3) via the _Parser
    subclass instead of returning.
    """
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 20)
    try:
        assert cli.main([str(path), "--site", "1-5", flag, value]) == 3
    except SystemExit as exc:
        assert exc.code == 3


@pytest.mark.parametrize("bad", [["--task", "nonsense"],
                                 ["--pae-cutoff", "notanumber"],
                                 ["--nosuchflag"]])
def test_usage_errors_are_bad_input_not_a_failed_verdict(tmp_path, bad):
    """argparse exits 2 by default, which this tool defines as FAIL.

    A mistyped flag would report the model as inadequate for the task.
    """
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 20)
    with pytest.raises(SystemExit) as exc:
        cli.main([str(path)] + bad)
    assert exc.value.code == 3


# ---------- found by code review ----------

def test_non_finite_cutoff_cannot_disable_the_pae_check():
    """NaN defeats every `>` comparison, silently neutering both guards."""
    n = 40
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[30.0] * n for _ in range(n)]
    s = Structure(res, "t", pae=mat)
    with pytest.raises(ValueError, match="finite"):
        assess(s, task="docking", site=[1, 2, 3], pae_cutoff=float("nan"))


def test_scattered_pae_excess_is_not_reported_as_consistent():
    """Half the core over the cutoff, but non-contiguous and the mean stays under.

    The run filter needs contiguity and the pooled guard needs the average to
    cross; uncertainty that is neither slipped past both and returned PASS
    while printing a worst residue above the cutoff.
    """
    n = 300
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[0.5] * n for _ in range(n)]
    for j in range(3, n):
        if j % 2 == 0:
            for si in (0, 1, 2):
                mat[si][j] = mat[j][si] = 8.0
    report = assess(Structure(res, "t", pae=mat), task="docking", site=[1, 2, 3])
    assert report.verdict is not Level.PASS
    pae = [f for f in report.findings if "placement" in f.title or "relative" in f.title]
    assert pae and pae[0].level is Level.WARN


def test_a_pass_never_claims_nothing_exceeded_when_something_did():
    """A lone outlier is suppressed by --pae-min-run, which is that flag's job.

    The verdict stays PASS, but the finding must not claim "no core residue
    exceeds the cutoff" while an outlier sits above it - that is the
    contradiction, not the PASS itself.
    """
    n = 300
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, n + 1)]
    mat = [[0.5] * n for _ in range(n)]
    for i in range(n):
        mat[i][199] = mat[199][i] = 95.0
    report = assess(Structure(res, "t", pae=mat), task="docking", site=[1, 2, 3])
    pae = [f for f in report.findings if "placement" in f.title or "relative" in f.title][0]
    assert pae.level is Level.PASS
    assert "No" not in pae.detail.split(".")[0], pae.detail
    assert "exceed" in pae.detail and "95.0 A" in pae.detail


def test_cif_hetatm_ligands_are_not_parsed_as_residues(tmp_path):
    """A calcium ion has label_atom_id 'CA', exactly like an alpha carbon."""
    rows = [["ATOM", i, "CA", "ALA", "A", i, "A", i, 0.0, 0.0, 0.0, 1.0, 95.0]
            for i in (1, 2, 3)]
    rows.append(["HETATM", 4, "CA", "CA", "A", ".", "A", 200, 0.0, 0.0, 0.0, 1.0, 95.0])
    s = parse_structure(make_cif(tmp_path, CIF_HEADERS, rows))
    assert [r.number for r in s.residues] == [1, 2, 3]


def test_a_ligand_on_its_own_chain_does_not_trigger_the_multimer_refusal(tmp_path):
    """AlphaFold3 writes ligands as separate chains; that is not a multimer."""
    rows = [["ATOM", i, "CA", "ALA", "A", i, "A", i, 0.0, 0.0, 0.0, 1.0, 95.0]
            for i in (1, 2, 3)]
    rows.append(["HETATM", 4, "CA", "CA", "B", ".", "B", 200, 0.0, 0.0, 0.0, 1.0, 95.0])
    assert parse_structure(make_cif(tmp_path, CIF_HEADERS, rows)).n_residues == 3


def test_non_finite_plddt_is_refused():
    """_validate_pae rejects NaN; _validate_plddt must match, or --json emits NaN."""
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 21)]
    res[5] = Residue(6, "ALA", "A", float("nan"))
    with pytest.raises(ParseError, match="not a number|finite"):
        Structure(res, "t")


def test_json_encoder_refuses_to_emit_nan(capsys):
    """The second layer, tested by bypassing the first.

    _validate_plddt now rejects non-finite pLDDT, so NaN cannot reach the
    encoder by the normal route - which is exactly why this needs to mutate a
    residue after construction. The risk allow_nan=False guards is a future
    code path producing NaN downstream of validation; Python's json emits a
    bare NaN token that jq, Go and Rust all reject.
    """
    import foldguard.cli as cli
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 21)], "t")
    s.residues[0].plddt = float("nan")          # bypasses __post_init__
    report = assess(s, task="fold", site=[2, 3])
    with pytest.raises(ValueError, match="[Nn]a[Nn]"):
        cli._print_json(report, {})


def test_json_output_parses_on_a_normal_model(tmp_path, capsys):
    import foldguard.cli as cli
    path = make_pdb(tmp_path, [95.0] * 20)
    cli.main([str(path), "--json", "--site", "1-5"])
    out = capsys.readouterr().out
    assert "NaN" not in out and "Infinity" not in out
    json.loads(out)


def test_disorder_without_a_site_does_not_claim_anything_about_a_site():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 101)]
    for i in range(66, 101):
        res[i - 1] = Residue(i, "ALA", "A", 25.0)
    report = assess(Structure(res, "t"), task="fold")
    d = [f for f in report.findings if "isorder" in f.title]
    assert d, "expected a disorder finding"
    assert "overlap" not in d[0].detail.lower()
    assert "elsewhere" not in d[0].title.lower()


def test_pae_supplied_without_a_site_is_reported_as_unused():
    res = [Residue(i, "ALA", "A", 95.0) for i in range(1, 51)]
    report = assess(Structure(res, "t", pae=[[0.5] * 50 for _ in range(50)]), task="fold")
    assert any("PAE" in f.title for f in report.findings), [f.title for f in report.findings]


def test_unreadable_ca_records_are_refused_not_silently_dropped(tmp_path):
    """A truncated file otherwise yields a confident verdict on a short model.

    Worse, when the dropped records sit under the site, the resulting FAIL
    blames residue numbering - "predicted models are often renumbered from 1" -
    which sends the user looking in entirely the wrong place.
    """
    lines = [
        f"ATOM  {i*5:5d}  CA  ALA A{i:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{95.0:6.2f}           C"
        for i in range(1, 101)
    ]
    lines[89:] = [ln[:40] for ln in lines[89:]]          # records 90-100 truncated
    path = tmp_path / "partial.pdb"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ParseError, match="could not be read|unreadable"):
        parse_structure(path)


def test_intact_files_are_not_affected_by_the_dropped_record_check(tmp_path):
    path = make_pdb(tmp_path, [95.0] * 30)
    assert parse_structure(path).n_residues == 30


def test_assigning_pae_after_construction_is_validated():
    """core.py claims a Structure holding a matrix always holds a usable one."""
    s = Structure([Residue(i, "ALA", "A", 95.0) for i in range(1, 51)], "t")
    with pytest.raises(ParseError, match="3x3.*50 residues"):
        s.pae = [[0.1] * 3 for _ in range(3)]
