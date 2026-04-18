"""Tests for the chain-splitting utility."""

import os
import subprocess
import sys

import pytest

from evals.split_chains import split_pdb_into_chains
from evals.tied_positions import chain_info


def test_split_c5_as_single_chain(c5_pdb, monomer_pdb, tmp_path):
    """Reverse of subunit extraction: take a monomer that came from C5 and
    verify splitting a concatenated version gives the right chain count.

    For this test we use c5_pdb directly as if it were one long chain: C5
    is 5 chains × 100 residues = 500, so splitting a hypothetical flattened
    version at subunit_length=100 should yield 5 chains. We simulate the
    flattened form by re-splitting c5_pdb — verify the output has 5 chains
    of 100 residues each.
    """
    out = tmp_path / "split.pdb"
    n = split_pdb_into_chains(c5_pdb, str(out), subunit_length=100)
    assert n == 5

    info = chain_info(str(out))
    assert info["num_chains"] == 5
    assert info["is_homo_oligomer"] is True
    assert set(info["chain_lengths"].values()) == {100}
    # Chain IDs should be A-E
    assert set(info["chain_lengths"].keys()) == {"A", "B", "C", "D", "E"}


def test_split_monomer_into_halves(monomer_pdb, tmp_path):
    """100-residue monomer → two chains of 50 residues each."""
    out = tmp_path / "halves.pdb"
    n = split_pdb_into_chains(monomer_pdb, str(out), subunit_length=50)
    assert n == 2
    info = chain_info(str(out))
    assert info["num_chains"] == 2
    assert set(info["chain_lengths"].values()) == {50}


def test_split_non_divisible_raises(monomer_pdb, tmp_path):
    """100 residues, subunit=30 → 100 % 30 != 0 → ValueError."""
    out = tmp_path / "bad.pdb"
    with pytest.raises(ValueError, match="not divisible"):
        split_pdb_into_chains(monomer_pdb, str(out), subunit_length=30)


def test_split_too_many_chains_raises(monomer_pdb, tmp_path):
    """100-residue PDB, subunit=1 → would need 100 chain IDs > 26 → error."""
    out = tmp_path / "huge.pdb"
    with pytest.raises(ValueError, match="Too many chains"):
        split_pdb_into_chains(monomer_pdb, str(out), subunit_length=1)


def test_split_cli(monomer_pdb, tmp_path):
    """Standalone CLI invocation via `python -m evals.split_chains`."""
    out = tmp_path / "cli.pdb"
    env = os.environ.copy()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    env["PYTHONPATH"] = repo_root + ":" + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [
            sys.executable, "-m", "evals.split_chains",
            monomer_pdb, str(out), "--subunit-length", "50",
        ],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out.exists()
    assert "Wrote 2 chains" in proc.stdout
    info = chain_info(str(out))
    assert info["num_chains"] == 2
