"""Tests for tied-positions auto-inference and subunit extraction."""

import json
import os

import pytest

from evals.tied_positions import (
    chain_info,
    extract_first_chain,
    infer_tied_positions,
)


def test_chain_info_homo_oligomer(c5_pdb):
    info = chain_info(c5_pdb)
    assert info["num_chains"] == 5
    assert set(info["chain_lengths"].values()) == {100}
    assert info["is_homo_oligomer"] is True


def test_chain_info_monomer(monomer_pdb):
    info = chain_info(monomer_pdb)
    assert info["num_chains"] == 1
    assert info["is_homo_oligomer"] is False


def test_infer_tied_positions_homo_oligomer(c5_pdb, tmp_path):
    out = tmp_path / "tied.jsonl"
    result = infer_tied_positions(c5_pdb, str(out), pdb_name="C5_2")
    assert result == str(out)
    assert out.exists()

    with open(out) as f:
        data = json.loads(f.read())
    assert list(data.keys()) == ["C5_2"]
    tied = data["C5_2"]
    assert len(tied) == 100, "should tie all 100 residue positions"
    # First position should tie A-E at residue 1.
    assert tied[0] == {"A": [1], "B": [1], "C": [1], "D": [1], "E": [1]}
    # Last position should tie at residue 100.
    assert tied[99] == {"A": [100], "B": [100], "C": [100], "D": [100], "E": [100]}


def test_infer_tied_positions_monomer(monomer_pdb, tmp_path):
    out = tmp_path / "tied.jsonl"
    # Single-chain: no tying.
    assert infer_tied_positions(monomer_pdb, str(out)) is None
    assert not out.exists()


def test_extract_first_chain(c5_pdb, tmp_path):
    out = tmp_path / "chainA.pdb"
    result = extract_first_chain(c5_pdb, str(out))
    assert result == str(out)
    # Extracted file should have only one chain.
    info = chain_info(str(out))
    assert info["num_chains"] == 1
    assert list(info["chain_lengths"].values()) == [100]


def test_extract_first_chain_monomer_returns_none(monomer_pdb, tmp_path):
    out = tmp_path / "chainA.pdb"
    assert extract_first_chain(monomer_pdb, str(out)) is None
    assert not out.exists()
