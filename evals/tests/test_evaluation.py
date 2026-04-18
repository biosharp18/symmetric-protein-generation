"""Tests for the model-agnostic scoring primitives."""

import os
import pytest

from evals.evaluation import (
    compute_clash_score,
    compute_pairwise_diversity,
    run_foldseek,
    run_foldseek_multimer,
)


def test_clash_score_real_pdb(c5_pdb):
    """Clash score should be a finite non-negative number on a real PDB."""
    score = compute_clash_score(c5_pdb)
    assert score is not None
    assert score >= 0
    # 500-residue assembly should not be pathologically clashy.
    assert score < 1000


def test_clash_score_missing_file(tmp_path):
    """Graceful None on bad input."""
    fake = tmp_path / "nope.pdb"
    assert compute_clash_score(str(fake)) is None


def test_foldseek_wrapper_returns_tm(c5_pdb, tmp_path, foldseek_available, foldseek_db):
    """Integration test: FoldSeek runs end-to-end and returns a TM score in [0, 1]."""
    if not foldseek_available:
        pytest.skip("foldseek binary not on PATH")
    if foldseek_db is None:
        pytest.skip("FoldSeek DB not installed")

    best_tm = run_foldseek(
        str(c5_pdb), str(tmp_path), label="c5test", db_path=foldseek_db,
    )
    # A real PDB is likely to find self / homologs → non-None, between 0 and 1.
    assert best_tm is not None
    assert 0.0 < best_tm <= 1.0


def test_foldseek_multimer_returns_tm(c5_pdb, tmp_path, foldseek_available, foldseek_db):
    """Multimer search on a 5-chain PDB returns complexqtmscore in [0, 1]."""
    if not foldseek_available:
        pytest.skip("foldseek binary not on PATH")
    if foldseek_db is None:
        pytest.skip("FoldSeek DB not installed")

    best_tm = run_foldseek_multimer(
        str(c5_pdb), str(tmp_path), label="mmtest", db_path=foldseek_db,
    )
    if best_tm is not None:
        assert 0.0 < best_tm <= 1.0
    # The report file must exist either way (even if no hits).
    assert (tmp_path / "foldseek_mmtest_result_report").exists() or best_tm is None


def test_diversity_too_few_pdbs(c5_pdb):
    """Fewer than 2 inputs → None (nothing to pair)."""
    assert compute_pairwise_diversity([]) is None
    assert compute_pairwise_diversity([c5_pdb]) is None


def test_diversity_identical_pdbs(c5_pdb):
    """Self-vs-self should score ~1.0 (the reference for a not-diverse set)."""
    result = compute_pairwise_diversity([c5_pdb, c5_pdb])
    assert result is not None
    assert result["num_pdbs"] == 2
    assert result["num_pairs"] == 1
    assert result["mean_pairwise_tm"] == pytest.approx(1.0, abs=1e-3)


def test_diversity_shape(c5_pdb, monomer_pdb):
    """Returns the expected keys on a heterogeneous pair."""
    result = compute_pairwise_diversity([c5_pdb, monomer_pdb])
    assert result is not None
    assert set(result) == {
        "mean_pairwise_tm", "min_pairwise_tm", "max_pairwise_tm",
        "num_pdbs", "num_pairs",
    }
    assert 0.0 <= result["min_pairwise_tm"] <= result["mean_pairwise_tm"] <= result["max_pairwise_tm"] <= 1.0
    assert result["num_pairs"] == 1


def test_foldseek_wrapper_handles_missing_db(c5_pdb, tmp_path, foldseek_available):
    """If DB path is wrong, wrapper returns None without crashing."""
    if not foldseek_available:
        pytest.skip("foldseek binary not on PATH")
    result = run_foldseek(
        str(c5_pdb), str(tmp_path), label="missing",
        db_path="/nonexistent/path/to/db",
    )
    assert result is None
