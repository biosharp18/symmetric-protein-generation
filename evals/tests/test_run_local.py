"""End-to-end CLI test for evals/run_local.py."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from pathlib import Path

from evals.run_local import _sample_id_from_path


def test_sample_id_flat_input(tmp_path):
    """RFdiffusion-style flat input — sample_id equals file stem."""
    pdb_dir = tmp_path / "inputs"
    pdb_dir.mkdir()
    pdb = pdb_dir / "c8_1.pdb"
    pdb.touch()
    assert _sample_id_from_path(pdb, pdb_dir) == "c8_1"


def test_sample_id_nested_input(tmp_path):
    """FrameDiff-style nested input — path flattened with double underscore."""
    pdb_dir = tmp_path / "inputs"
    nested = pdb_dir / "length_150" / "sample_0"
    nested.mkdir(parents=True)
    pdb = nested / "sample_1.pdb"
    pdb.touch()
    assert _sample_id_from_path(pdb, pdb_dir) == "length_150__sample_0__sample_1"


def test_sample_id_deep_nesting_is_unique(tmp_path):
    """Ensure siblings with same leaf filename do NOT collide."""
    pdb_dir = tmp_path / "inputs"
    for i in range(3):
        p = pdb_dir / "length_150" / f"sample_{i}"
        p.mkdir(parents=True)
        (p / "sample_1.pdb").touch()

    ids = {
        _sample_id_from_path(p, pdb_dir)
        for p in pdb_dir.glob("**/sample_1.pdb")
    }
    assert len(ids) == 3, f"collision! got {ids}"


def test_fold_neighbours_k_wiring(c5_pdb, pmpnn_dir, tmp_path):
    """CPU-only integration: --fold-neighbours-k=3 crops C5 → 3-chain subcomplex,
    records the new fields, and the tied_positions JSONL references the cropped
    chain IDs. SC / FoldSeek are skipped so no GPU or foldseek DB is needed.
    """
    from pathlib import Path
    from evals.run_local import evaluate_one

    sample_dir = tmp_path / "sample"
    result = evaluate_one(
        pdb_path=Path(c5_pdb),
        sample_dir=str(sample_dir),
        sample_id="c5_test",
        pmpnn_dir=pmpnn_dir,
        foldseek_db="/nonexistent",
        fold_fn=None,
        fold_fn_batch=None,
        num_seqs=2,
        sampling_temp=0.1,
        tied_positions_mode="auto",
        designability_tm=0.5,
        designability_rmsd=2.0,
        skip_clash=False,
        skip_sc=True,
        skip_foldseek=True,
        device="cpu",
        subunit_length=None,
        fold_neighbours_k=3,
    )

    # New fields populated.
    assert result["fold_neighbours_k"] == 3
    assert result["neighbours_pdb_path"] is not None
    assert result["neighbours_kept_chains"] == ["A", "B", "E"]
    assert len(result["neighbours_distances"]) == 3
    assert result["neighbours_distances"][0] == 0.0
    assert result["neighbours_distances"][1] < result["neighbours_distances"][2] + 1e-3

    # Cropped PDB exists and has exactly the renumbered subcomplex chains.
    neigh_path = Path(result["neighbours_pdb_path"])
    assert neigh_path.exists()
    from evals.neighbours import chain_centroids
    assert set(chain_centroids(str(neigh_path))) == {"A", "B", "C"}

    # tied_positions JSONL was produced against the cropped PDB (3 chains tied,
    # not 5).
    tied_path = result["tied_positions_used"]
    assert tied_path is not None and Path(tied_path).exists()
    with open(tied_path) as f:
        tied = json.loads(f.readline())
    # ProteinMPNN JSONL: one record with `masked_list`/tied-positions keyed by
    # chain IDs. Exactly the three kept (renumbered) chains should appear.
    chain_keys = {
        k for v in tied.values() if isinstance(v, list)
        for entry in v for k in (entry.keys() if isinstance(entry, dict) else [])
    }
    assert chain_keys == {"A", "B", "C"}


def test_fold_neighbours_k_noop_when_k_geq_chains(c5_pdb, pmpnn_dir, tmp_path):
    """k >= num_chains: no cropping, no extra PDB file, neighbours fields None."""
    from pathlib import Path
    from evals.run_local import evaluate_one

    sample_dir = tmp_path / "sample"
    result = evaluate_one(
        pdb_path=Path(c5_pdb),
        sample_dir=str(sample_dir),
        sample_id="c5_noop",
        pmpnn_dir=pmpnn_dir,
        foldseek_db="/nonexistent",
        fold_fn=None, fold_fn_batch=None,
        num_seqs=2, sampling_temp=0.1,
        tied_positions_mode="auto",
        designability_tm=0.5, designability_rmsd=2.0,
        skip_clash=False, skip_sc=True, skip_foldseek=True,
        device="cpu",
        subunit_length=None,
        fold_neighbours_k=5,  # == num_chains(C5) → no-op
    )
    assert result["fold_neighbours_k"] == 5
    assert result["neighbours_pdb_path"] is None
    assert result["neighbours_kept_chains"] is None


def test_cli_end_to_end(c5_pdb, pmpnn_dir, foldseek_db, gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("No GPU available")

    pdb_input_dir = tmp_path / "inputs"
    pdb_input_dir.mkdir()
    shutil.copy(c5_pdb, pdb_input_dir / "C5_2.pdb")

    output_dir = tmp_path / "outputs"

    cmd = [
        sys.executable, "-m", "evals.run_local",
        "--pdb-dir", str(pdb_input_dir),
        "--output-dir", str(output_dir),
        "--num-seqs", "2",
        "--pmpnn-dir", pmpnn_dir,
        "--tied-positions", "auto",
        "--device", "cuda:0",
    ]
    if foldseek_db is None:
        cmd.append("--skip-foldseek")
    else:
        cmd.extend(["--foldseek-db", foldseek_db])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ) + ":" + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # Per-sample JSON must exist and have expected keys.
    sample_json = output_dir / "per_sample" / "C5_2" / "sample_results.json"
    assert sample_json.exists(), f"missing {sample_json}"
    with open(sample_json) as f:
        result = json.load(f)
    assert result["num_chains"] == 5
    assert result["is_homo_oligomer"] is True
    assert result["clash_score"] is not None
    assert result["best_scTM"] is not None
    assert len(result["sc_results"]) >= 1

    # Aggregate CSVs must exist.
    assert (output_dir / "all_results.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "summary.json").exists()

    with open(output_dir / "summary.json") as f:
        summary = json.load(f)
    assert summary["num_samples"] == 1
    assert summary["mean_best_scTM"] is not None


def test_cli_idempotent(c5_pdb, pmpnn_dir, gpu_available, tmp_path):
    """Second run on the same --output-dir should skip already-done samples."""
    if not gpu_available:
        pytest.skip("No GPU available")

    pdb_input_dir = tmp_path / "inputs"
    pdb_input_dir.mkdir()
    shutil.copy(c5_pdb, pdb_input_dir / "C5_2.pdb")
    output_dir = tmp_path / "outputs"

    common = [
        sys.executable, "-m", "evals.run_local",
        "--pdb-dir", str(pdb_input_dir),
        "--output-dir", str(output_dir),
        "--num-seqs", "2",
        "--pmpnn-dir", pmpnn_dir,
        "--skip-foldseek",  # keep the test fast
        "--device", "cuda:0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ) + ":" + env.get("PYTHONPATH", "")

    # First run.
    proc1 = subprocess.run(common, env=env, capture_output=True, text=True, timeout=900)
    assert proc1.returncode == 0, proc1.stderr

    # Second run: should short-circuit.
    proc2 = subprocess.run(common, env=env, capture_output=True, text=True, timeout=120)
    assert proc2.returncode == 0
    assert "Skipping" in proc2.stderr or "already done" in proc2.stderr
