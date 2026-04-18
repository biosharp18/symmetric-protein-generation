"""End-to-end test for run_self_consistency_generic.

Runs ProteinMPNN + ESMFold on the C5 fixture with tied positions.
"""

import os

import pytest

from evals.evaluation import run_self_consistency_generic
from evals.tied_positions import infer_tied_positions


def test_run_self_consistency_batched_matches_sequential(
    c5_pdb, pmpnn_dir, gpu_available, tmp_path,
):
    """Batched and sequential SC paths should produce near-identical scTM/scRMSD.

    Uses the same PMPNN seed so the designed sequences are identical in both
    runs; any divergence comes from ESMFold batch non-determinism.
    """
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory, fold_fn_batch_factory

    tied_jsonl = str(tmp_path / "tied.jsonl")
    os.makedirs(os.path.dirname(tied_jsonl), exist_ok=True)
    assert infer_tied_positions(c5_pdb, tied_jsonl, pdb_name="C5_2") is not None

    fold_fn = fold_fn_factory(device="cuda:0")
    fold_fn_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)

    # Sequential baseline
    seq_dir = tmp_path / "seq_run"
    seq_results = run_self_consistency_generic(
        pdb_path=c5_pdb, output_dir=str(seq_dir), pmpnn_dir=pmpnn_dir,
        seq_per_sample=2, tied_positions_path=tied_jsonl,
        fold_fn=fold_fn, gpu_id=0,
    )

    # Batched run
    bat_dir = tmp_path / "bat_run"
    bat_results = run_self_consistency_generic(
        pdb_path=c5_pdb, output_dir=str(bat_dir), pmpnn_dir=pmpnn_dir,
        seq_per_sample=2, tied_positions_path=tied_jsonl,
        fold_fn=fold_fn, fold_fn_batch=fold_fn_batch, gpu_id=0,
    )

    assert len(seq_results) == len(bat_results) >= 1
    # ProteinMPNN is seeded → same sequences, same order.
    for s, b in zip(seq_results, bat_results):
        assert s["sequence"] == b["sequence"], (
            f"sequence mismatch (check PMPNN determinism): {s['header']} vs {b['header']}"
        )
        # fp16 batch noise budget on a small monomer-ish fold:
        assert abs(s["scTM"] - b["scTM"]) < 0.03, (
            f"scTM diverged: sequential={s['scTM']:.4f} batched={b['scTM']:.4f}"
        )
        assert abs(s["scRMSD"] - b["scRMSD"]) < 1.0, (
            f"scRMSD diverged: sequential={s['scRMSD']:.3f} batched={b['scRMSD']:.3f}"
        )


def test_run_self_consistency_end_to_end(
    c5_pdb, pmpnn_dir, gpu_available, tmp_path,
):
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory

    # Set up tied positions for the C5 fixture.
    tied_jsonl = str(tmp_path / "self_consistency" / "tied.jsonl")
    os.makedirs(os.path.dirname(tied_jsonl), exist_ok=True)
    result = infer_tied_positions(c5_pdb, tied_jsonl, pdb_name="C5_2")
    assert result is not None

    fold_fn = fold_fn_factory(device="cuda:0")
    results = run_self_consistency_generic(
        pdb_path=c5_pdb,
        output_dir=str(tmp_path),
        pmpnn_dir=pmpnn_dir,
        seq_per_sample=2,  # small for speed
        tied_positions_path=tied_jsonl,
        fold_fn=fold_fn,
        gpu_id=0,
    )

    assert len(results) >= 1, "expected at least one sequence folded"
    for r in results:
        assert "scTM" in r and "scRMSD" in r
        assert 0.0 <= r["scTM"] <= 1.0 + 1e-6, f"scTM out of range: {r['scTM']}"
        assert r["scRMSD"] >= 0.0

        # Tied design: every chain in the sequence must be identical.
        seq = r["sequence"]
        chains = seq.split("/")
        assert len(chains) == 5, f"expected 5 tied chains, got {len(chains)}"
        assert all(c == chains[0] for c in chains), (
            "chains not identical despite tied positions"
        )

    # Artifact checks.
    assert os.path.exists(os.path.join(tmp_path, "self_consistency", "sc_results.csv"))
    assert os.path.exists(
        os.path.join(tmp_path, "self_consistency", "seqs", "C5_2.fa")
    )
    # ESMFold outputs
    esmf_dir = os.path.join(tmp_path, "self_consistency", "esmfold")
    assert any(f.endswith(".pdb") for f in os.listdir(esmf_dir))
