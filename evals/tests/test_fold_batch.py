"""Tests for the batched ESMFold fold_fn_batch factory.

Equivalence and robustness checks: batched folding should produce PDBs
structurally near-identical to per-sequence folding (modulo fp16 batch
non-determinism), must handle mixed lengths, and must gracefully fall
back to per-sequence folding when a batch fails.
"""

import os

import pytest


_SHORT_SEQS = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",  # 53 aa
    "GPLLTLLMGVGGLIFPGLVGGLMVLGALLPSLRPAIERLAAEAMALAAKAAAE",  # 53 aa
    "GLALLVLAGVGGLLLPGFVGGLMVLGALLPSLLPLLERLAARLYAAAAAAAAA",  # 53 aa
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQW",  # 53 aa
]


def _ca_positions(pdb_path: str):
    """Read CA coordinates in Angstroms."""
    import mdtraj as md
    traj = md.load(pdb_path)
    ca_idx = traj.topology.select("name CA")
    return traj.xyz[0, ca_idx] * 10.0


def _kabsch_rmsd(p1, p2):
    """Optimally-aligned CA RMSD between two same-length point sets."""
    import numpy as np
    assert len(p1) == len(p2)
    a = p1 - p1.mean(axis=0)
    b = p2 - p2.mean(axis=0)
    h = b.T @ a
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = b @ rot.T
    return float(np.sqrt(np.mean(np.sum((aligned - a) ** 2, axis=-1))))


def test_batch_single_sequence_matches_per_seq(gpu_available, tmp_path):
    """fold_fn_batch([seq], [path]) should agree with fold_fn(seq, path)."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory, fold_fn_batch_factory

    fold_one = fold_fn_factory(device="cuda:0")
    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)

    seq = _SHORT_SEQS[0]
    p_seq = tmp_path / "per_seq.pdb"
    p_bat = tmp_path / "batched.pdb"
    fold_one(seq, str(p_seq))
    fold_batch([seq], [str(p_bat)])

    assert p_seq.exists() and p_bat.exists()
    ca_s = _ca_positions(str(p_seq))
    ca_b = _ca_positions(str(p_bat))
    assert len(ca_s) == len(ca_b) == len(seq)
    # batch-of-one should be deterministically identical; allow tiny numerical noise
    rmsd = _kabsch_rmsd(ca_s, ca_b)
    assert rmsd < 0.2, f"batch-of-one diverged from per-seq: RMSD={rmsd:.3f}"


def test_batch_multiple_sequences_equivalent(gpu_available, tmp_path):
    """Each sequence's batched fold should match its per-seq fold within tolerance."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory, fold_fn_batch_factory

    fold_one = fold_fn_factory(device="cuda:0")
    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)

    per_seq_paths = [tmp_path / f"seq_{i}.pdb" for i in range(len(_SHORT_SEQS))]
    batch_paths = [tmp_path / f"bat_{i}.pdb" for i in range(len(_SHORT_SEQS))]
    for s, p in zip(_SHORT_SEQS, per_seq_paths):
        fold_one(s, str(p))
    fold_batch(_SHORT_SEQS, [str(p) for p in batch_paths])

    for i, (ps, pb) in enumerate(zip(per_seq_paths, batch_paths)):
        assert ps.exists() and pb.exists(), f"missing PDB for seq {i}"
        ca_s = _ca_positions(str(ps))
        ca_b = _ca_positions(str(pb))
        assert len(ca_s) == len(ca_b) == len(_SHORT_SEQS[i])
        rmsd = _kabsch_rmsd(ca_s, ca_b)
        # fp16 batch nondeterminism budget: generous (1.0 Å) since batching
        # changes reduction order. If this ever trips, investigate before
        # loosening.
        assert rmsd < 1.0, f"seq {i}: batched vs per-seq RMSD={rmsd:.3f}"


def test_batch_heterogeneous_lengths(gpu_available, tmp_path):
    """Mixed-length sequences in one call should all produce valid PDBs of the right length."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_batch_factory

    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)

    mixed = [_SHORT_SEQS[0][:30], _SHORT_SEQS[1], _SHORT_SEQS[2][:40]]  # 30, 53, 40
    paths = [tmp_path / f"m_{i}.pdb" for i in range(len(mixed))]
    fold_batch(mixed, [str(p) for p in paths])

    for seq, p in zip(mixed, paths):
        assert p.exists(), f"missing PDB: {p}"
        ca = _ca_positions(str(p))
        assert len(ca) == len(seq), f"expected {len(seq)} CAs, got {len(ca)}"


def test_batch_empty_input(gpu_available):
    """Empty input should be a no-op, not an error."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_batch_factory

    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)
    fold_batch([], [])  # should not raise


def test_batch_length_mismatch_raises(gpu_available):
    """fold_fn_batch must reject mismatched seq/path list lengths."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_batch_factory

    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)
    with pytest.raises(AssertionError):
        fold_batch(["A" * 30], ["p1.pdb", "p2.pdb"])


def test_batch_oom_falls_back(gpu_available, tmp_path, monkeypatch):
    """Simulate a batch OOM; wrapper must fall back to per-sequence folding."""
    if not gpu_available:
        pytest.skip("No GPU available")

    import torch
    from evals import fold as fold_module
    from evals.fold import fold_fn_batch_factory

    fold_batch = fold_fn_batch_factory(device="cuda:0", batch_size=4)

    model = fold_module.get_esmfold("cuda:0")
    original_infer_pdbs = model.infer_pdbs
    call_count = {"n": 0}

    def flaky_infer_pdbs(seqs, *args, **kwargs):
        # Fail only the first time we're called with more than one seq.
        call_count["n"] += 1
        if len(seqs) > 1 and call_count["n"] == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return original_infer_pdbs(seqs, *args, **kwargs)

    monkeypatch.setattr(model, "infer_pdbs", flaky_infer_pdbs)

    seqs = _SHORT_SEQS[:2]
    paths = [tmp_path / "oom_0.pdb", tmp_path / "oom_1.pdb"]
    fold_batch(seqs, [str(p) for p in paths])

    # Both PDBs should exist via the per-seq fallback path.
    for s, p in zip(seqs, paths):
        assert p.exists(), f"fallback failed for {p}"
        assert len(_ca_positions(str(p))) == len(s)
