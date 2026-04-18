"""ESMFold smoke test. Requires a GPU."""

import os
import pytest


def test_fold_fn_produces_valid_pdb(gpu_available, tmp_path):
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory

    fold = fold_fn_factory(device="cuda:0")
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
    out = tmp_path / "esmf.pdb"
    fold(seq, str(out))

    assert out.exists()
    content = out.read_text()
    assert content.startswith(("ATOM", "HEADER", "PARENT", "TITLE", "REMARK", "MODEL"))
    assert "ATOM" in content
    # Sanity: PDB should encode at least len(seq) residues worth of ATOMs.
    num_atom_lines = sum(1 for line in content.splitlines() if line.startswith("ATOM"))
    assert num_atom_lines >= len(seq)


def test_fold_fn_singleton(gpu_available):
    """Second call should hit the cache, not reload the 2.6 GB weights."""
    if not gpu_available:
        pytest.skip("No GPU available")

    from evals.fold import fold_fn_factory, get_esmfold

    m1 = get_esmfold("cuda:0")
    m2 = get_esmfold("cuda:0")
    assert m1 is m2
