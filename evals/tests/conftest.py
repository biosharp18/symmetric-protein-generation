"""Shared fixtures for the evals test suite."""

import os
import shutil
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
C5_PDB = os.path.join(REPO_ROOT, "C5_2.pdb")
PMPNN_DIR = os.path.join(REPO_ROOT, "models", "framediff", "ProteinMPNN")
FOLDSEEK_DB = "/xuanwu-tank/west/gaorory/foldseek_db/pdb"


@pytest.fixture(scope="session")
def c5_pdb():
    """Path to a 5-chain × 100-residue C5 symmetric PDB (homo-oligomer fixture)."""
    assert os.path.exists(C5_PDB), f"test fixture missing: {C5_PDB}"
    return C5_PDB


@pytest.fixture(scope="session")
def monomer_pdb(tmp_path_factory, c5_pdb):
    """Single-chain PDB extracted from C5_2.pdb chain A."""
    from evals.tied_positions import extract_first_chain
    out = tmp_path_factory.mktemp("monomer") / "chainA.pdb"
    extract_first_chain(c5_pdb, str(out))
    assert out.exists()
    return str(out)


@pytest.fixture(scope="session")
def pmpnn_dir():
    assert os.path.exists(PMPNN_DIR), f"ProteinMPNN not found at {PMPNN_DIR}"
    return PMPNN_DIR


@pytest.fixture(scope="session")
def foldseek_db():
    """Return the FoldSeek DB prefix, or None if missing (tests should skip)."""
    if os.path.exists(FOLDSEEK_DB + ".dbtype"):
        return FOLDSEEK_DB
    return None


@pytest.fixture(scope="session")
def foldseek_available():
    return shutil.which("foldseek") is not None


@pytest.fixture(scope="session")
def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
