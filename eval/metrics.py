"""
Protein generation evaluation metrics.

Implements:
  - CA-CA steric clash detection
  - CA-CA bond deviation
  - Pairwise TM-score diversity
  - Secondary structure composition (via mdtraj DSSP)
  - Radius of gyration
"""

import numpy as np
import mdtraj as md
from tmtools import tm_align
from scipy.spatial import cKDTree
from itertools import combinations
from tqdm import tqdm


# ── Per-structure metrics ────────────────────────────────────────────────────

def extract_ca_coords(pdb_path: str) -> np.ndarray:
    """Extract CA atom coordinates (in Angstroms) from a PDB file."""
    traj = md.load(pdb_path)
    top = traj.topology
    ca_indices = top.select("name CA")
    # mdtraj uses nanometers internally; convert to Angstroms
    return traj.xyz[0, ca_indices, :] * 10.0


def ca_ca_clashes(ca_pos: np.ndarray, tol: float = 1.5):
    """
    Detect steric clashes between non-bonded CA atoms.
    Uses KD-tree for O(N log N) performance on large proteins.

    Args:
        ca_pos: (N, 3) array of CA positions in Angstroms.
        tol: distance threshold in Angstroms below which a pair is a clash.

    Returns:
        num_clashes: total number of clashing non-bonded pairs.
        clash_percent: fraction of non-bonded pairs that clash.
    """
    n = len(ca_pos)
    num_non_bonded = n * (n - 1) // 2 - (n - 1)  # total pairs minus adjacent
    if num_non_bonded <= 0:
        return 0, 0.0
    tree = cKDTree(ca_pos)
    close_pairs = tree.query_pairs(r=tol)
    # exclude sequentially adjacent residues (|i-j| == 1)
    num_clashes = sum(1 for i, j in close_pairs if abs(i - j) > 1)
    return num_clashes, float(num_clashes / num_non_bonded)


def ca_ca_bond_deviation(ca_pos: np.ndarray, ideal_bond: float = 3.8, tol: float = 0.1):
    """
    Measure deviation of sequential CA-CA distances from ideal bond length.

    Returns:
        mean_dev: mean absolute deviation from ideal (Angstroms).
        valid_percent: fraction of bonds within tolerance.
    """
    bond_dists = np.linalg.norm(np.diff(ca_pos, axis=0), axis=-1)
    mean_dev = float(np.mean(np.abs(bond_dists - ideal_bond)))
    valid_pct = float(np.mean(bond_dists < (ideal_bond + tol)))
    return mean_dev, valid_pct


def secondary_structure_metrics(pdb_path: str) -> dict:
    """Compute secondary structure fractions and radius of gyration."""
    traj = md.load(pdb_path)
    ss = md.compute_dssp(traj, simplified=True)
    rg = md.compute_rg(traj)[0]  # in nm
    return {
        "coil_percent": float(np.mean(ss == "C")),
        "helix_percent": float(np.mean(ss == "H")),
        "strand_percent": float(np.mean(ss == "E")),
        "radius_of_gyration_nm": float(rg),
    }


def per_sample_metrics(pdb_path: str) -> dict:
    """Compute all per-sample metrics for a single PDB file."""
    ca_pos = extract_ca_coords(pdb_path)
    num_clashes, clash_pct = ca_ca_clashes(ca_pos)
    bond_dev, bond_valid_pct = ca_ca_bond_deviation(ca_pos)
    ss = secondary_structure_metrics(pdb_path)
    return {
        "pdb_path": pdb_path,
        "num_residues": len(ca_pos),
        "num_ca_steric_clashes": num_clashes,
        "ca_steric_clash_percent": clash_pct,
        "ca_ca_bond_dev": bond_dev,
        "ca_ca_valid_percent": bond_valid_pct,
        **ss,
    }


# ── Diversity (pairwise TM-score) ───────────────────────────────────────────

def pairwise_tm_scores(ca_positions: list) -> np.ndarray:
    """
    Compute all pairwise TM-scores for a list of CA coordinate arrays.

    Uses the max of the two normalized TM-scores for each pair (symmetric).

    Args:
        ca_positions: list of (N_i, 3) arrays.

    Returns:
        Symmetric matrix of TM-scores (N_structures x N_structures).
    """
    n = len(ca_positions)
    total_pairs = n * (n - 1) // 2
    tm_matrix = np.eye(n)
    for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="    TM-score pairs"):
        seq_i = "A" * len(ca_positions[i])
        seq_j = "A" * len(ca_positions[j])
        res = tm_align(ca_positions[i], ca_positions[j], seq_i, seq_j)
        score = max(res.tm_norm_chain1, res.tm_norm_chain2)
        tm_matrix[i, j] = score
        tm_matrix[j, i] = score
    return tm_matrix


def diversity_score(ca_positions: list) -> dict:
    """
    Compute diversity as mean pairwise TM-score (lower = more diverse).

    Args:
        ca_positions: list of (N_i, 3) CA coordinate arrays.

    Returns:
        Dictionary with diversity score and number of structures compared.
    """
    n = len(ca_positions)
    if n < 2:
        return {"diversity_tm_mean": float("nan"), "num_structures": n}

    tm_matrix = pairwise_tm_scores(ca_positions)
    # extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    mean_tm = float(np.mean(tm_matrix[triu_idx]))
    return {
        "diversity_tm_mean": mean_tm,
        "num_structures": n,
        "num_pairs": len(triu_idx[0]),
    }
