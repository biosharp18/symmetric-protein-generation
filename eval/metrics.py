"""
Protein generation evaluation metrics.

Implements:
  - CA-CA steric clash detection
  - CA-CA bond deviation
  - Pairwise TM-score diversity
  - Secondary structure composition (via mdtraj DSSP)
  - Radius of gyration
  - Self-consistency (ProteinMPNN + ESMFold scTM / scRMSD)
"""

import os
import subprocess
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


# ── Self-consistency (ProteinMPNN → ESMFold → scTM/scRMSD) ────────────────────


def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Align A onto B via SVD-based rigid body superposition. Returns aligned A."""
    assert A.shape == B.shape
    A = A.T
    B = B.T
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ Bm.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return (R @ A + t).T


def calc_aligned_rmsd(pos_1: np.ndarray, pos_2: np.ndarray) -> float:
    """RMSD between two coordinate arrays after rigid alignment."""
    aligned = rigid_transform_3D(pos_1, pos_2)
    return float(np.mean(np.linalg.norm(aligned - pos_2, axis=-1)))


def _get_chain_info(pdb_path: str) -> list[tuple[str, int]]:
    """Get chain IDs and residue counts from a PDB file.

    Returns list of (chain_id, num_residues) tuples.
    """
    traj = md.load(pdb_path)
    chains = []
    for chain in traj.topology.chains:
        chains.append((chr(ord('A') + chain.index), chain.n_residues))
    return chains


def _make_tied_positions_jsonl(
    parsed_jsonl_path: str,
    output_path: str,
):
    """Generate tied_positions_jsonl for homooligomeric design.

    Ties all positions across all chains so every chain gets the same sequence.
    Equivalent to running make_tied_positions_dict.py --homooligomer 1.
    """
    import json
    with open(parsed_jsonl_path, 'r') as f:
        json_list = list(f)

    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)
        all_chain_list = sorted(
            [item[-1:] for item in list(result) if item[:9] == 'seq_chain']
        )
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        tied_positions_list = []
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for chain in all_chain_list:
                temp_dict[chain] = [i]
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list

    with open(output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


def run_proteinmpnn(
    pdb_path: str,
    pmpnn_dir: str,
    output_dir: str,
    num_seqs: int = 8,
    sampling_temp: float = 0.1,
    seed: int = 123,
) -> list[str]:
    """Run ProteinMPNN inverse folding on a PDB file.

    For multi-chain (symmetric) PDBs, uses tied positions so all chains
    get the same designed sequence.

    Returns list of designed sequences (excluding the native/input sequence).
    For multi-chain PDBs, each sequence contains '/' separating chains.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy PDB into a temp input dir so parse_multiple_chains finds only this file
    input_dir = os.path.join(output_dir, "pmpnn_input")
    os.makedirs(input_dir, exist_ok=True)
    import shutil
    dest_pdb = os.path.join(input_dir, os.path.basename(pdb_path))
    shutil.copy2(pdb_path, dest_pdb)

    # Step 1: parse PDB to JSONL
    jsonl_path = os.path.join(output_dir, "parsed_pdbs.jsonl")
    subprocess.run(
        [
            "python",
            os.path.join(pmpnn_dir, "helper_scripts", "parse_multiple_chains.py"),
            f"--input_path={input_dir}",
            f"--output_path={jsonl_path}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Step 1b: detect multi-chain and create tied positions for homooligomers
    chain_info = _get_chain_info(pdb_path)
    is_multichain = len(chain_info) > 1
    tied_positions_path = None
    if is_multichain:
        tied_positions_path = os.path.join(output_dir, "tied_positions.jsonl")
        _make_tied_positions_jsonl(jsonl_path, tied_positions_path)

    # Step 2: run ProteinMPNN
    pmpnn_args = [
        "python",
        os.path.join(pmpnn_dir, "protein_mpnn_run.py"),
        "--out_folder", output_dir,
        "--jsonl_path", jsonl_path,
        "--num_seq_per_target", str(num_seqs),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    if tied_positions_path is not None:
        pmpnn_args.extend(["--tied_positions_jsonl", tied_positions_path])

    subprocess.run(
        pmpnn_args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Step 3: parse output FASTA
    from biotite.sequence.io import fasta
    pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
    fasta_path = os.path.join(output_dir, "seqs", f"{pdb_stem}.fa")
    fasta_file = fasta.FastaFile.read(fasta_path)
    # First entry is the "native" (input) sequence, rest are designed
    sequences = []
    for i, (header, seq) in enumerate(fasta_file.items()):
        if i == 0:
            continue  # skip native
        sequences.append(seq)
    return sequences


def run_esmfold(sequence: str, save_path: str, folding_model) -> str:
    """Run ESMFold on a single sequence or multimer.

    For multimer prediction, pass chains separated by ':' (e.g. "MKTV:MKTV").
    """
    import torch
    with torch.no_grad():
        pdb_string = folding_model.infer_pdb(sequence)
    with open(save_path, "w") as f:
        f.write(pdb_string)
    return pdb_string


def self_consistency_metrics(
    pdb_path: str,
    folding_model,
    pmpnn_dir: str,
    sc_output_dir: str,
    num_seqs: int = 8,
    sampling_temp: float = 0.1,
) -> dict:
    """Run the full self-consistency pipeline for one backbone PDB.

    Handles both single-chain and multi-chain (symmetric) proteins:
      - Multi-chain: ProteinMPNN runs with tied positions (homooligomer mode),
        ESMFold folds the full complex using ':' chain separator.
      - Single-chain: standard pipeline.

    Saves all intermediate outputs (sequences, refolded PDBs, CSV) to sc_output_dir.

    Returns dict with best scTM, best scRMSD, and per-sequence results.
    """
    import pandas as pd

    os.makedirs(sc_output_dir, exist_ok=True)

    # Detect multi-chain
    chain_info = _get_chain_info(pdb_path)
    num_chains = len(chain_info)

    # 1. Run ProteinMPNN (with tied positions for multi-chain)
    pmpnn_output_dir = os.path.join(sc_output_dir, "pmpnn")
    sequences = run_proteinmpnn(
        pdb_path, pmpnn_dir, pmpnn_output_dir,
        num_seqs=num_seqs, sampling_temp=sampling_temp,
    )
    if not sequences:
        return {"sc_tm": float("nan"), "sc_rmsd": float("nan"), "num_seqs_designed": 0}

    # Extract CA coords from original backbone (all chains)
    original_ca = extract_ca_coords(pdb_path)
    original_len = len(original_ca)
    poly_seq = "A" * original_len

    # 2. ESMFold each sequence + 3. compute metrics
    esmf_dir = os.path.join(sc_output_dir, "esmf")
    os.makedirs(esmf_dir, exist_ok=True)

    results = {
        "sequence_idx": [],
        "tm_score": [],
        "rmsd": [],
        "sequence": [],
        "esmf_pdb_path": [],
    }

    for i, seq in enumerate(sequences):
        esmf_path = os.path.join(esmf_dir, f"refolded_{i}.pdb")
        try:
            # For multi-chain: ProteinMPNN outputs "SEQ_A/SEQ_B/SEQ_C"
            # ESMFold expects "SEQ_A:SEQ_B:SEQ_C" for multimer prediction
            if num_chains > 1 and "/" in seq:
                esmf_input = seq.replace("/", ":")
            else:
                esmf_input = seq

            run_esmfold(esmf_input, esmf_path, folding_model)
            refolded_ca = extract_ca_coords(esmf_path)

            # Lengths must match for comparison
            if len(refolded_ca) != original_len:
                print(f"    WARN: length mismatch for seq {i}: "
                      f"original={original_len}, refolded={len(refolded_ca)}, skipping")
                continue

            # scTM
            res = tm_align(original_ca, refolded_ca, poly_seq, poly_seq)
            sc_tm = max(res.tm_norm_chain1, res.tm_norm_chain2)

            # scRMSD
            sc_rmsd = calc_aligned_rmsd(original_ca, refolded_ca)

            results["sequence_idx"].append(i)
            results["tm_score"].append(sc_tm)
            results["rmsd"].append(sc_rmsd)
            results["sequence"].append(seq)
            results["esmf_pdb_path"].append(esmf_path)
        except Exception as e:
            print(f"    WARN: ESMFold/metrics failed for seq {i}: {e}")

    # Save per-sequence results CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(sc_output_dir, "sc_results.csv"), index=False)

    if not results["tm_score"]:
        return {"sc_tm": float("nan"), "sc_rmsd": float("nan"), "num_seqs_designed": len(sequences)}

    # Report best (max scTM) across designed sequences
    best_idx = int(np.argmax(results["tm_score"]))
    return {
        "sc_tm": results["tm_score"][best_idx],
        "sc_rmsd": results["rmsd"][best_idx],
        "sc_tm_mean": float(np.mean(results["tm_score"])),
        "sc_rmsd_mean": float(np.mean(results["rmsd"])),
        "num_seqs_designed": len(sequences),
        "num_seqs_successful": len(results["tm_score"]),
        "num_chains": num_chains,
    }
