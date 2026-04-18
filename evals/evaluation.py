"""Shared evaluation pipeline: self-consistency, FoldSeek, clash score.

Model-agnostic — works with any PDB file regardless of how it was generated.
"""

import os
import sys
import json
import subprocess
import logging
import shutil

log = logging.getLogger(__name__)


def run_foldseek(pdb_path: str, output_dir: str, label: str = "complex",
                 db_path: str = "/root/foldseek_db/pdb",
                 timeout: int = 300) -> float | None:
    """Run FoldSeek easy-search against PDB and return best TM-score.

    Returns None if no valid hits found or FoldSeek fails.
    """
    result_path = os.path.join(output_dir, f"foldseek_{label}_results.tsv")
    tmp_dir = os.path.join(output_dir, f"foldseek_{label}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-search",
        pdb_path, db_path, result_path, tmp_dir,
        # qtmscore = TM-score normalized by QUERY length, always in [0, 1].
        # This is the standard TM-score from TM-align that the protein
        # generation literature uses for PDB-novelty assessment. alntmscore
        # (normalized by alignment length) is NOT appropriate here — it
        # rewards short partial motif matches with values > 1.
        "--format-output", "query,target,alntmscore",
        #"-e", "inf",
        "--alignment-type", "1",  # global alignment (TM-align)
        "--max-seqs", "1000",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            log.warning(f"FoldSeek ({label}) failed: {proc.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        log.warning(f"FoldSeek ({label}) timed out")
        return None

    best_tm = 0.0
    if os.path.exists(result_path):
        with open(result_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    try:
                        tm = float(parts[2])
                        if tm <= 1.0:
                            best_tm = max(best_tm, tm)
                    except ValueError:
                        pass
    return best_tm if best_tm > 0 else None


def run_foldseek_multimer(pdb_path: str, output_dir: str, label: str = "complex",
                          db_path: str = "/xuanwu-tank/west/gaorory/foldseek_db/pdb",
                          timeout: int = 600) -> float | None:
    """Run FoldSeek multimer search (complex-level alignment) and return best complexqtmscore.

    Uses `foldseek easy-multimersearch`, which treats the input as a multi-chain
    assembly and searches for whole-complex matches against the target DB
    (grouping target chains by assembly prefix like `1xyz-assembly1_*`).

    The `result_report` file emitted by foldseek has fixed columns:
      query | target | qchains | tchains | complexqtmscore | complexttmscore |
      complexu | complext | qcomplexcoverage | tcomplexcoverage |
      qchaintms | tchaintms | interfacelddt | complexassignid

    Returns the best complexqtmscore (query-length-normalized, in [0, 1]) or
    None on failure / empty result.
    """
    result_path = os.path.join(output_dir, f"foldseek_{label}_result")
    report_path = result_path + "_report"
    tmp_dir = os.path.join(output_dir, f"foldseek_{label}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "foldseek", "easy-multimersearch",
        pdb_path, db_path, result_path, tmp_dir,
        # Multimer search requires the default 3Di alignment — TMalign mode
        # (--alignment-type 1) silently produces empty output. complexqtmscore
        # is already normalized by query length so it stays in [0, 1].
        # `-e inf` is required: FrameDiff/RFdiffusion outputs are far from PDB,
        # so the default e-value filter drops every hit and gives empty report.
        "-e", "inf",
        "--max-seqs", "300",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            log.warning(f"FoldSeek multimer ({label}) failed: {proc.stderr[:500]}")
            return None
    except subprocess.TimeoutExpired:
        log.warning(f"FoldSeek multimer ({label}) timed out")
        return None

    best_tm = 0.0
    if os.path.exists(report_path):
        with open(report_path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                try:
                    tm = float(parts[4])
                    if 0.0 <= tm <= 1.0:
                        best_tm = max(best_tm, tm)
                except ValueError:
                    pass
    return best_tm if best_tm > 0 else None


def compute_pairwise_diversity(pdb_paths: list[str]) -> dict | None:
    """Mean pairwise TM-score across a set of PDBs (the run's "diversity" metric).

    Loads each PDB's CA coordinates and runs tmtools.tm_align on every unordered
    pair, using max(tm_norm_chain1, tm_norm_chain2) to match scTM's symmetric
    convention. For symmetric assemblies the chains are emitted in a consistent
    order (A, B, C, ...), so concatenated-CA alignment is equivalent to proper
    multi-chain TM-align.

    A lower mean_pairwise_tm means more structurally diverse outputs.

    Returns None if fewer than 2 usable PDBs.
    """
    import itertools
    import numpy as np
    import mdtraj as md
    from tmtools import tm_align

    entries = []
    for path in pdb_paths:
        try:
            traj = md.load(path)
            ca_idx = traj.topology.select("name CA")
            if len(ca_idx) == 0:
                log.warning(f"Diversity: {path} has no CA atoms, skipping")
                continue
            pos = traj.xyz[0, ca_idx] * 10.0  # nm -> A
            seq = "A" * len(ca_idx)
            entries.append((path, pos, seq))
        except Exception as e:
            log.warning(f"Diversity: failed to load CAs from {path}: {e}")

    if len(entries) < 2:
        return None

    tms = []
    for (_, pa, sa), (_, pb, sb) in itertools.combinations(entries, 2):
        try:
            res = tm_align(pa, pb, sa, sb)
            tms.append(float(max(res.tm_norm_chain1, res.tm_norm_chain2)))
        except Exception as e:
            log.warning(f"Diversity: tm_align failed: {e}")

    if not tms:
        return None

    arr = np.array(tms)
    return {
        "mean_pairwise_tm": float(arr.mean()),
        "min_pairwise_tm": float(arr.min()),
        "max_pairwise_tm": float(arr.max()),
        "num_pdbs": len(entries),
        "num_pairs": len(tms),
    }


def compute_clash_score(pdb_path: str, clash_threshold: float = 1.5) -> float | None:
    """Compute steric clashes per 1000 atoms from a PDB file.

    Uses mdtraj for all-atom analysis. Falls back gracefully on error.
    """
    try:
        import mdtraj as md
        import numpy as np

        traj = md.load(pdb_path)
        heavy_idx = traj.topology.select("not element H")
        if len(heavy_idx) == 0:
            return 0.0
        heavy_pos = traj.xyz[0, heavy_idx]  # nm

        res_idx = np.array([
            traj.topology.atom(i).residue.index for i in heavy_idx
        ])

        diff = heavy_pos[:, None, :] - heavy_pos[None, :, :]
        dists = np.linalg.norm(diff, axis=-1) * 10.0  # nm -> A

        n = len(heavy_idx)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        pair_res_sep = np.abs(res_idx[i_idx] - res_idx[j_idx])

        mask = (pair_res_sep > 2) & (pair_dists < clash_threshold)
        n_clashes = int(np.sum(mask))
        return n_clashes / (n / 1000.0)
    except Exception as e:
        log.warning(f"Clash score failed: {e}")
        return None


def run_self_consistency_generic(
    pdb_path: str,
    output_dir: str,
    pmpnn_dir: str,
    seq_per_sample: int = 8,
    tied_positions_path: str | None = None,
    fold_fn=None,
    fold_fn_batch=None,
    gpu_id: int = 0,
    sampling_temp: float = 0.1,
    pmpnn_seed: int = 38,
) -> list[dict]:
    """Run ProteinMPNN -> fold -> scTM/scRMSD pipeline.

    This is a generic version that accepts a fold callback.
    For model-specific self-consistency, the generator can provide
    its own Sampler.run_self_consistency method instead.

    Args:
        pdb_path: reference backbone PDB.
        output_dir: directory for outputs (seqs/, esmf/, sc_results.csv).
        pmpnn_dir: path to ProteinMPNN repo.
        seq_per_sample: number of ProteinMPNN sequences.
        tied_positions_path: optional tied positions JSONL for symmetric design.
        fold_fn: callable(sequence, save_path) that folds one sequence to PDB.
            Used when `fold_fn_batch` is None (per-sequence loop).
        fold_fn_batch: optional callable(list[str], list[str]) that folds all
            sequences at once (batched ESMFold). If provided, takes precedence
            over `fold_fn` — the per-sequence loop is replaced by one batched
            call and a subsequent metrics loop over the written PDBs.
        gpu_id: CUDA device.

    Returns:
        List of dicts with keys: header, sequence, scTM, scRMSD.
    """
    import pandas as pd

    # Copy reference PDB into output dir
    sc_dir = os.path.join(output_dir, "self_consistency")
    os.makedirs(sc_dir, exist_ok=True)
    ref_copy = os.path.join(sc_dir, os.path.basename(pdb_path))
    if not os.path.exists(ref_copy):
        shutil.copy(pdb_path, ref_copy)

    # Run ProteinMPNN (use the current interpreter so env stays consistent).
    parsed_path = os.path.join(sc_dir, "parsed_pdbs.jsonl")
    subprocess.run([
        sys.executable, f"{pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
        f"--input_path={sc_dir}", f"--output_path={parsed_path}",
    ], check=True, capture_output=True)

    pmpnn_args = [
        sys.executable, f"{pmpnn_dir}/protein_mpnn_run.py",
        "--out_folder", sc_dir,
        "--jsonl_path", parsed_path,
        "--num_seq_per_target", str(seq_per_sample),
        "--sampling_temp", str(sampling_temp),
        "--seed", str(pmpnn_seed),
        "--batch_size", "1",
        "--device", str(gpu_id),
    ]
    if tied_positions_path:
        pmpnn_args.extend(["--tied_positions_jsonl", tied_positions_path])

    subprocess.run(pmpnn_args, check=True, capture_output=True)

    # Find FASTA output
    fasta_path = os.path.join(
        sc_dir, "seqs",
        os.path.basename(pdb_path).replace(".pdb", ".fa"),
    )
    if not os.path.exists(fasta_path):
        log.error(f"ProteinMPNN FASTA not found: {fasta_path}")
        return []

    # Fold each sequence and compute metrics
    import biotite.sequence.io.fasta as fasta_io
    from tmtools import tm_align

    fasta_seqs = fasta_io.FastaFile.read(fasta_path)

    # Parse reference CA positions
    import mdtraj as md
    ref_traj = md.load(pdb_path)
    ref_ca = ref_traj.topology.select("name CA")
    ref_pos = ref_traj.xyz[0, ref_ca] * 10.0  # nm -> A
    ref_seq = "A" * len(ref_ca)

    fold_dir = os.path.join(sc_dir, "esmfold")
    os.makedirs(fold_dir, exist_ok=True)

    # ProteinMPNN's FASTA prepends a "template" record — the native-sequence
    # echo of the input PDB — before the N designed sequences. Its header
    # lacks the `T=<temp>, sample=N` marker that every design has. Drop it:
    # it's not a design, folding it wastes a GPU call, and it pollutes the
    # per-sample CSV.
    fasta_items = [
        (h, s) for h, s in fasta_seqs.items()
        if h.startswith("T=") or ", T=" in h
    ]
    expected = seq_per_sample
    if len(fasta_items) != expected:
        log.warning(
            f"Expected {expected} designed sequences from ProteinMPNN, "
            f"got {len(fasta_items)} after filtering template record."
        )

    # ---- Folding stage ----
    # If a batched fold fn was supplied, fold everything in one call.
    # Otherwise fall back to the legacy per-sequence loop. Both code paths
    # write PDBs to the same `fold_dir/sample_<i>.pdb` layout so the
    # downstream metrics loop below is unchanged.
    if fold_fn_batch is not None:
        seqs = [s.replace("/", ":") for _, s in fasta_items]
        paths = [os.path.join(fold_dir, f"sample_{i}.pdb") for i in range(len(fasta_items))]
        try:
            fold_fn_batch(seqs, paths)
        except Exception as e:
            log.warning(f"Batched folding failed at the top level: {e}")
    else:
        for i, (header, string) in enumerate(fasta_items):
            fold_path = os.path.join(fold_dir, f"sample_{i}.pdb")
            try:
                fold_fn(string.replace("/", ":"), fold_path)
            except Exception as e:
                log.warning(f"Folding failed for seq {i}: {e}")

    # ---- Metrics stage ----
    results = []
    for i, (header, string) in enumerate(fasta_items):
        fold_path = os.path.join(fold_dir, f"sample_{i}.pdb")
        if not os.path.exists(fold_path):
            log.warning(f"No ESMFold output for seq {i}, skipping metrics")
            continue

        try:
            fold_traj = md.load(fold_path)
            fold_ca = fold_traj.topology.select("name CA")
            fold_pos = fold_traj.xyz[0, fold_ca] * 10.0

            # Both scTM and scRMSD come from the same tm_align call, so the
            # TM-score and the RMSD are over a consistent residue alignment.
            fold_seq = "A" * len(fold_ca)
            res = tm_align(ref_pos, fold_pos, ref_seq, fold_seq)
            tm_score = max(res.tm_norm_chain1, res.tm_norm_chain2)
            rmsd = float(res.rmsd)
        except Exception as e:
            log.warning(f"Metrics failed for seq {i}: {e}")
            tm_score = 0.0
            rmsd = 999.0

        results.append({
            "header": header,
            "sequence": string,
            "scTM": float(tm_score),
            "scRMSD": float(rmsd),
        })

    # Save CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(sc_dir, "sc_results.csv"), index=False)

    return results
