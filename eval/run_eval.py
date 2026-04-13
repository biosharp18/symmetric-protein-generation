#!/usr/bin/env python3
"""
Evaluate protein generation samples.

Usage:
    python eval/run_eval.py <sample_dir> [<sample_dir2> ...] [--output results.json] [--pdb-glob "*.pdb"]

Examples:
    # Single directory
    python eval/run_eval.py ../Framediff/inference_outputs/symm_efficient_n2000_c20/length_2000/

    # Multiple directories (compare models)
    python eval/run_eval.py \
        ../rfdiffusion/samples/length_150_original/ \
        ../Framediff/inference_outputs/symm_efficient_n2000_c20/length_2000/ \
        --output comparison.json

    # Custom PDB glob pattern
    python eval/run_eval.py ./my_samples/ --pdb-glob "sample_*.pdb"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from metrics import per_sample_metrics, extract_ca_coords, diversity_score, self_consistency_metrics


def find_pdb_files(sample_dir: str, pdb_glob: str = "**/*.pdb") -> list[Path]:
    """Recursively find PDB files, excluding trajectory files."""
    all_pdbs = sorted(Path(sample_dir).glob(pdb_glob))
    # Filter out trajectory files (bb_traj_*, x0_traj_*)
    filtered = [p for p in all_pdbs if not any(
        prefix in p.name for prefix in ("bb_traj", "x0_traj")
    )]
    return filtered


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Compute aggregate statistics over per-sample metrics."""
    if not per_sample:
        return {}
    numeric_keys = [
        k for k in per_sample[0]
        if isinstance(per_sample[0][k], (int, float)) and k != "num_residues"
    ]
    agg = {}
    for key in numeric_keys:
        values = [s[key] for s in per_sample]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    # Also report residue count stats
    lengths = [s["num_residues"] for s in per_sample]
    agg["num_residues"] = {
        "mean": float(np.mean(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }
    agg["num_samples"] = len(per_sample)
    return agg


def evaluate_directory(
    sample_dir: str,
    pdb_glob: str,
    skip_diversity: bool = False,
    self_consistency: bool = False,
    folding_model=None,
    pmpnn_dir: str = None,
    num_seqs: int = 8,
    sampling_temp: float = 0.1,
) -> dict:
    """Run full evaluation on a directory of PDB samples."""
    pdb_files = find_pdb_files(sample_dir, pdb_glob)
    if not pdb_files:
        print(f"  WARNING: No PDB files found in {sample_dir} (glob: {pdb_glob})")
        return {"error": "no PDB files found", "sample_dir": sample_dir}

    print(f"  Found {len(pdb_files)} PDB files")

    # Per-sample metrics
    print("  Computing per-sample metrics...")
    per_sample = []
    for i, pdb_path in enumerate(pdb_files):
        try:
            m = per_sample_metrics(str(pdb_path))
            per_sample.append(m)
        except Exception as e:
            print(f"    WARN: Failed on {pdb_path.name}: {e}")
        if (i + 1) % 10 == 0 or i == len(pdb_files) - 1:
            print(f"    [{i+1}/{len(pdb_files)}]")

    agg = aggregate_metrics(per_sample)

    # Diversity (pairwise TM-score)
    div = {}
    if skip_diversity:
        print("  Skipping diversity computation (--skip-diversity).")
    else:
        print("  Computing diversity (pairwise TM-scores)...")
        t0 = time.time()
        ca_positions = []
        for pdb_path in pdb_files:
            try:
                ca_positions.append(extract_ca_coords(str(pdb_path)))
            except Exception:
                print(f"    WARN: Failed to extract CA coords from {pdb_path.name}, skipping.")
        div = diversity_score(ca_positions)
        dt = time.time() - t0
        print(f"    Diversity TM-score mean: {div['diversity_tm_mean']:.4f} "
              f"({div['num_pairs']} pairs, {dt:.1f}s)")

    # Self-consistency (ProteinMPNN → ESMFold → scTM/scRMSD)
    sc_results = {}
    if self_consistency:
        print("  Computing self-consistency (ProteinMPNN + ESMFold)...")
        sc_output_base = Path(sample_dir) / "self_consistency"
        sc_output_base.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        sc_per_sample = []
        for i, pdb_path in enumerate(pdb_files):
            sc_output_dir = str(sc_output_base / pdb_path.stem)
            try:
                sc = self_consistency_metrics(
                    str(pdb_path),
                    folding_model=folding_model,
                    pmpnn_dir=pmpnn_dir,
                    sc_output_dir=sc_output_dir,
                    num_seqs=num_seqs,
                    sampling_temp=sampling_temp,
                )
                sc_per_sample.append(sc)
                # Attach to the corresponding per_sample entry
                if i < len(per_sample):
                    per_sample[i].update(sc)
                print(f"    [{i+1}/{len(pdb_files)}] {pdb_path.name}: "
                      f"scTM={sc['sc_tm']:.3f}, scRMSD={sc['sc_rmsd']:.2f}")
            except Exception as e:
                print(f"    WARN: SC failed on {pdb_path.name}: {e}")
        dt = time.time() - t0

        # Aggregate SC metrics
        if sc_per_sample:
            valid_tms = [s["sc_tm"] for s in sc_per_sample if not np.isnan(s["sc_tm"])]
            valid_rmsds = [s["sc_rmsd"] for s in sc_per_sample if not np.isnan(s["sc_rmsd"])]
            if valid_tms:
                designable = [t for t in valid_tms if t > 0.5]
                sc_results = {
                    "sc_tm_mean": float(np.mean(valid_tms)),
                    "sc_tm_std": float(np.std(valid_tms)),
                    "sc_rmsd_mean": float(np.mean(valid_rmsds)),
                    "sc_rmsd_std": float(np.std(valid_rmsds)),
                    "designability_rate": len(designable) / len(valid_tms),
                    "num_designable_TM": len(designable),
                    "num_designable_RMSD": len([r for r in valid_rmsds if r < 2.0]),
                    "num_evaluated": len(valid_tms),
                }
                print(f"    scTM: {sc_results['sc_tm_mean']:.4f} ± {sc_results['sc_tm_std']:.4f}")
                print(f"    scRMSD: {sc_results['sc_rmsd_mean']:.2f} ± {sc_results['sc_rmsd_std']:.2f}")
                print(f"    Designability (scTM>0.5): {sc_results['designability_rate']:.1%} "
                      f"({sc_results['num_designable_TM']}/{sc_results['num_evaluated']})")
                print(f"    Designability (scRMSD<2Å): {sc_results['num_designable_RMSD'] / len(valid_rmsds):.1%} "
                      f"({sc_results['num_designable_RMSD']}/{len(valid_rmsds)})")
        print(f"    Self-consistency done in {dt:.1f}s")

    return {
        "sample_dir": str(Path(sample_dir).resolve()),
        "num_pdbs_found": len(pdb_files),
        "aggregate": agg,
        "diversity": div,
        "self_consistency": sc_results,
        "per_sample": per_sample,
    }


def print_summary(results: list[dict]):
    """Print a human-readable comparison table."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for r in results:
        if "error" in r:
            print(f"\n{r['sample_dir']}: {r['error']}")
            continue

        name = Path(r["sample_dir"]).name
        agg = r["aggregate"]
        div = r["diversity"]

        print(f"\n── {name} ({r['num_pdbs_found']} samples) ──")
        if div and "diversity_tm_mean" in div:
            print(f"  Diversity (mean pairwise TM):   {div['diversity_tm_mean']:.4f}  (lower = more diverse)")
        if "ca_steric_clash_percent" in agg:
            clash = agg["ca_steric_clash_percent"]
            print(f"  Steric clash %:                 {clash['mean']:.6f} ± {clash['std']:.6f}")
        if "num_ca_steric_clashes" in agg:
            nc = agg["num_ca_steric_clashes"]
            print(f"  Num CA clashes (per sample):    {nc['mean']:.1f} ± {nc['std']:.1f}")
        if "ca_ca_bond_dev" in agg:
            bd = agg["ca_ca_bond_dev"]
            print(f"  CA-CA bond deviation (Å):       {bd['mean']:.4f} ± {bd['std']:.4f}")
        if "ca_ca_valid_percent" in agg:
            bv = agg["ca_ca_valid_percent"]
            print(f"  CA-CA valid bond %:             {bv['mean']:.4f} ± {bv['std']:.4f}")
        if "helix_percent" in agg:
            h = agg["helix_percent"]
            s = agg["strand_percent"]
            c = agg["coil_percent"]
            print(f"  Secondary structure:            H={h['mean']:.2%}  E={s['mean']:.2%}  C={c['mean']:.2%}")
        if "radius_of_gyration_nm" in agg:
            rg = agg["radius_of_gyration_nm"]
            print(f"  Radius of gyration (nm):        {rg['mean']:.3f} ± {rg['std']:.3f}")

        sc = r.get("self_consistency", {})
        if sc:
            print(f"  scTM (best-of-N, mean):         {sc['sc_tm_mean']:.4f} ± {sc['sc_tm_std']:.4f}")
            print(f"  scRMSD (best-of-N, mean):       {sc['sc_rmsd_mean']:.2f} ± {sc['sc_rmsd_std']:.2f}")
            print(f"  Designability (scTM > 0.5):     {sc['designability_rate']:.1%} "
                  f"({sc['num_designable_TM']}/{sc['num_evaluated']})")
            print(f"  Designability (scRMSD < 2Å):    {sc['num_designable_RMSD'] / sc['num_evaluated']:.1%} "
                  f"({sc['num_designable_RMSD']}/{sc['num_evaluated']})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate protein generation samples (diversity + clash metrics)."
    )
    parser.add_argument(
        "sample_dirs",
        nargs="+",
        help="One or more directories containing generated PDB files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save JSON results (default: print summary only).",
    )
    parser.add_argument(
        "--pdb-glob",
        default="**/sample_*.pdb",
        help='Glob pattern for finding PDB files (default: "**/sample_*.pdb").',
    )
    parser.add_argument(
        "--skip-diversity",
        action="store_true",
        help="Skip pairwise TM-score diversity computation (slow for large proteins).",
    )
    parser.add_argument(
        "--self-consistency",
        action="store_true",
        help="Run self-consistency evaluation (ProteinMPNN + ESMFold). Requires GPU.",
    )
    parser.add_argument(
        "--pmpnn-dir",
        default=None,
        help="Path to ProteinMPNN directory (default: models/framediff/ProteinMPNN/).",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=8,
        help="Number of ProteinMPNN sequences per backbone (default: 8).",
    )
    parser.add_argument(
        "--sampling-temp",
        type=float,
        default=0.1,
        help="ProteinMPNN sampling temperature (default: 0.1).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to use for ESMFold (default: auto-select).",
    )
    args = parser.parse_args()

    # Resolve ProteinMPNN directory
    if args.pmpnn_dir is None:
        # Default: relative to this script's repo root
        repo_root = Path(__file__).resolve().parent.parent
        args.pmpnn_dir = str(repo_root / "models" / "framediff" / "ProteinMPNN")
    if args.self_consistency and not Path(args.pmpnn_dir).is_dir():
        print(f"ERROR: ProteinMPNN directory not found: {args.pmpnn_dir}")
        sys.exit(1)

    # Load ESMFold model once if needed
    folding_model = None
    if args.self_consistency:
        print("Loading ESMFold model...")
        # ESMFold depends on openfold; add FoldFlow's bundled copy to path if needed
        try:
            import openfold  # noqa: F401
        except ImportError:
            # Look for openfold in common locations
            candidates = [
                Path(args.pmpnn_dir).parent,  # models/framediff/
                Path.home() / "FoldFlow",      # ~/FoldFlow/
            ]
            for candidate in candidates:
                of_path = candidate / "openfold"
                if of_path.is_dir():
                    sys.path.insert(0, str(candidate))
                    break
        import torch
        import esm
        if args.gpu is not None:
            device = f"cuda:{args.gpu}"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        folding_model = esm.pretrained.esmfold_v1().eval().to(device)
        print(f"  ESMFold loaded on {device}")

    results = []
    for d in args.sample_dirs:
        print(f"\nEvaluating: {d}")
        r = evaluate_directory(
            d, args.pdb_glob,
            skip_diversity=args.skip_diversity,
            self_consistency=args.self_consistency,
            folding_model=folding_model,
            pmpnn_dir=args.pmpnn_dir,
            num_seqs=args.num_seqs,
            sampling_temp=args.sampling_temp,
        )
        results.append(r)

    print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
