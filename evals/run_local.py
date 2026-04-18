"""Local evaluation CLI for symmetric protein generation outputs.

Generator-agnostic: takes a folder of .pdb predictions, runs ProteinMPNN +
ESMFold self-consistency, clash score, and FoldSeek novelty, then writes
per-sample JSON + aggregate CSVs.

Usage:
    conda activate sym_prot_eval
    python -m evals.run_local \\
        --pdb-dir /path/to/predictions \\
        --output-dir ./eval_outputs/run_$(date +%F) \\
        --foldseek-db /xuanwu-tank/west/gaorory/foldseek_db/pdb
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PMPNN = os.path.join(REPO_ROOT, "models", "framediff", "ProteinMPNN")
DEFAULT_FOLDSEEK_DB = "/xuanwu-tank/west/gaorory/foldseek_db/pdb"


def find_pdb_files(pdb_dir: str, pdb_glob: str) -> list:
    """Return every file matching `pdb_glob` under `pdb_dir`.

    No implicit filtering — the glob is authoritative. Users pass a precise
    pattern (e.g. `c8_*.pdb`, `sample_*.pdb`, `**/final.pdb`) to control what
    counts as a sample.
    """
    return sorted(p for p in Path(pdb_dir).glob(pdb_glob) if p.is_file())


def _sample_id_from_path(pdb_path: Path, pdb_dir: Path) -> str:
    """Flatten a PDB's path-relative-to-pdb_dir into a unique, filesystem-safe key.

    Flat layouts (RFdiffusion): `c8_1.pdb` under pdb_dir → `c8_1`.
    Nested layouts (FrameDiff): `length_150/sample_0/sample_1.pdb` → `length_150__sample_0__sample_1`.
    """
    try:
        rel = pdb_path.relative_to(pdb_dir)
    except ValueError:
        rel = Path(pdb_path.name)
    return str(rel.with_suffix("")).replace("/", "__").replace("\\", "__")


def evaluate_one(
    pdb_path: Path,
    sample_dir: str,
    pmpnn_dir: str,
    foldseek_db: str,
    fold_fn,
    num_seqs: int,
    sampling_temp: float,
    tied_positions_mode: str,
    designability_tm: float,
    designability_rmsd: float,
    skip_clash: bool,
    skip_sc: bool,
    skip_foldseek: bool,
    device: str,
    sample_id: str,
    subunit_length: int | None = None,
    fold_neighbours_k: int | None = None,
    fold_fn_batch=None,
) -> dict:
    """Run clash + self-consistency + FoldSeek on a single PDB.

    If `fold_neighbours_k` is set and smaller than the complex's chain count,
    the reference chain (A) + its k-1 nearest neighbours (by centroid distance)
    are cropped into a subcomplex and used as the input to ProteinMPNN /
    ESMFold self-consistency. Clash and FoldSeek still run on the full
    assembly. Designability (scTM/scRMSD thresholds) is therefore measured on
    the subcomplex — interpret accordingly.
    """
    from evals.evaluation import (
        compute_clash_score,
        run_foldseek,
        run_foldseek_multimer,
        run_self_consistency_generic,
    )
    from evals.tied_positions import (
        chain_info,
        extract_first_chain,
        infer_tied_positions,
    )

    os.makedirs(sample_dir, exist_ok=True)
    # Two distinct names:
    #   sample_id — globally unique per (pdb_dir, pdb) — used for output dirs, JSON keys, aggregation.
    #   pdb_stem  — the raw filename stem — needed by ProteinMPNN's parse_multiple_chains.py,
    #               which sets each record's `name` from the basename of the copied PDB.
    pdb_stem = pdb_path.stem

    # Optional chain-splitting for models that emit symmetric assemblies as
    # one long chain (FrameDiff, Foldflow). We write the split version into
    # the sample dir and point the rest of the pipeline at it so that the
    # input recorded in sample_results.json still references the original.
    working_pdb = pdb_path
    split_pdb_path = None
    if subunit_length is not None:
        from evals.split_chains import split_pdb_into_chains

        split_pdb_path = os.path.join(sample_dir, f"{pdb_stem}_split.pdb")
        try:
            split_pdb_into_chains(str(pdb_path), split_pdb_path, subunit_length)
            working_pdb = Path(split_pdb_path)
        except ValueError as e:
            log.warning(
                f"Chain split skipped for {pdb_path.name}: {e}. "
                "Proceeding with the original (un-split) PDB."
            )
            split_pdb_path = None

    info = chain_info(str(working_pdb))

    # --- Clash ---
    clash_score = None
    if not skip_clash:
        clash_score = compute_clash_score(str(working_pdb))

    # --- Subunit path ---
    # Monomer input: the whole PDB *is* the subunit.
    # Multimer input: extract chain A as the canonical subunit.
    if info["num_chains"] > 1:
        subunit_path = os.path.join(sample_dir, "subunit.pdb")
        extract_first_chain(str(working_pdb), subunit_path)
    else:
        subunit_path = str(working_pdb)

    # --- Neighbour-only cropping (optional) ---
    # For large symmetric complexes, each subunit only really "sees" its
    # spatial neighbours, so folding a k-chain subcomplex is a reasonable
    # proxy for full-complex designability at a fraction of the ESMFold cost
    # (≈ quadratic in sequence length). When enabled and tractable, the
    # cropped PDB becomes the input to tied_positions + ProteinMPNN + ESMFold.
    # Clash and FoldSeek intentionally stay on the full assembly.
    neighbours_info = None
    sc_pdb = working_pdb
    if fold_neighbours_k is not None and info["num_chains"] <= fold_neighbours_k:
        # Silent no-op is a nasty footgun: the user explicitly asked for
        # cropping, usually to avoid ESMFold OOM on a big complex. If we pass
        # through the full PDB they'll OOM and not know why. Common cause: a
        # single-chain Framediff/FoldFlow input that needed --subunit-length.
        log.warning(
            f"--fold-neighbours-k={fold_neighbours_k} has no effect on "
            f"{pdb_path.name} (num_chains={info['num_chains']}). "
            "If this is a Framediff/FoldFlow single-chain assembly, pass "
            "--subunit-length so the chain is split before cropping."
        )
    if fold_neighbours_k is not None and info["num_chains"] > fold_neighbours_k:
        from evals.neighbours import crop_to_neighbours

        neigh_path = os.path.join(
            sample_dir, f"{pdb_stem}_k{fold_neighbours_k}.pdb",
        )
        crop_result = crop_to_neighbours(
            str(working_pdb), neigh_path, k=fold_neighbours_k, reference="A",
        )
        sc_pdb = Path(neigh_path)
        neighbours_info = {**crop_result, "pdb_path": neigh_path}
        log.info(
            f"Neighbour-only SC: {info['num_chains']} chains → "
            f"{crop_result['num_kept']} "
            f"(kept {crop_result['kept_chains']} at dist "
            f"{[round(d,1) for d in crop_result['distances']]} Å)"
        )

    sc_info = chain_info(str(sc_pdb)) if sc_pdb != working_pdb else info

    # --- Tied positions ---
    tied_jsonl = None
    sc_dir = os.path.join(sample_dir, "self_consistency")
    os.makedirs(sc_dir, exist_ok=True)
    if tied_positions_mode == "auto":
        if sc_info["is_homo_oligomer"]:
            tied_jsonl = os.path.join(sc_dir, "tied_positions.jsonl")
            infer_tied_positions(
                str(sc_pdb), tied_jsonl, pdb_name=Path(sc_pdb).stem,
            )
    elif tied_positions_mode == "none":
        tied_jsonl = None
    else:
        # Explicit path
        if not os.path.exists(tied_positions_mode):
            raise FileNotFoundError(
                f"--tied-positions path not found: {tied_positions_mode}"
            )
        tied_jsonl = tied_positions_mode

    # --- Self-consistency ---
    sc_data = []
    if not skip_sc:
        gpu_id = 0
        if device.startswith("cuda") and ":" in device:
            gpu_id = int(device.split(":")[1])
        try:
            sc_data = run_self_consistency_generic(
                pdb_path=str(sc_pdb),
                output_dir=sample_dir,
                pmpnn_dir=pmpnn_dir,
                seq_per_sample=num_seqs,
                tied_positions_path=tied_jsonl,
                fold_fn=fold_fn,
                fold_fn_batch=fold_fn_batch,
                gpu_id=gpu_id,
                sampling_temp=sampling_temp,
            )
        except Exception as e:
            log.exception(f"Self-consistency failed for {pdb_path.name}: {e}")
            sc_data = []

    # --- Designability gate ---
    best_scTM = max((r["scTM"] for r in sc_data), default=0.0)
    best_scRMSD = min((r["scRMSD"] for r in sc_data), default=999.0)
    is_designable = (best_scTM > designability_tm) or (best_scRMSD < designability_rmsd)

    # --- FoldSeek novelty (only if designable) ---
    pdb_tm_complex = None
    pdb_tm_subunit = None
    if is_designable and not skip_foldseek:
        fs_dir = os.path.join(sample_dir, "foldseek")
        os.makedirs(fs_dir, exist_ok=True)
        # Complex-level multimer search: only defined for multi-chain assemblies.
        if info["num_chains"] > 1:
            pdb_tm_complex = run_foldseek_multimer(
                str(working_pdb), fs_dir, label="complex", db_path=foldseek_db,
            )
        # Subunit-level monomer search: always runs — for a monomer input the
        # whole PDB is the subunit.
        if subunit_path and os.path.exists(subunit_path):
            pdb_tm_subunit = run_foldseek(
                subunit_path, fs_dir, label="subunit", db_path=foldseek_db,
            )

    # --- Record ---
    sample_result = {
        "sample_id": sample_id,
        "pdb_path": str(pdb_path.resolve()),
        "pdb_stem": pdb_stem,
        "split_pdb_path": split_pdb_path,
        "subunit_length": subunit_length,
        "num_chains": info["num_chains"],
        "chain_lengths": info["chain_lengths"],
        "is_homo_oligomer": info["is_homo_oligomer"],
        "fold_neighbours_k": fold_neighbours_k,
        "neighbours_pdb_path": neighbours_info["pdb_path"] if neighbours_info else None,
        "neighbours_kept_chains": neighbours_info["kept_chains"] if neighbours_info else None,
        "neighbours_distances": neighbours_info["distances"] if neighbours_info else None,
        "tied_positions_used": tied_jsonl,
        "clash_score": clash_score,
        "pdbTM_complex": pdb_tm_complex,
        "pdbTM_subunit": pdb_tm_subunit,
        "is_designable": is_designable,
        "best_scTM": best_scTM if sc_data else None,
        "best_scRMSD": best_scRMSD if sc_data else None,
        "sc_results": sc_data,
    }
    with open(os.path.join(sample_dir, "sample_results.json"), "w") as f:
        json.dump(sample_result, f, indent=2)
    return sample_result


def rerun_foldseek_only(output_dir: str, foldseek_db: str) -> int:
    """Re-run FoldSeek for every already-designable sample under output_dir.

    Loads each per_sample/<stem>/sample_results.json, and if the sample was
    marked designable, deletes the old foldseek/ subdir and re-queries the
    database, updating `pdbTM_complex` and `pdbTM_subunit` in place.

    Returns the number of samples refreshed.
    """
    from evals.evaluation import run_foldseek, run_foldseek_multimer
    from pathlib import Path as _P

    per_sample_dir = _P(output_dir) / "per_sample"
    refreshed = 0
    for sample_dir in sorted(per_sample_dir.iterdir()):
        result_file = sample_dir / "sample_results.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            result = json.load(f)
        if not result.get("is_designable"):
            continue

        fs_dir = sample_dir / "foldseek"
        if fs_dir.exists():
            shutil.rmtree(fs_dir)
        fs_dir.mkdir(parents=True, exist_ok=True)

        # If the original run used --subunit-length, the multi-chain PDB lives
        # at split_pdb_path, not pdb_path (which is the single-chain original).
        effective_pdb = result.get("split_pdb_path") or result["pdb_path"]
        num_chains = result.get("num_chains", 1)
        log.info(f"Re-running FoldSeek on {sample_dir.name}")

        # Complex search only for multi-chain assemblies.
        if num_chains > 1:
            result["pdbTM_complex"] = run_foldseek_multimer(
                effective_pdb, str(fs_dir), label="complex", db_path=foldseek_db,
            )
        else:
            result["pdbTM_complex"] = None

        # Subunit: first chain for multimers, whole PDB for monomers.
        subunit = sample_dir / "subunit.pdb"
        subunit_path = str(subunit) if subunit.exists() else effective_pdb
        result["pdbTM_subunit"] = run_foldseek(
            subunit_path, str(fs_dir), label="subunit", db_path=foldseek_db,
        )

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        refreshed += 1
    return refreshed


def compute_diversity_for_run(output_dir: str) -> dict | None:
    """Compute mean pairwise TM-score across designable samples in a finished run.

    Reads every per_sample/*/sample_results.json, filters to is_designable, and
    calls compute_pairwise_diversity on the effective PDB for each (split PDB
    when --subunit-length was used at eval time, else the raw pdb_path). Writes
    the result to <output_dir>/diversity.json; aggregate() will merge that into
    summary.json on the next aggregation pass.
    """
    from evals.evaluation import compute_pairwise_diversity

    per_sample_dir = Path(output_dir) / "per_sample"
    designable_pdbs = []
    for sample_dir in sorted(per_sample_dir.iterdir()):
        result_file = sample_dir / "sample_results.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            result = json.load(f)
        if not result.get("is_designable"):
            continue
        effective_pdb = result.get("split_pdb_path") or result.get("pdb_path")
        if effective_pdb and os.path.exists(effective_pdb):
            designable_pdbs.append(effective_pdb)
        else:
            log.warning(
                f"Diversity: designable sample {sample_dir.name} has no readable "
                f"PDB at {effective_pdb!r}, skipping"
            )

    log.info(f"Computing diversity over {len(designable_pdbs)} designable samples")
    diversity = compute_pairwise_diversity(designable_pdbs)

    diversity_path = os.path.join(output_dir, "diversity.json")
    payload = diversity if diversity is not None else {
        "mean_pairwise_tm": None,
        "num_pdbs": len(designable_pdbs),
        "num_pairs": 0,
    }
    with open(diversity_path, "w") as f:
        json.dump(payload, f, indent=2)
    return diversity


def aggregate(output_dir: str) -> dict:
    """Aggregate per_sample/*/sample_results.json into CSVs."""
    import pandas as pd

    per_sample_dir = os.path.join(output_dir, "per_sample")
    rows = []
    for result_file in sorted(
        glob.glob(os.path.join(per_sample_dir, "**", "sample_results.json"), recursive=True)
    ):
        with open(result_file) as f:
            result = json.load(f)
        sc_data = result.get("sc_results") or []
        # Prefer sample_id (unique across nested layouts); fall back to pdb_stem
        # for pre-refactor result files that don't carry sample_id.
        sample_key = result.get("sample_id") or result["pdb_stem"]
        base = {
            "sample_id": sample_key,
            "pdb_stem": result["pdb_stem"],
            "pdb_path": result["pdb_path"],
            "num_chains": result["num_chains"],
            "is_homo_oligomer": result["is_homo_oligomer"],
            "fold_neighbours_k": result.get("fold_neighbours_k"),
            "clash_score": result.get("clash_score"),
            "pdbTM_complex": result.get("pdbTM_complex"),
            "pdbTM_subunit": result.get("pdbTM_subunit"),
            "is_designable": result.get("is_designable", False),
            "best_scTM": result.get("best_scTM"),
            "best_scRMSD": result.get("best_scRMSD"),
        }
        if sc_data:
            for seq in sc_data:
                rows.append({
                    **base,
                    "seq_header": seq.get("header", ""),
                    "scTM": seq.get("scTM"),
                    "scRMSD": seq.get("scRMSD"),
                })
        else:
            rows.append({**base, "seq_header": "", "scTM": None, "scRMSD": None})

    if not rows:
        log.warning("No per-sample results to aggregate.")
        return {"num_samples": 0, "num_rows": 0}

    df = pd.DataFrame(rows)
    all_path = os.path.join(output_dir, "all_results.csv")
    df.to_csv(all_path, index=False)

    # Per-PDB summary (one row per sample, not per sequence).
    by_sample = df.drop_duplicates(subset=["sample_id"]).copy()
    designable = by_sample[by_sample["is_designable"] == True]
    summary = {
        "num_samples": int(len(by_sample)),
        "num_rows": int(len(df)),
        "num_designable": int(len(designable)),
        "frac_designable": float(by_sample["is_designable"].mean()),
        "mean_clash": float(by_sample["clash_score"].mean()) if by_sample["clash_score"].notna().any() else None,
        "mean_best_scTM": float(by_sample["best_scTM"].mean()) if by_sample["best_scTM"].notna().any() else None,
        "mean_best_scRMSD": float(by_sample["best_scRMSD"].mean()) if by_sample["best_scRMSD"].notna().any() else None,
        "mean_best_scTM_designable": (
            float(designable["best_scTM"].mean())
            if len(designable) and designable["best_scTM"].notna().any() else None
        ),
        "mean_best_scRMSD_designable": (
            float(designable["best_scRMSD"].mean())
            if len(designable) and designable["best_scRMSD"].notna().any() else None
        ),
        "mean_pdbTM_complex": float(by_sample["pdbTM_complex"].mean()) if by_sample["pdbTM_complex"].notna().any() else None,
        "mean_pdbTM_subunit": float(by_sample["pdbTM_subunit"].mean()) if by_sample["pdbTM_subunit"].notna().any() else None,
    }

    diversity_path = os.path.join(output_dir, "diversity.json")
    if os.path.exists(diversity_path):
        with open(diversity_path) as f:
            div = json.load(f)
        summary["diversity_mean_pairwise_tm"] = div.get("mean_pairwise_tm")
        summary["diversity_min_pairwise_tm"] = div.get("min_pairwise_tm")
        summary["diversity_max_pairwise_tm"] = div.get("max_pairwise_tm")
        summary["diversity_num_pdbs"] = div.get("num_pdbs")
        summary["diversity_num_pairs"] = div.get("num_pairs")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also dump a one-row CSV for quick comparison across runs.
    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    log.info(f"Aggregation: wrote {all_path} and {summary_path}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdb-dir", required=True,
                        help="Folder containing .pdb predictions.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write per-sample results and CSVs.")
    parser.add_argument("--pdb-glob", default="*.pdb",
                        help='Glob for input PDBs, relative to --pdb-dir '
                             '(default: "*.pdb" = top-level only, no recursion). '
                             'Examples: "**/*.pdb" (recurse all), '
                             '"c8_*.pdb" (RFdiffusion samples, skipping traj/), '
                             '"sample_*.pdb" (FrameDiff samples).')
    parser.add_argument("--num-seqs", type=int, default=8,
                        help="ProteinMPNN sequences per backbone.")
    parser.add_argument("--sampling-temp", type=float, default=0.1,
                        help="ProteinMPNN sampling temperature.")
    parser.add_argument("--pmpnn-dir", default=DEFAULT_PMPNN,
                        help="Path to ProteinMPNN repo.")
    parser.add_argument("--foldseek-db", default=DEFAULT_FOLDSEEK_DB,
                        help="Path to FoldSeek database (prefix).")
    parser.add_argument("--tied-positions", default="auto",
                        help='"auto" (default), "none", or path to JSONL.')
    parser.add_argument("--subunit-length", type=int, default=None,
                        help="If set, pre-split each input PDB into chains of "
                             "this many residues (needed for FrameDiff / "
                             "Foldflow monomer-represented assemblies). Total "
                             "residues must be divisible by this value.")
    parser.add_argument("--fold-neighbours-k", type=int, default=None,
                        help="If set, crop each multimer to chain A plus its "
                             "k-1 nearest chains (by centroid distance) before "
                             "ProteinMPNN / ESMFold. Clash and FoldSeek still "
                             "run on the full assembly. Designability (scTM / "
                             "scRMSD) is measured on the subcomplex — a proxy "
                             "for full-complex designability that holds when "
                             "k is large enough to cover the neighbour "
                             "interface. No-op if k >= num_chains.")
    parser.add_argument("--fold-batch-size", type=int, default=1,
                        help="ESMFold batch size for self-consistency. "
                             "1 (default) = legacy per-sequence path. "
                             ">1 batches sequences through model.infer_pdbs "
                             "for 2-3x faster SC at some GPU memory cost. "
                             "OOM-resilient: chunks that OOM fall back "
                             "to per-sequence folding.")
    parser.add_argument("--designability-tm", type=float, default=0.5)
    parser.add_argument("--designability-rmsd", type=float, default=2.0)
    parser.add_argument("--skip-clash", action="store_true")
    parser.add_argument("--skip-sc", action="store_true")
    parser.add_argument("--skip-foldseek", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip per-sample work; just re-aggregate CSVs.")
    parser.add_argument("--rerun-foldseek", action="store_true",
                        help="Re-run FoldSeek for already-designable samples "
                             "in --output-dir (no GPU / no ProteinMPNN). Use "
                             "after changing the FoldSeek metric or DB.")
    parser.add_argument("--recompute-diversity", action="store_true",
                        help="Compute pairwise-TM diversity over the already-"
                             "designable samples in --output-dir and append to "
                             "summary.json (no GPU / no ProteinMPNN / no "
                             "FoldSeek). Use to backfill diversity on a run "
                             "that was evaluated before this metric existed.")
    parser.add_argument("--skip-diversity", action="store_true",
                        help="Skip the end-of-run diversity computation in the "
                             "normal (full-eval) flow.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = os.path.abspath(args.output_dir)
    per_sample_dir = os.path.join(output_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    if args.aggregate_only:
        summary = aggregate(output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.rerun_foldseek:
        n = rerun_foldseek_only(output_dir, args.foldseek_db)
        log.info(f"Refreshed FoldSeek for {n} designable samples")
        summary = aggregate(output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    if args.recompute_diversity:
        compute_diversity_for_run(output_dir)
        summary = aggregate(output_dir)
        print(json.dumps(summary, indent=2))
        return 0

    pdb_files = find_pdb_files(args.pdb_dir, args.pdb_glob)
    if not pdb_files:
        log.error(f"No PDBs found under {args.pdb_dir} with glob {args.pdb_glob!r}")
        return 1
    log.info(f"Found {len(pdb_files)} PDB files (glob={args.pdb_glob!r})")

    # Lazy-load ESMFold (only if SC enabled). Both `fold_fn` and
    # `fold_fn_batch` are defined here so the `evaluate_one` call below
    # can always pass them, regardless of branch taken.
    fold_fn = None
    fold_fn_batch = None
    if not args.skip_sc:
        from evals.fold import fold_fn_factory, fold_fn_batch_factory
        log.info(f"Loading ESMFold on {args.device}")
        fold_fn = fold_fn_factory(device=args.device)
        # Only build the batched fold_fn when the user opts in (>1).
        # batch_size=1 keeps the legacy per-sequence path bit-identical.
        if args.fold_batch_size > 1:
            fold_fn_batch = fold_fn_batch_factory(
                device=args.device, batch_size=args.fold_batch_size,
            )
            log.info(f"ESMFold batching enabled: batch_size={args.fold_batch_size}")

    # Warn once if FoldSeek DB is missing rather than failing every sample.
    if not args.skip_foldseek:
        has_db = any(
            os.path.exists(args.foldseek_db + suf)
            for suf in ("", ".dbtype", ".index")
        )
        if not has_db:
            log.warning(
                f"FoldSeek DB {args.foldseek_db} not found — runs will "
                "record pdbTM=None. Pass --skip-foldseek to silence."
            )

    pdb_dir_path = Path(args.pdb_dir).resolve()
    n_done = 0
    n_skipped = 0
    for pdb in pdb_files:
        sample_id = _sample_id_from_path(pdb.resolve(), pdb_dir_path)
        sample_dir = os.path.join(per_sample_dir, sample_id)
        result_file = os.path.join(sample_dir, "sample_results.json")
        if os.path.exists(result_file):
            log.info(f"Skipping {sample_id} (already done)")
            n_skipped += 1
            continue

        log.info(f"[{n_done + n_skipped + 1}/{len(pdb_files)}] {sample_id}")
        try:
            evaluate_one(
                pdb_path=pdb,
                sample_dir=sample_dir,
                sample_id=sample_id,
                subunit_length=args.subunit_length,
                fold_neighbours_k=args.fold_neighbours_k,
                pmpnn_dir=args.pmpnn_dir,
                foldseek_db=args.foldseek_db,
                fold_fn=fold_fn,
                fold_fn_batch=fold_fn_batch,
                num_seqs=args.num_seqs,
                sampling_temp=args.sampling_temp,
                tied_positions_mode=args.tied_positions,
                designability_tm=args.designability_tm,
                designability_rmsd=args.designability_rmsd,
                skip_clash=args.skip_clash,
                skip_sc=args.skip_sc,
                skip_foldseek=args.skip_foldseek,
                device=args.device,
            )
            n_done += 1
        except Exception as e:
            log.exception(f"Failed on {pdb.name}: {e}")

    log.info(f"Completed {n_done}, skipped {n_skipped} of {len(pdb_files)}")

    if not args.skip_diversity:
        compute_diversity_for_run(output_dir)

    summary = aggregate(output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
