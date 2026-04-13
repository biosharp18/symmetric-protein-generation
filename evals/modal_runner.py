"""Symmetric protein generation metrics evaluation on Modal.

Generates symmetric protein backbones using a configurable generator model,
then evaluates designability, novelty, and structural quality.

Supported generators:
  - framediff: FrameDiff with efficient/non-efficient symmetric attention
  - (extend by adding to evals/generators/)

Usage:
    # Run ALL jobs (48 combos x 10 samples each) with FrameDiff:
    modal run evals/modal_runner.py

    # Test run (1 sample, smallest length per symmetry):
    modal run evals/modal_runner.py --test-run

    # Collect results only:
    modal run evals/modal_runner.py --collect-only

    # Use a different generator:
    modal run evals/modal_runner.py --generator rfdiffusion
"""

import os
import json
import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

GPU_TYPE = "H100"
TIMEOUT_SECONDS = 14400  # 4 hours

app = modal.App("symm-protein-metrics")

# Path to the parent symmetric-protein-generation repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to FrameDiff repo (sibling or submodule)
FRAMEDIFF_DIR = os.environ.get(
    "FRAMEDIFF_DIR",
    os.path.join(REPO_ROOT, "..", "Framediff"),
)
# Resolve to absolute
FRAMEDIFF_DIR = os.path.abspath(FRAMEDIFF_DIR)

spec_file = os.path.join(FRAMEDIFF_DIR, "config", "modal_env.yml")

metrics_image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install(
        spec_file=spec_file,
        channels=["pytorch", "nvidia", "conda-forge", "defaults"],
    )
    .run_commands(
        "pip install protenix --no-deps",
        "pip install rdkit gemmi==0.6.5 modelcif==0.7 pdbeccdutils==0.8.5 "
        "scikit-learn-extra optree 'numpy==1.22.4'",
        "pip install biotite==1.0.1 --no-deps",
    )
    .env({
        "CUDA_HOME": "/opt/conda",
        "DS_BUILD_OPS": "0",
    })
    .run_commands(
        # Pre-download ESMFold weights
        "python -c \""
        "import urllib.request, os; "
        "os.makedirs('/root/.cache/torch/checkpoints', exist_ok=True); "
        "urllib.request.urlretrieve("
        "'https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt', "
        "'/root/.cache/torch/checkpoints/esmfold_3B_v1.pt')\"",
    )
    .run_commands(
        # Install FoldSeek + PDB database
        "apt-get update && apt-get install -y wget aria2 && "
        "wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz && "
        "tar xzf foldseek-linux-avx2.tar.gz && "
        "mv foldseek/bin/foldseek /usr/local/bin/ && "
        "rm -rf foldseek foldseek-linux-avx2.tar.gz",
        "mkdir -p /root/foldseek_db && "
        "foldseek databases PDB /root/foldseek_db/pdb /tmp/foldseek_tmp",
    )
    .env({"CACHE_BUST": "metrics-v2"})
    # Add the FrameDiff model repo
    .add_local_dir(
        FRAMEDIFF_DIR,
        remote_path="/root/Framediff",
        ignore=[".git", "__pycache__", ".cache", "node_modules", "eval_outputs"],
    )
    # Add the symmetric-protein-generation repo (for evals/ package)
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/symmetric-protein-generation",
        ignore=[".git", "__pycache__", "models"],
    )
)

output_volume = modal.Volume.from_name(
    "symm-protein-metrics-outputs", create_if_missing=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLES_PER_JOB = 10
SEQ_PER_SAMPLE = 8
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

def _get_generator(name: str):
    """Factory for backbone generators. Add new models here."""
    if name == "framediff":
        from evals.generators.framediff import FrameDiffGenerator
        return FrameDiffGenerator(framediff_dir="/root/Framediff")
    # elif name == "rfdiffusion":
    #     from evals.generators.rfdiffusion import RFdiffusionGenerator
    #     return RFdiffusionGenerator(...)
    else:
        raise ValueError(
            f"Unknown generator: {name}. "
            f"Available: framediff"
        )


# ---------------------------------------------------------------------------
# Remote worker
# ---------------------------------------------------------------------------

@app.function(
    image=metrics_image,
    gpu=GPU_TYPE,
    volumes={"/outputs": output_volume},
    timeout=TIMEOUT_SECONDS,
    memory=32768,
    max_containers=6,
)
def run_single_job(
    spec_dict: dict,
    generator_name: str,
    num_samples: int,
    efficient_symmetry: bool,
):
    """Run generation + evaluation for one (symmetry, length, efficient) combo."""
    import sys
    import time
    import logging
    import shutil
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/root/Framediff")
    sys.path.insert(0, "/root/symmetric-protein-generation")
    os.chdir("/root/Framediff")

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    from evals.generators.base import SymmetrySpec
    from evals.evaluation import run_foldseek, compute_clash_score

    # Reconstruct SymmetrySpec from dict
    spec = SymmetrySpec(**spec_dict)
    spec.extra["efficient_symmetry"] = efficient_symmetry

    eff_label = "efficient" if efficient_symmetry else "nonefficient"
    symm_label = spec.label

    job_dir = (
        f"/outputs/metrics/{generator_name}/"
        f"{symm_label}/length_{spec.total_length}/{eff_label}"
    )
    os.makedirs(job_dir, exist_ok=True)

    log.info(f"=== Job: {generator_name} {symm_label} "
             f"total_len={spec.total_length} {eff_label} ===")

    # Initialize generator
    generator = _get_generator(generator_name)
    generator.setup(job_dir, gpu_id=0)

    all_results = []
    for sample_i in range(num_samples):
        seed = BASE_SEED + sample_i

        sample_dir = os.path.join(job_dir, "results", f"sample_{sample_i}")
        result_file = os.path.join(sample_dir, "sample_results.json")
        if os.path.exists(result_file):
            log.info(f"  Sample {sample_i} already done, skipping")
            continue
        os.makedirs(sample_dir, exist_ok=True)

        log.info(f"  Generating sample {sample_i} (seed={seed})...")

        # --- Generate backbone ---
        try:
            gen_result = generator.generate(spec, sample_dir, seed)
        except Exception as e:
            log.error(f"  Sample {sample_i} generation failed: {e}")
            continue

        pdb_path = gen_result.pdb_path

        # --- Clash score ---
        clash_score = compute_clash_score(pdb_path)

        # --- Self-consistency (ProteinMPNN + ESMFold) ---
        # Use FrameDiff's built-in self-consistency if available
        sc_data = []
        if generator_name == "framediff":
            sc_output_dir = os.path.join(sample_dir, "self_consistency")
            os.makedirs(sc_output_dir, exist_ok=True)
            shutil.copy(
                pdb_path,
                os.path.join(sc_output_dir, os.path.basename(pdb_path)),
            )
            try:
                sampler = generator.get_self_consistency_runner()
                sampler.run_self_consistency(
                    sc_output_dir,
                    pdb_path,
                    motif_mask=None,
                    tied_positions_path=gen_result.tied_positions_path,
                )
                # Read results
                sc_csv = os.path.join(sc_output_dir, "sc_results.csv")
                if os.path.exists(sc_csv):
                    df = pd.read_csv(sc_csv)
                    for _, row in df.iterrows():
                        sc_data.append({
                            "header": row.get("header", ""),
                            "sequence": row.get("sequence", ""),
                            "scTM": float(row["tm_score"]),
                            "scRMSD": float(row["rmsd"]),
                        })
            except Exception as e:
                log.error(f"  Self-consistency failed: {e}")

        # --- FoldSeek (only if designable) ---
        best_scTM = max((r["scTM"] for r in sc_data), default=0)
        best_scRMSD = min((r["scRMSD"] for r in sc_data), default=999)
        is_designable = (best_scTM > 0.5) or (best_scRMSD < 2.0)

        pdb_tm_complex = None
        pdb_tm_subunit = None
        if is_designable:
            log.info(f"  Sample {sample_i} is designable — running FoldSeek...")
            pdb_tm_complex = run_foldseek(pdb_path, sample_dir, label="complex")
            if gen_result.subunit_pdb_path and os.path.exists(
                gen_result.subunit_pdb_path
            ):
                pdb_tm_subunit = run_foldseek(
                    gen_result.subunit_pdb_path, sample_dir, label="subunit")
        else:
            log.info(f"  Sample {sample_i} not designable — skipping FoldSeek")

        # --- Save results ---
        sample_result = {
            "generator": generator_name,
            "symmetry": symm_label,
            "total_length": spec.total_length,
            "subunit_length": spec.subunit_length,
            "efficient_symmetry": efficient_symmetry,
            "sample_index": sample_i,
            "seed": seed,
            "clash_score": clash_score,
            "pdbTM_complex": pdb_tm_complex,
            "pdbTM_subunit": pdb_tm_subunit,
            "is_designable": is_designable,
            "sc_results": sc_data,
            "pdb_path": pdb_path,
        }
        with open(result_file, "w") as f:
            json.dump(sample_result, f, indent=2)
        all_results.append(sample_result)

        clash_str = f"{clash_score:.2f}" if clash_score is not None else "N/A"
        cx_str = f"{pdb_tm_complex:.3f}" if pdb_tm_complex is not None else "N/A"
        su_str = f"{pdb_tm_subunit:.3f}" if pdb_tm_subunit is not None else "N/A"
        sc_str = f", best_scTM={best_scTM:.3f}" if sc_data else ""
        log.info(
            f"  Sample {sample_i} done: "
            f"clash={clash_str}, pdbTM_cx={cx_str}, pdbTM_su={su_str}{sc_str}"
        )

    # Save aggregated job results
    with open(os.path.join(job_dir, "job_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    output_volume.commit()
    return {
        "status": "success",
        "generator": generator_name,
        "symmetry": symm_label,
        "total_length": spec.total_length,
        "efficient": efficient_symmetry,
        "num_samples": len(all_results),
    }


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@app.function(
    image=metrics_image,
    volumes={"/outputs": output_volume},
    timeout=600,
)
def collect_results():
    """Collect all results into CSVs."""
    import pandas as pd
    import glob

    metrics_dir = "/outputs/metrics"
    rows = []

    for result_file in sorted(
        glob.glob(os.path.join(metrics_dir, "**/sample_results.json"), recursive=True)
    ):
        try:
            with open(result_file) as f:
                result = json.load(f)
            _add_rows(rows, result)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")

    if not rows:
        print("No results found!")
        return {"status": "empty"}

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(metrics_dir, "all_results.csv"), index=False)

    summary = df.groupby(
        ["generator", "symmetry", "total_length", "efficient_symmetry"]
    ).agg(
        n_samples=("sample_index", "nunique"),
        mean_clash=("clash_score", "mean"),
        mean_pdbTM_complex=("pdbTM_complex", "mean"),
        mean_pdbTM_subunit=("pdbTM_subunit", "mean"),
        mean_scTM=("best_scTM", "mean"),
        mean_scRMSD=("best_scRMSD", "mean"),
        frac_designable_tm=("best_scTM", lambda x: (x > 0.5).mean()),
        frac_designable_rmsd=("best_scRMSD", lambda x: (x < 2.0).mean()),
    ).reset_index()

    summary.to_csv(os.path.join(metrics_dir, "summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("METRICS SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    output_volume.commit()
    return {"status": "success", "total_rows": len(df)}


def _add_rows(rows, result):
    """Flatten a sample result dict into rows for the DataFrame."""
    sc_data = result.get("sc_results", [])
    best_scTM = max((r["scTM"] for r in sc_data), default=None)
    best_scRMSD = min((r["scRMSD"] for r in sc_data), default=None)

    base = {
        "generator": result.get("generator", "unknown"),
        "symmetry": result["symmetry"],
        "total_length": result["total_length"],
        "subunit_length": result["subunit_length"],
        "efficient_symmetry": result["efficient_symmetry"],
        "sample_index": result["sample_index"],
        "seed": result["seed"],
        "clash_score": result.get("clash_score"),
        "pdbTM_complex": result.get("pdbTM_complex"),
        "pdbTM_subunit": result.get("pdbTM_subunit"),
        "is_designable": result.get("is_designable", False),
        "best_scTM": best_scTM,
        "best_scRMSD": best_scRMSD,
    }

    if sc_data:
        for seq_result in sc_data:
            rows.append({
                **base,
                "seq_header": seq_result.get("header", ""),
                "scTM": seq_result["scTM"],
                "scRMSD": seq_result["scRMSD"],
            })
    else:
        rows.append({
            **base,
            "seq_header": "",
            "scTM": None,
            "scRMSD": None,
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    collect_only: bool = False,
    test_run: bool = False,
    generator: str = "framediff",
):
    """Launch symmetric protein generation metrics evaluation.

    Args:
        collect_only: Only collect existing results into CSV.
        test_run: 1 sample per symmetry at smallest length.
        generator: Generator model name (framediff, rfdiffusion, ...).
    """
    import sys
    sys.path.insert(0, REPO_ROOT)

    from evals.job_specs import build_all_specs, build_test_specs
    from dataclasses import asdict

    if collect_only:
        print("Collecting results...")
        result = collect_results.remote()
        print(f"Collection result: {result}")
        return

    # Build job list
    jobs = []
    for efficient in [True, False]:
        if test_run:
            specs = build_test_specs(efficient_symmetry=efficient)
        else:
            specs = build_all_specs(efficient_symmetry=efficient)

        n_samples = 1 if test_run else SAMPLES_PER_JOB
        for spec in specs:
            jobs.append((asdict(spec), generator, n_samples, efficient))

    mode = "TEST RUN" if test_run else "FULL RUN"
    print(f"{mode}: Launching {len(jobs)} jobs with generator={generator}")
    print(f"  {len(jobs)//2} specs x 2 efficiency modes, "
          f"{'1 sample' if test_run else f'{SAMPLES_PER_JOB} samples'} each")

    results = list(run_single_job.starmap(jobs))

    succeeded = sum(1 for r in results if r["status"] == "success")
    print(f"\n=== {succeeded}/{len(jobs)} jobs completed ===")
    for r in results:
        eff = "efficient" if r["efficient"] else "nonefficient"
        print(f"  {r['generator']} {r['symmetry']} "
              f"len={r['total_length']} {eff}: {r['num_samples']} samples")

    print("\nCollecting results...")
    collect_result = collect_results.remote()
    print(f"Collection: {collect_result}")
