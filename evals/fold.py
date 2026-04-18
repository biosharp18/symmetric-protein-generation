"""ESMFold singleton + fold_fn factory for the evaluation pipeline.

Torch hub cache is redirected to the shared tank storage so we don't
re-download the 2.6 GB ESMFold weights on every user / every env.
"""

import os
import sys
import logging

log = logging.getLogger(__name__)

TANK_TORCH_CACHE = "/xuanwu-tank/west/gaorory/torch_cache"

if "TORCH_HOME" not in os.environ and os.path.isdir(TANK_TORCH_CACHE):
    os.environ["TORCH_HOME"] = TANK_TORCH_CACHE

# fair-esm's ESMFold imports `openfold`. The framediff submodule bundles a
# pure-python openfold subset, which avoids the CUDA-toolchain build needed by
# the official openfold package. Add it to sys.path if not already importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRAMEDIFF_DIR = os.path.join(_REPO_ROOT, "models", "framediff")
if _FRAMEDIFF_DIR not in sys.path and os.path.isdir(
    os.path.join(_FRAMEDIFF_DIR, "openfold")
):
    sys.path.insert(0, _FRAMEDIFF_DIR)

_MODEL = None
_MODEL_DEVICE = None


def get_esmfold(device: str = "cuda:0"):
    """Load (or return cached) ESMFold v1 model on the given device."""
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None and _MODEL_DEVICE == device:
        return _MODEL

    import torch
    import esm

    log.info(f"Loading ESMFold v1 on {device}")
    model = esm.pretrained.esmfold_v1()
    model = model.eval()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        model = model.to(device)
    _MODEL = model
    _MODEL_DEVICE = device
    return _MODEL


def fold_fn_factory(device: str = "cuda:0"):
    """Return a fold(sequence, save_path) callback bound to an ESMFold instance."""
    model = get_esmfold(device)

    def fold(sequence: str, save_path: str) -> str:
        import torch
        with torch.no_grad():
            pdb_str = model.infer_pdb(sequence)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(pdb_str)
        return pdb_str

    return fold


def fold_fn_batch_factory(device: str = "cuda:0", batch_size: int = 4):
    """Return a batched fold_batch(sequences, save_paths) callback.

    Folds `sequences` in chunks of `batch_size` via ESMFold's native
    `infer_pdbs`, which runs all sequences in a chunk through a single
    forward pass. On CUDA OOM (or any batch-level exception), the chunk
    that failed is retried sequentially so one bad sequence doesn't
    poison the rest.

    Arguments must be parallel lists. Each PDB is written to its
    corresponding `save_paths[i]`. Caller is responsible for ensuring
    output directories exist; we also `makedirs` defensively per path.
    """
    model = get_esmfold(device)

    def _fold_one(seq: str, path: str):
        import torch
        with torch.no_grad():
            pdb_str = model.infer_pdb(seq)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(pdb_str)

    def _fold_chunk(seqs, paths):
        import torch
        with torch.no_grad():
            pdb_strs = model.infer_pdbs(seqs)
        for p, s in zip(paths, pdb_strs):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(s)

    def fold_batch(sequences, save_paths):
        import torch
        assert len(sequences) == len(save_paths), (
            f"length mismatch: {len(sequences)} seqs vs {len(save_paths)} paths"
        )
        if not sequences:
            return

        for start in range(0, len(sequences), batch_size):
            chunk_seqs = sequences[start : start + batch_size]
            chunk_paths = save_paths[start : start + batch_size]
            try:
                _fold_chunk(chunk_seqs, chunk_paths)
            except torch.cuda.OutOfMemoryError as e:
                log.warning(
                    f"ESMFold batch OOM (chunk start={start}, size={len(chunk_seqs)}): "
                    f"{e}. Falling back to per-sequence folding for this chunk."
                )
                torch.cuda.empty_cache()
                for s, p in zip(chunk_seqs, chunk_paths):
                    try:
                        _fold_one(s, p)
                    except Exception as inner:
                        log.warning(f"  per-seq fallback failed for {p}: {inner}")
            except Exception as e:
                # Non-OOM batch failure. Still fall back; one bad seq
                # shouldn't wipe out the whole chunk.
                log.warning(
                    f"ESMFold batch failed (chunk start={start}, size={len(chunk_seqs)}): "
                    f"{e}. Falling back to per-sequence folding."
                )
                for s, p in zip(chunk_seqs, chunk_paths):
                    try:
                        _fold_one(s, p)
                    except Exception as inner:
                        log.warning(f"  per-seq fallback failed for {p}: {inner}")

    return fold_batch
