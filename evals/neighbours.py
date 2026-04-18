"""Crop a multi-chain PDB to the reference chain + its k-1 nearest neighbours.

Used by --fold-neighbours-k to bound self-consistency cost at fixed k chains
regardless of total complex size. "Nearest" is chain-centroid Euclidean
distance (Å), computed over CA atoms.

Usage (standalone):
    python -m evals.neighbours input.pdb output.pdb --k 3 --reference A

Usage (from Python):
    from evals.neighbours import crop_to_neighbours
    info = crop_to_neighbours("in.pdb", "out.pdb", k=3, reference="A")
"""

import argparse
import logging
import math
import os
import string
import sys

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

log = logging.getLogger(__name__)


def chain_centroids(pdb_path: str) -> dict[str, tuple[float, float, float]]:
    """Return {chain_id: CA-centroid (Å)} for every chain with at least one CA.

    Chains without CA atoms (e.g. ligand-only) are omitted rather than raising —
    the callers only care about protein chains.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)

    out: dict[str, tuple[float, float, float]] = {}
    for model in structure:
        for chain in model:
            sx = sy = sz = 0.0
            n = 0
            for res in chain:
                if "CA" not in res:
                    continue
                x, y, z = res["CA"].coord
                sx += float(x); sy += float(y); sz += float(z)
                n += 1
            if n > 0:
                out[chain.id] = (sx / n, sy / n, sz / n)
        break  # first model only
    return out


def rank_chains_by_proximity(
    pdb_path: str, reference: str = "A",
) -> list[tuple[str, float]]:
    """Rank chains by centroid distance to `reference` chain, closest first.

    Returns [(chain_id, distance_A), ...] excluding the reference itself.
    Stable tiebreaker on chain_id so equidistant neighbours (common in
    symmetric complexes) always land in the same order.
    """
    cents = chain_centroids(pdb_path)
    if reference not in cents:
        raise ValueError(
            f"Reference chain {reference!r} not found in {pdb_path}; "
            f"available chains: {sorted(cents)}"
        )
    ref = cents[reference]
    ranked = []
    for cid, xyz in cents.items():
        if cid == reference:
            continue
        d = math.sqrt(
            (xyz[0] - ref[0]) ** 2
            + (xyz[1] - ref[1]) ** 2
            + (xyz[2] - ref[2]) ** 2
        )
        ranked.append((cid, d))
    return sorted(ranked, key=lambda t: (t[1], t[0]))


def crop_to_neighbours(
    pdb_in: str,
    pdb_out: str,
    k: int,
    reference: str = "A",
    renumber_chains: bool = True,
) -> dict:
    """Write `pdb_out` with `reference` + (k-1) nearest chains from `pdb_in`.

    Args:
        pdb_in: source multi-chain PDB.
        pdb_out: destination PDB.
        k: number of chains to retain (must be >= 1).
        reference: anchor chain ID; always kept.
        renumber_chains: if True, rename kept chains to A, B, C… in proximity
            order (reference → A, nearest neighbour → B, ...). This is what
            ProteinMPNN / infer_tied_positions expect so the subcomplex looks
            like a clean homo-oligomer.

    Returns:
        {
            "kept_chains":    [original_chain_ids, in proximity order],
            "distances":      [float Å, same order; reference has 0.0],
            "renumber":       {orig_id: new_id},
            "num_kept":       int,
            "num_total":      int,  # chains in input
        }

    Notes:
        - If k >= total chains, the cropped PDB contains every chain (no-op,
          still written so callers can use a uniform path).
        - k must be >= 1. k==1 keeps only the reference chain.
        - If more than 26 chains would need IDs, renumbering fails.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    ranked = rank_chains_by_proximity(pdb_in, reference=reference)
    total = len(ranked) + 1
    take_neighbours = min(max(0, k - 1), len(ranked))
    kept = [reference] + [cid for cid, _ in ranked[:take_neighbours]]
    distances = [0.0] + [d for _, d in ranked[:take_neighbours]]

    if renumber_chains:
        new_ids = string.ascii_uppercase
        if len(kept) > len(new_ids):
            raise ValueError(
                f"Too many chains to renumber: {len(kept)} > 26"
            )
        renumber = {old: new_ids[i] for i, old in enumerate(kept)}
    else:
        renumber = {c: c for c in kept}

    _write_cropped(pdb_in, pdb_out, kept, renumber)
    return {
        "kept_chains": kept,
        "distances": distances,
        "renumber": renumber,
        "num_kept": len(kept),
        "num_total": total,
    }


def _write_cropped(
    pdb_in: str, pdb_out: str,
    kept: list[str], renumber: dict[str, str],
) -> None:
    """Write a new PDB containing only `kept` chains, renamed per `renumber`,
    in the order given by `kept`."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_in)

    new_structure = Structure("cropped")
    new_model = Model(0)
    new_structure.add(new_model)

    # Index source chains by ID for deterministic output-order control.
    source_chains: dict[str, Chain] = {}
    for model in structure:
        for chain in model:
            source_chains[chain.id] = chain
        break  # first model only

    for orig_id in kept:
        src = source_chains.get(orig_id)
        if src is None:
            raise ValueError(
                f"Chain {orig_id!r} missing from {pdb_in} at write time"
            )
        new_chain = Chain(renumber[orig_id])
        new_model.add(new_chain)
        for res in src:
            new_res = res.copy()
            new_res.detach_parent()
            new_chain.add(new_res)

    out_dir = os.path.dirname(os.path.abspath(pdb_out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(pdb_out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input_pdb")
    p.add_argument("output_pdb")
    p.add_argument("--k", type=int, required=True,
                   help="Number of chains to keep (reference + k-1 nearest).")
    p.add_argument("--reference", default="A",
                   help='Anchor chain (default: "A").')
    p.add_argument("--no-renumber", dest="renumber", action="store_false",
                   help="Keep original chain IDs instead of renumbering A,B,C...")
    args = p.parse_args(argv)

    info = crop_to_neighbours(
        args.input_pdb, args.output_pdb,
        k=args.k, reference=args.reference,
        renumber_chains=args.renumber,
    )
    print(
        f"Kept {info['num_kept']} / {info['num_total']} chains: "
        f"{info['kept_chains']} (dist Å: "
        f"{[f'{d:.2f}' for d in info['distances']]})"
    )
    print(f"Renumber: {info['renumber']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
