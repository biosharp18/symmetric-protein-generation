"""Auto-infer ProteinMPNN tied positions for homo-oligomer PDBs.

If the input PDB has >1 chain and all chains have equal CA count, we tie
corresponding positions across chains so ProteinMPNN designs identical
subunit sequences (matches FrameDiff's symmetric design behavior).

Also provides first-chain subunit extraction for subunit-level pdbTM.
"""

import json
import os
from typing import Optional

from Bio.PDB import PDBParser, PDBIO, Select


class _FirstChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def _chain_ca_counts(pdb_path: str) -> dict:
    """Return {chain_id: ca_count} for standard residues (skip HETATM-only chains)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    counts = {}
    for chain in structure.get_chains():
        n = 0
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" in res:
                n += 1
        counts[chain.get_id()] = n
    return counts


def infer_tied_positions(
    pdb_path: str,
    output_jsonl_path: str,
    pdb_name: Optional[str] = None,
) -> Optional[str]:
    """Write a ProteinMPNN tied_positions JSONL if the PDB is a homo-oligomer.

    Returns the output path if tying was applied, else None (single chain,
    unequal chains, or no standard residues).

    The output format matches ProteinMPNN's make_tied_positions_dict.py with
    --homooligomer 1: one JSON object mapping pdb_name -> list of per-position
    dicts [{chain: [pos]}, ...] using 1-indexed positions.
    """
    counts = _chain_ca_counts(pdb_path)
    # Drop empty chains (e.g., ligand-only chains).
    counts = {c: n for c, n in counts.items() if n > 0}

    if len(counts) < 2:
        return None
    lengths = set(counts.values())
    if len(lengths) != 1:
        return None  # unequal chains — can't tie homo-oligomerically

    chain_list = sorted(counts.keys())
    chain_length = counts[chain_list[0]]

    if pdb_name is None:
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    tied_positions_list = [
        {chain: [i] for chain in chain_list}
        for i in range(1, chain_length + 1)
    ]
    my_dict = {pdb_name: tied_positions_list}

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    with open(output_jsonl_path, "w") as f:
        f.write(json.dumps(my_dict) + "\n")
    return output_jsonl_path


def extract_first_chain(pdb_path: str, output_path: str) -> Optional[str]:
    """Write the first chain of a multi-chain PDB to output_path.

    Returns the output path on success, None if the PDB has one or zero chains.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    chains = list(structure.get_chains())
    if len(chains) <= 1:
        return None
    first_id = chains[0].get_id()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, _FirstChainSelect(first_id))
    return output_path


def chain_info(pdb_path: str) -> dict:
    """Return chain counts and whether the PDB looks homo-oligomeric."""
    counts = _chain_ca_counts(pdb_path)
    counts = {c: n for c, n in counts.items() if n > 0}
    lengths = set(counts.values())
    return {
        "num_chains": len(counts),
        "chain_lengths": counts,
        "is_homo_oligomer": len(counts) >= 2 and len(lengths) == 1,
    }
