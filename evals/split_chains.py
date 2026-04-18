"""Split a single-chain PDB into multiple chains by subunit length.

FrameDiff and Foldflow emit symmetric assemblies as one long chain (residues
1..N where N = num_subunits * subunit_length). Downstream tools —
ProteinMPNN tied-positions, FoldSeek multimer search, our evaluation
pipeline — need proper multi-chain PDBs with distinct chain IDs. This
utility performs that split.

Assumes residues are ordered subunit-by-subunit: the first `subunit_length`
residues go to chain A, the next `subunit_length` to B, and so on.

Usage (standalone):
    python -m evals.split_chains input.pdb output.pdb --subunit-length 100

Usage (from Python):
    from evals.split_chains import split_pdb_into_chains
    split_pdb_into_chains("in.pdb", "out.pdb", 100)
"""

import argparse
import os
import string
import sys

from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure


def split_pdb_into_chains(
    input_pdb: str,
    output_pdb: str,
    subunit_length: int,
) -> int:
    """Split a single-chain PDB into chains of `subunit_length` residues each.

    Residues are assigned to chains in the order they appear in the file,
    and residue numbering restarts at 1 within each chain. Chain IDs are
    assigned A, B, C, … up to Z.

    Args:
        input_pdb: path to input PDB.
        output_pdb: path to write the split multi-chain PDB.
        subunit_length: residues per subunit.

    Returns:
        Number of chains written.

    Raises:
        ValueError: if total residues is not divisible by subunit_length, or
            if the result would require more than 26 chain IDs.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)

    residues = [
        res
        for model in structure
        for chain in model
        for res in chain
        if res.id[0] == " "  # standard residues only (skip HETATM)
    ]

    total = len(residues)
    if total == 0:
        raise ValueError(f"No standard residues found in {input_pdb}")
    if total % subunit_length != 0:
        raise ValueError(
            f"{input_pdb}: total residues ({total}) not divisible by "
            f"subunit_length ({subunit_length})"
        )
    num_chains = total // subunit_length
    if num_chains > 26:
        raise ValueError(
            f"Too many chains ({num_chains}); only 26 single-letter chain IDs available."
        )

    new_structure = Structure("split")
    new_model = Model(0)
    new_structure.add(new_model)

    chain_ids = string.ascii_uppercase[:num_chains]
    for ci, chain_id in enumerate(chain_ids):
        new_chain = Chain(chain_id)
        new_model.add(new_chain)
        for ri, orig_res in enumerate(
            residues[ci * subunit_length : (ci + 1) * subunit_length]
        ):
            new_res = orig_res.copy()
            new_res.detach_parent()
            new_res.id = (" ", ri + 1, " ")
            new_chain.add(new_res)

    out_dir = os.path.dirname(os.path.abspath(output_pdb))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    return num_chains


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_pdb", help="Input single-chain PDB.")
    parser.add_argument("output_pdb", help="Output multi-chain PDB.")
    parser.add_argument(
        "--subunit-length", type=int, required=True,
        help="Residues per subunit.",
    )
    args = parser.parse_args(argv)

    n = split_pdb_into_chains(args.input_pdb, args.output_pdb, args.subunit_length)
    print(f"Wrote {n} chains ({args.subunit_length} residues each) to {args.output_pdb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
