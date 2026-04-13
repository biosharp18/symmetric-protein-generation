"""Job specification definitions for symmetric protein generation evaluation.

Defines all (symmetry_type, total_length) combinations and computes
appropriate radius parameters for each.
"""

import math
from .generators.base import SymmetrySpec


# ---------------------------------------------------------------------------
# Radius heuristics
# ---------------------------------------------------------------------------

def cyclic_origin(subunit_len: int, symm_n: int) -> list:
    """Heuristic for cyclic symmetry origin translation.

    Reference: C10 @ subunit=50 uses origin=[4.0, 0, 0].
    """
    if symm_n <= 3:
        base = 0.5 * math.sqrt(subunit_len / 50.0)
    else:
        base = 4.0 * (subunit_len / 50.0) * (symm_n / 10.0) ** 0.3
    return [round(base, 1), 0.0, 0.0]


def polyhedral_origin(symmetry_type: str, subunit_len: int) -> tuple:
    """Scale polyhedral origin_translation and face_offset by subunit size.

    References:
        T @ subunit=50:  origin=[0, 1.7, 0], face_offset=1.5
        O @ subunit=50:  origin=[0, 3.0, 0], face_offset=1.5
        I @ subunit=100: origin=[0, 5.1, 0], face_offset=1.9
    """
    configs = {
        "T": {"origin": [0.0, 1.7, 0.0], "face_offset": 1.5, "ref": 50},
        "O": {"origin": [0.0, 3.0, 0.0], "face_offset": 1.5, "ref": 50},
        "I": {"origin": [0.0, 5.1, 0.0], "face_offset": 1.9, "ref": 100},
    }
    cfg = configs[symmetry_type]
    scale = subunit_len / cfg["ref"]
    scaled_origin = [round(v * scale, 2) for v in cfg["origin"]]
    scaled_face = round(cfg["face_offset"] * scale, 2)
    return scaled_origin, scaled_face


# ---------------------------------------------------------------------------
# Job spec builders
# ---------------------------------------------------------------------------

CYCLIC_JOBS = [
    ("C", 2,  [100, 200, 400]),
    ("C", 3,  [150, 300, 600]),
    ("C", 6,  [300, 600, 1200]),
    ("C", 10, [500, 1000, 2000]),
    ("C", 30, [1500, 3000, 6000]),
]

POLYHEDRAL_JOBS = [
    ("T", 12, [720, 1440, 2880]),
    ("O", 24, [1440, 2880, 5760]),
    ("I", 60, [1800, 3600, 6000]),
]


def build_all_specs(efficient_symmetry: bool = True) -> list[SymmetrySpec]:
    """Build SymmetrySpec for every (symmetry, total_length) combo."""
    specs = []

    for symm_type, symm_n, total_lengths in CYCLIC_JOBS:
        for total_len in total_lengths:
            subunit_len = total_len // symm_n
            origin = cyclic_origin(subunit_len, symm_n)
            specs.append(SymmetrySpec(
                symmetry_type=symm_type,
                symmetry_order=symm_n,
                total_length=total_len,
                subunit_length=subunit_len,
                symmetry_origin_translation=origin,
                face_offset_radius=0.0,
                extra={
                    "diffuse_subunit_only": True,
                    "efficient_symmetry": efficient_symmetry,
                },
            ))

    for symm_type, symm_order, total_lengths in POLYHEDRAL_JOBS:
        for total_len in total_lengths:
            subunit_len = total_len // symm_order
            origin, face_offset = polyhedral_origin(symm_type, subunit_len)
            specs.append(SymmetrySpec(
                symmetry_type=symm_type,
                symmetry_order=symm_order,
                total_length=total_len,
                subunit_length=subunit_len,
                symmetry_origin_translation=origin,
                face_offset_radius=face_offset,
                extra={
                    "diffuse_subunit_only": True,
                    "efficient_symmetry": efficient_symmetry,
                },
            ))

    return specs


def build_test_specs(efficient_symmetry: bool = True) -> list[SymmetrySpec]:
    """Build specs for test run: smallest length per symmetry only."""
    all_specs = build_all_specs(efficient_symmetry)
    smallest = {}
    for spec in all_specs:
        key = (spec.symmetry_type, spec.symmetry_order)
        if key not in smallest or spec.total_length < smallest[key].total_length:
            smallest[key] = spec
    return list(smallest.values())
