"""Abstract base class for symmetric protein backbone generators.

Any model that generates symmetric protein backbones can be integrated
into the evaluation pipeline by subclassing ``BackboneGenerator`` and
implementing the three required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class SymmetrySpec:
    """Specification for a symmetric generation job."""
    symmetry_type: str          # "C", "T", "O", "I"
    symmetry_order: int         # e.g. 2 for C2, 12 for T, 24 for O, 60 for I
    total_length: int           # total residues across all subunits
    subunit_length: int         # residues per subunit
    symmetry_origin_translation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    face_offset_radius: float = 0.0
    # Model-specific extra config (passed through to the generator)
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.symmetry_type == "C":
            return f"C{self.symmetry_order}"
        return self.symmetry_type


@dataclass
class GenerationResult:
    """Result of generating a single backbone sample."""
    pdb_path: str                           # path to the generated PDB
    subunit_pdb_path: Optional[str] = None  # path to extracted first-chain PDB
    tied_positions_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BackboneGenerator(ABC):
    """Abstract interface for symmetric protein backbone generators.

    Subclass this for each model (FrameDiff, RFdiffusion, Chroma, etc.)
    """

    @abstractmethod
    def setup(self, job_dir: str, gpu_id: int = 0):
        """One-time setup: load weights, build the model, etc.

        Called once per Modal container before any sampling.

        Args:
            job_dir: directory for this job's outputs.
            gpu_id: CUDA device index.
        """

    @abstractmethod
    def generate(
        self,
        spec: SymmetrySpec,
        output_dir: str,
        seed: int,
    ) -> GenerationResult:
        """Generate a single symmetric backbone sample.

        Args:
            spec: symmetry and length specification.
            output_dir: directory to write PDB and trajectory files.
            seed: random seed for reproducibility.

        Returns:
            GenerationResult with paths to the generated PDB(s).
        """

    @abstractmethod
    def name(self) -> str:
        """Short name for this generator (e.g. 'framediff', 'rfdiffusion')."""
