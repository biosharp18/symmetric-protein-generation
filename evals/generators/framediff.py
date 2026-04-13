"""FrameDiff symmetric backbone generator."""

import os
import logging
import numpy as np
import torch

from .base import BackboneGenerator, SymmetrySpec, GenerationResult

log = logging.getLogger(__name__)


class FrameDiffGenerator(BackboneGenerator):

    def __init__(self, framediff_dir: str = "/root/Framediff",
                 weights_path: str = None, fold_method: str = "esmfold"):
        self._framediff_dir = framediff_dir
        self._weights_path = weights_path or os.path.join(
            framediff_dir, "weights", "paper_weights.pth")
        self._fold_method = fold_method
        self._sampler = None
        self._conf = None

    def name(self) -> str:
        return "framediff"

    def setup(self, job_dir: str, gpu_id: int = 0):
        import sys
        sys.path.insert(0, self._framediff_dir)
        os.chdir(self._framediff_dir)

        from omegaconf import OmegaConf
        base_cfg = OmegaConf.load(
            os.path.join(self._framediff_dir, "config", "base.yaml"))
        infer_cfg = OmegaConf.load(
            os.path.join(self._framediff_dir, "config", "inference.yaml"))
        self._conf = OmegaConf.merge(base_cfg, infer_cfg)
        if "defaults" in self._conf:
            del self._conf["defaults"]

        self._conf.inference.gpu_id = gpu_id
        self._conf.inference.pt_hub_dir = "/root/.cache/torch/"
        self._conf.inference.pmpnn_dir = os.path.join(
            self._framediff_dir, "ProteinMPNN")
        self._conf.inference.weights_path = self._weights_path
        self._conf.inference.fold_method = self._fold_method
        self._conf.inference.output_dir = job_dir
        self._conf.inference.name = "results"

        # Profiling
        self._conf.inference.profiling.enabled = False

        self._job_dir = job_dir

    def _configure_symmetry(self, spec: SymmetrySpec, efficient_symmetry: bool):
        """Apply symmetry settings to the config."""
        self._conf.model.symmetry_type = spec.symmetry_type
        if spec.symmetry_type == "C":
            self._conf.model.symmetry = spec.symmetry_order
        else:
            self._conf.model.symmetry = 0
        self._conf.model.efficient_symmetry = efficient_symmetry
        self._conf.model.diffuse_subunit_only = spec.extra.get(
            "diffuse_subunit_only", True)
        self._conf.model.symmetry_origin_translation = spec.symmetry_origin_translation
        if spec.face_offset_radius > 0:
            self._conf.model.face_offset_radius = spec.face_offset_radius

        # Sample settings
        self._conf.inference.samples.min_length = spec.subunit_length
        self._conf.inference.samples.max_length = spec.subunit_length
        self._conf.inference.samples.length_step = 1
        self._conf.inference.samples.seq_per_sample = 8
        self._conf.inference.samples.samples_per_length = 1

        self._conf.inference.diffusion.num_t = 500
        self._conf.inference.diffusion.noise_scale = 0.1
        self._conf.inference.diffusion.min_t = 0.01

    def generate(
        self,
        spec: SymmetrySpec,
        output_dir: str,
        seed: int,
    ) -> GenerationResult:
        efficient = spec.extra.get("efficient_symmetry", True)
        self._configure_symmetry(spec, efficient)

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Lazy-init the sampler (or re-init if symmetry changed)
        from experiments.inference_se3_diffusion import Sampler
        from data.symmetry_utils import get_symmetry_order

        sampler = Sampler(self._conf)
        symm_order = get_symmetry_order(
            spec.symmetry_type,
            spec.symmetry_order if spec.symmetry_type == "C" else 0,
        )

        sample_output = sampler.sample(spec.subunit_length)
        traj_paths = sampler.save_traj(
            sample_output["prot_traj"],
            sample_output["rigid_0_traj"],
            np.ones(spec.total_length),
            output_dir=output_dir,
            symm_order=symm_order,
        )

        pdb_path = traj_paths["sample_path"]

        # Extract first-chain subunit PDB
        subunit_path = os.path.join(output_dir, "subunit.pdb")
        try:
            _extract_subunit_pdb(pdb_path, subunit_path)
        except Exception as e:
            log.warning(f"Subunit extraction failed: {e}")
            subunit_path = None

        return GenerationResult(
            pdb_path=pdb_path,
            subunit_pdb_path=subunit_path,
            tied_positions_path=traj_paths.get("tied_path"),
            metadata={
                "efficient_symmetry": efficient,
                "symm_order": symm_order,
            },
        )

    def get_self_consistency_runner(self):
        """Return the Sampler's run_self_consistency method for reuse."""
        if self._sampler is None:
            from experiments.inference_se3_diffusion import Sampler
            self._sampler = Sampler(self._conf)
        return self._sampler


def _extract_subunit_pdb(pdb_path, output_path):
    """Extract the first chain (subunit) from a multi-chain PDB file."""
    from Bio.PDB import PDBParser, PDBIO, Select

    class FirstChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id
        def accept_chain(self, chain):
            return chain.get_id() == self.chain_id

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    first_chain = next(structure.get_chains())
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, FirstChainSelect(first_chain.get_id()))
