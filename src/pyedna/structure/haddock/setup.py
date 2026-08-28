"""Prepare HADDOCK inputs and post-process docked DNA–dye structures."""

from pathlib import Path

from .config import _flatten_docking_overrides, _write_docking_config
from .finalize import _reformat_docked_models, _select_best_models
from .restraints import _prepare_dna_for_haddock, _write_bond_restraints
from .topology import _combine_ligand_topologies, _prepare_dye_topologies

__all__ = ["HaddockSetup"]


class HaddockSetup:
    """Prepare HADDOCK inputs and convert completed docking output into final structures."""

    def __init__(self, config, dna_pdb, instances, workdir="."):
        self.config = config
        self.dna_pdb = Path(dna_pdb)
        self.instances = instances
        self.workdir = Path(workdir)
        self.haddock_dir = self.workdir / "haddock"
        self.structure_dir = self.workdir / "structures"
        self.docking_overrides = _flatten_docking_overrides(
            self.config.haddock.overrides
        )

    def prepare_inputs(self):
        """Write dye topologies, DNA inputs, restraints, and the HADDOCK configuration."""

        _prepare_dye_topologies(self.instances, self.workdir)
        self.top_file, self.par_file = _combine_ligand_topologies(
            self.instances, self.workdir
        )
        self.haddock_dna_pdb, self.bonding_csv = _prepare_dna_for_haddock(
            self.dna_pdb, self.instances, self.workdir
        )
        self.restraint_file, self.bond_file = _write_bond_restraints(
            self.instances,
            self.haddock_dna_pdb,
            output=self.haddock_dir / "bond_restraint.tbl",
            bond_output=self.haddock_dir / "bonds.csv",
        )
        self.docking_config = _write_docking_config(
            dna_pdb=self.haddock_dna_pdb,
            instances=self.instances,
            top_file=self.top_file,
            par_file=self.par_file,
            restraint_file=self.restraint_file,
            workdir=self.workdir,
            override_values=self.docking_overrides,
            template=(
                Path(__file__).resolve().parents[2]
                / "data"
                / "haddock_templates"
                / "docking_config.cfg"
            ),
        )
        return self

    def process_results(self):
        """Select, reconstruct, and annotate the requested top HADDOCK models."""

        for instance in self.instances:
            instance.set_prepared_paths(self.workdir)
            required = [
                instance.pdb, instance.top, instance.par,
                instance.attach, instance.mapping,
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{instance.name}: missing prepared HADDOCK files: {missing}"
                )

        _select_best_models(
            run_dir=self.haddock_dir / "run",
            output_dir=self.structure_dir,
            top=self.config.haddock.top_models,
            structure_name=self.config.name,
        )
        _reformat_docked_models(
            instances=self.instances,
            dna_template=self.haddock_dir / f"{self.config.dna.name}_haddock.pdb",
            bonding_csv=self.haddock_dir / f"{self.config.dna.name}_bonding.csv",
            structure_dir=self.structure_dir,
            bond_file=self.haddock_dir / "bonds.csv",
            model_pattern=f"{self.config.name}_*.pdb",
            attachments=self.config.attachments,
        )
        return self
