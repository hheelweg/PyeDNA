"""Coordinate DNA, attachment, HADDOCK, and Amber structure workflows."""

import os
from pathlib import Path

from pyedna.config import get_config

from .attachments import (
    DyeLinkerConfig,
    create_dye_instances,
    load_dye_definitions,
)
from .config import StructureConfig
from .dna import prepare_dna


def _cleanup_file(path, label):
    """Remove a temporary workflow file if it exists."""
    path = Path(path)

    if path.exists():
        path.unlink()
        print(f"Removed {label}: {path}")


class StructureBuilder:
    """Coordinate DNA preparation, HADDOCK docking, and Amber input preparation."""

    def __init__(self, structure_config, workdir=".", dna_dir=None, dye_dir=None,
                 lnk_dir=None):
        self.config = structure_config
        self.workdir = Path(workdir)
        config = get_config()
        dna_root = dna_dir or config.libraries.dna_dir
        dye_root = dye_dir or config.libraries.dye_dir
        lnk_root = lnk_dir or config.libraries.linker_dir
        self.dna_dir = Path(dna_root) if dna_root else None
        self.dye_dir = Path(dye_root) if dye_root else None
        self.lnk_dir = Path(lnk_root) if lnk_root else None
        self.dna_pdb = self.workdir / f"{self.config.dna.name}.pdb"
        self.dye_definitions = {}
        self.dye_instances = []
        self.generated_dyelnks = {}

    @classmethod
    def from_file(cls, path, workdir=None, **kwargs):
        """Construct a builder from a structure TOML file."""

        path = Path(path)
        return cls(
            StructureConfig.from_file(path),
            workdir=path.parent if workdir is None else workdir,
            **kwargs,
        )

    def _load_dyes(self):
        """Resolve dye definitions and instantiate the configured dye copies."""

        if self.dye_dir is None and not self.config.attachments:
            raise EnvironmentError("DYE_DIR is not set")
        self._load_generated_dyelnks()
        self.dye_definitions = load_dye_definitions(
            self.config.dyes,
            self.dye_dir,
            generated=self.generated_dyelnks,
            workdir=self.workdir,
            dye_forcefield=self.config.amber.dye_forcefield,
        )
        self.dye_instances = create_dye_instances(
            self.config.dyes, self.dye_definitions
        )
        return self.dye_instances

    def _load_generated_dyelnks(self):
        """Resolve unique dye-linker template definitions requested by attachments."""

        if not self.config.attachments:
            self.generated_dyelnks = {}
            return self.generated_dyelnks

        self.generated_dyelnks = {
            attachment.name: DyeLinkerConfig.from_names(
                attachment.dye,
                attachment.linker,
                dye_forcefield=self.config.amber.dye_forcefield,
                dna_forcefield=self.config.amber.dna_forcefield,
                dye_dir=self.dye_dir,
                lnk_dir=self.lnk_dir,
            )
            for attachment in self.config.attachments
        }
        return self.generated_dyelnks

    def _prepare_linked_dyes(self):
        """Generate linked dye MOL2 files used as explicit intermediates."""

        for name, dyelnk in self._load_generated_dyelnks().items():
            mol2_output = self.workdir / f"{name}_linked.mol2"
            frcmod_output = self.workdir / f"{name}_linked.frcmod"

            if not mol2_output.exists():
                mol2_output = dyelnk.build_linked_mol2(self.workdir, name=name)
                print(f"Generated dye-linker MOL2: {mol2_output}")
            elif not frcmod_output.exists():
                frcmod_output = dyelnk.build_linked_frcmod(
                    mol2_output,
                    output_file=frcmod_output,
                    workdir=self.workdir,
                )
                print(f"Generated dye-linker FRCMOD: {frcmod_output}")

        return self.generated_dyelnks

    def prepare_dna(self):
        """Generate DNA with NAB or copy it from the configured DNA library."""

        if self.config.dna.source == "library" and self.dna_dir is None:
            raise EnvironmentError("DNA_DIR is not set")
        self.dna_pdb = prepare_dna(
            self.config, dna_dir=self.dna_dir, workdir=self.workdir
        )
        return self.dna_pdb

    def prepare_haddock(self):
        """Prepare DNA, dyes, restraints, and configuration for a HADDOCK run."""

        from .haddock import HaddockSetup

        self.prepare_dna()
        self._prepare_linked_dyes()
        self._load_dyes()
        setup = HaddockSetup(
            config=self.config,
            dna_pdb=self.dna_pdb,
            instances=self.dye_instances,
            workdir=self.workdir,
        )
        setup.prepare_inputs()
        _cleanup_file(self.dna_pdb, "temporary DNA PDB")
        return setup

    def finalize_haddock(self):
        """Convert completed HADDOCK output into final DNA–dye PDB structures."""

        from .haddock import HaddockSetup

        self._load_dyes()
        setup = HaddockSetup(
            config=self.config,
            dna_pdb=self.dna_pdb,
            instances=self.dye_instances,
            workdir=self.workdir,
        )
        setup.process_results()
        return setup

    def prepare(self):
        """Backward-compatible alias for prepare_haddock."""

        return self.prepare_haddock()

    def finalize(self):
        """Backward-compatible alias for finalize_haddock."""

        return self.finalize_haddock()

    def prepare_amber(self, run_tleap=True):
        """Prepare a finalized structure and run tleap by default."""

        from .amber import AmberSetup

        setup = AmberSetup.from_config(
            self.config,
            workdir=self.workdir,
            dye_dir=self.dye_dir,
            lnk_dir=self.lnk_dir,
        )
        setup.prepare(run_tleap=run_tleap)
        return setup
