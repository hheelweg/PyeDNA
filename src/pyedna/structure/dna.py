"""Prepare DNA structures for structure-generation workflows."""

import os
import shutil
import subprocess
from pathlib import Path

from .. import config
from .. import fileproc as fp
from .. import utils


def _normalize_dna_pdb(pdb_file, chain="A", segid="A"):
    """Normalize chain and segment identifiers in a DNA PDB file in place."""

    pdb_file = Path(pdb_file)
    lines = []

    for line in pdb_file.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = fp.set_chain_and_segid(line, chain=chain, segid=segid)
        lines.append(line)

    pdb_file.write_text("\n".join(lines) + "\n")


def prepare_dna(config, dna_dir, workdir="."):
    """Create the configured DNA PDB from NAB or a DNA library template."""

    workdir = Path(workdir)
    output_pdb = workdir / f"{config.dna.name}.pdb"

    if config.dna.source == "library":
        source_pdb = Path(dna_dir) / f"{config.dna.name}.pdb"

        if not source_pdb.exists():
            raise FileNotFoundError(f"DNA template not found: {source_pdb}")

        shutil.copy2(source_pdb, output_pdb)
        print(f"Copied DNA template: {source_pdb} -> {output_pdb}")

    elif config.dna.source == "generate":
        dna = CreateDNA(name=config.dna.name, type=config.dna.type, workdir=workdir)
        dna.feedDNAseq(config.dna.sequence)
        dna.createDNA()

        generated_pdb = workdir / f"{config.dna.name}.pdb"

        if not generated_pdb.exists():
            raise FileNotFoundError(f"Generated DNA PDB not found: {generated_pdb}")

        if generated_pdb.resolve() != output_pdb.resolve():
            shutil.move(generated_pdb, output_pdb)

        print(f"Generated DNA structure: {output_pdb}")

    else:
        raise ValueError(
            f"Unknown dna_source {config.dna.source!r}; "
            "expected 'library' or 'generate'"
        )

    _normalize_dna_pdb(output_pdb, chain="A", segid="A")
    return output_pdb


# class for creating DNA structure (.pdb) from DNA sequence
class CreateDNA():

    def __init__(self, name = 'dna', type = 'double_helix', workdir='.'):

        self.type = type                                        # type of DNA strcuture we want to create
        if type != 'double_helix':
            raise NotImplementedError("Other DNA structures not implemented yet!")

        self.name = name                                        # name of DNA structure
        self.workdir = Path(workdir)
        self.is_sequence = False                                # flag to indicate whether DNA sequence has been specified


    # feed desired DNA sequence
    def feedDNAseq(self, DNA_sequence):
        self.sequence = DNA_sequence
        self.is_sequence = True

    # load DNA template for self.type from DNA data library
    def loadTemplate(self):
        # get directory for DNA templates
        dna_template_dir = os.path.join(config.PROJECT_HOME, 'data', 'dna_templates')
        # find template
        template_file = utils.findFileWithName(f"{self.type}.nab", dir=dna_template_dir)
        # load template
        with open(template_file, "r") as file:
            template = file.read()
        return template

    # writes NAB .nad input file
    def writeNAB(self):

        # (1) load DNA template
        self.template = self.loadTemplate()

        # (2) check if sequence is fed
        if not self.is_sequence:
            raise ValueError("Specify a DNA sequence first before proceeding!")

        # (3) replace sequence placeholder in template and set pdb name
        self.nab_script = self.template.replace("{DNA_SEQUENCE}", self.sequence.lower())
        self.nab_script = self.nab_script.replace("{PDB_NAME}", f"{self.name}.pdb")

        # (4) write .nab file
        self.workdir.mkdir(parents=True, exist_ok=True)
        with (self.workdir / f"{self.name}.nab").open("w") as file:
            file.write(self.nab_script)


    # run NAB to produce .pdb file of DNA strcture
    def createDNA(self, remove_nab = True):

        # (0) write .nab file
        self.writeNAB()

        # (1) locate shell script for running NAB and creating DNA pdb
        run_nab_script = os.path.join(config.PROJECT_HOME, 'bin', 'create_dna.sh')

        # (2) run NAB
        subprocess.run(
            ["bash", run_nab_script, f"{self.name}.nab"],
            cwd=self.workdir,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"*** Creation of {self.name}.pdb completed: DNA type = {self.type}, DNA sequence = {self.sequence}")

        # (3) clean directory (auxiliary .nab file)
        if remove_nab:
            (self.workdir / f"{self.name}.nab").unlink(missing_ok=True)
