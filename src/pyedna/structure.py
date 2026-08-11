"""DNA preparation and molecular wrappers shared by structure and QM workflows."""

from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import config
from . import fileproc as fp
from . import geomtools as geo


def set_chain_and_segid(line, chain="A", segid="A"):
    """Return a PDB record with normalized chain and segment identifiers."""
    line = line[:21] + chain + line[22:]
    padded = line.ljust(76)
    return padded[:72] + f"{segid:>4s}" + padded[76:]


def normalize_dna_pdb(pdb_file, chain="A", segid="A"):
    """Normalize coordinate-record chain and segment identifiers in place."""
    pdb_file = Path(pdb_file)
    lines = [
        set_chain_and_segid(line, chain, segid)
        if line.startswith(("ATOM  ", "HETATM")) else line
        for line in pdb_file.read_text().splitlines()
    ]
    pdb_file.write_text("\n".join(lines) + "\n")


def prepare_dna(structure_config, dna_dir, workdir="."):
    """Copy or generate configured DNA and normalize it for HADDOCK."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    output_pdb = workdir / f"{structure_config.dna_name}.pdb"

    # (1) Obtain the DNA from the library or generate it with NAB.
    if structure_config.dna_source == "library":
        source_pdb = Path(dna_dir) / f"{structure_config.dna_name}.pdb"
        if not source_pdb.exists():
            raise FileNotFoundError(f"DNA template not found: {source_pdb}")
        shutil.copy2(source_pdb, output_pdb)
        print(f"Copied DNA template: {source_pdb} -> {output_pdb}")
    elif structure_config.dna_source == "generate":
        dna = CreateDNA(structure_config.dna_name, structure_config.dna_type)
        dna.feedDNAseq(structure_config.dna_sequence)
        generated_pdb = dna.createDNA(workdir=workdir)
        if generated_pdb != output_pdb:
            shutil.move(generated_pdb, output_pdb)
        print(f"Generated DNA structure: {output_pdb}")
    else:
        raise ValueError(
            f"Unknown dna_source {structure_config.dna_source!r}; "
            "expected 'library' or 'generate'"
        )

    # (2) Standardize identifiers expected by the downstream workflow.
    normalize_dna_pdb(output_pdb)
    return output_pdb


class CreateDNA:
    """Generate double-helical DNA from a sequence with AmberTools NAB."""

    def __init__(self, name="dna", type="double_helix"):
        """Store the output name and supported DNA structure type."""
        if type != "double_helix":
            raise NotImplementedError("Other DNA structures are not implemented")
        self.name = name
        self.type = type
        self.sequence = None

    def feedDNAseq(self, DNA_sequence):
        """Set the nonempty sequence used to generate the DNA structure."""
        if not DNA_sequence:
            raise ValueError("DNA sequence cannot be empty")
        self.sequence = DNA_sequence

    @staticmethod
    def parseDNAStructure(file):
        """Read legacy DNA-generation keys from a parameter file."""
        params = fp.readParams(file)
        return {key: params.get(key) for key in ("dna_sequence", "dna_type", "dna_name")}

    def loadTemplate(self):
        """Load the NAB template for the configured DNA structure type."""
        template = Path(config.PROJECT_HOME) / "data" / "dna_templates" / f"{self.type}.nab"
        if not template.exists():
            raise FileNotFoundError(f"DNA template not found: {template}")
        return template.read_text()

    def writeNAB(self, workdir="."):
        """Render the configured sequence into a NAB input file."""
        if self.sequence is None:
            raise ValueError("Specify a DNA sequence before writing NAB input")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        script = self.loadTemplate().replace("{DNA_SEQUENCE}", self.sequence.lower())
        script = script.replace("{PDB_NAME}", f"{self.name}.pdb")
        nab_file = workdir / f"{self.name}.nab"
        nab_file.write_text(script)
        return nab_file

    def createDNA(self, remove_nab=True, workdir="."):
        """Run NAB and return the generated PDB path."""
        workdir = Path(workdir)
        nab_file = self.writeNAB(workdir)
        runner = Path(config.PROJECT_HOME) / "bin" / "create_dna.sh"
        subprocess.run(["bash", str(runner), nab_file.name], cwd=workdir, check=True)
        pdb_file = workdir / f"{self.name}.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(f"NAB did not create {pdb_file}")
        if remove_nab:
            nab_file.unlink()
        print(
            f"*** Creation of {pdb_file.name} completed: "
            f"DNA type = {self.type}, DNA sequence = {self.sequence}"
        )
        return pdb_file


class Chromophore:
    """Expose molecular coordinates and labels used by trajectory and QM analysis."""

    def __init__(self, Chromophore_u):
        """Wrap an MDAnalysis universe without changing its topology."""
        self.chromophore_u = Chromophore_u
        self.xyz, self.names, self.types, self.com, self.resnames = self.parseStructure()
        self.natoms = len(self.xyz)
        unique_names = np.unique(self.resnames)
        self.dye_name = unique_names[0] if len(unique_names) else None

    def parseStructure(self):
        """Return coordinates, atom metadata, and center of geometry."""
        return geo.getCoords(self.chromophore_u, "all")


def cleanPDB(inPath, outPath, res_code="DYE", mol_title="Dye molecule", printCONNECT=False):
    """Normalize and uniquely number atom names in a PDB file."""
    pdb = fp.PDB_DF()
    pdb.read_file(inPath)
    hetatm = pdb.data["HETATM"].copy()
    hetatm["atom_name"] = fp.clean_numbers(hetatm["atom_name"])
    hetatm = hetatm.sort_values(by=["atom_name", "atom_id"])
    symbols, counts = np.unique(hetatm["atom_name"], return_counts=True)
    hetatm["atom_name"] = fp.make_names(symbols, counts)
    pdb.data["MOLECULE"] = mol_title
    pdb.data["HETATM"] = hetatm
    pdb.write_file(outPath, resname=res_code, print_connect=printCONNECT)
