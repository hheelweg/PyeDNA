"""Coordinate DNA–dye assembly and retain temporary compatibility classes."""

import numpy as np
import subprocess
import os
import glob
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# from current package
from .. import fileproc as fp
from .. import geomtools as geo
from .. import utils
from .. import config
from .dye import create_dye_instances, load_dye_definitions
from .dyelnk import DyeLinkerConfig


@dataclass(frozen=True)
class DNAConfig:
    """Define whether DNA is generated from a sequence or copied from a library."""

    source: str
    name: str
    sequence: Optional[str] = None
    type: Optional[str] = None

    def __post_init__(self):
        if self.source not in {"generate", "library"}:
            raise ValueError("'dna.source' must be 'generate' or 'library'")
        if not self.name:
            raise ValueError("'dna.name' must be specified")
        if self.source == "generate" and (not self.sequence or not self.type):
            raise ValueError(
                "'dna.sequence' and 'dna.type' are required when dna.source='generate'"
            )

    def as_parameters(self):
        """Return the DNA mapping expected by MD and trajectory classes."""

        return {
            "dna_sequence": self.sequence,
            "dna_type": self.type,
            "dna_name": self.name,
        }


@dataclass(frozen=True)
class DyePlacement:
    """Describe one dye and the consecutive DNA residues it replaces."""

    name: str
    sites: list[int]


@dataclass(frozen=True)
class AttachmentConfig:
    """Describe one dye/linker attachment requested by structure TOML."""

    dye: str
    linker: str
    residue: int

    @property
    def name(self):
        return f"{self.dye}_{self.linker}"

    def as_placement(self):
        return DyePlacement(name=self.name, sites=[self.residue])


@dataclass(frozen=True)
class HaddockConfig:
    """Store HADDOCK model selection and optional parameter overrides."""

    top_models: int = 5
    overrides: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.top_models < 1:
            raise ValueError("'haddock.top_models' must be at least 1")
        if not isinstance(self.overrides, dict):
            raise ValueError("'haddock.overrides' must contain TOML sections")
        if any(not isinstance(values, dict) for values in self.overrides.values()):
            raise ValueError("Each 'haddock.overrides' section must contain key-value pairs")


@dataclass(frozen=True)
class AmberConfig:
    """Store tleap force-field, solvation, and output options."""

    model: int = 1
    output_name: Optional[str] = None
    dna_forcefield: str = "leaprc.DNA.OL15"
    dye_forcefield: str = "leaprc.gaff"
    water_forcefield: str = "leaprc.water.tip3p"
    water_model: str = "TIP3P"
    solvent_padding: float = 20.0
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    neutralize: bool = True

    def __post_init__(self):
        if self.model < 1:
            raise ValueError("'amber.model' must be at least 1")


@dataclass(frozen=True)
class StructureConfig:
    """Store and validate all structure-generation and Amber settings."""

    name: str
    dna: DNAConfig
    dyes: list[DyePlacement]
    attachments: list[AttachmentConfig] = field(default_factory=list)
    haddock: HaddockConfig = field(default_factory=HaddockConfig)
    amber: AmberConfig = field(default_factory=AmberConfig)

    def __post_init__(self):
        if not self.name:
            raise ValueError("'structure.name' must be specified")
        if self.amber.model > self.haddock.top_models:
            raise ValueError("'amber.model' cannot exceed 'haddock.top_models'")
        if self.attachments and self.dyes != [a.as_placement() for a in self.attachments]:
            raise ValueError("Do not mix legacy [[dyes]] with [[attachments]]")

        occupied = set()
        for dye in self.dyes:
            if not dye.name or not dye.sites:
                raise ValueError("Each dye requires a name and at least one site")
            sites = sorted(set(dye.sites))
            if sites != list(range(sites[0], sites[-1] + 1)):
                raise ValueError(f"{dye.name}: sites must be consecutive: {sites}")
            overlap = occupied.intersection(sites)
            if overlap:
                raise ValueError(
                    f"{dye.name}: DNA sites already assigned: {sorted(overlap)}"
                )
            occupied.update(sites)

    @classmethod
    def from_file(cls, path):
        """Load a complete structure configuration from a TOML file."""

        path = Path(path)
        with path.open("rb") as handle:
            data = tomllib.load(handle)

        try:
            structure = data["structure"]
            dna = data["dna"]
        except KeyError as exc:
            raise ValueError(f"{path}: missing [{exc.args[0]}] section") from exc

        try:
            legacy_dyes = data.get("dyes", [])
            attachments = [AttachmentConfig(**entry)
                           for entry in data.get("attachments", [])]

            if legacy_dyes and attachments:
                raise ValueError(f"{path}: do not mix [[dyes]] and [[attachments]]")

            dyes = (
                [attachment.as_placement() for attachment in attachments]
                if attachments else [DyePlacement(**dye) for dye in legacy_dyes]
            )

            return cls(
                name=structure["name"],
                dna=DNAConfig(**dna),
                dyes=dyes,
                attachments=attachments,
                haddock=HaddockConfig(**data.get("haddock", {})),
                amber=AmberConfig(**data.get("amber", {})),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: invalid configuration field: {exc}") from exc


def _normalize_dna_pdb(pdb_file, chain="A", segid="A"):
    """Normalize chain and segment identifiers in a DNA PDB file in place."""

    pdb_file = Path(pdb_file)
    lines = []

    for line in pdb_file.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = fp.set_chain_and_segid(line, chain=chain, segid=segid)
        lines.append(line)

    pdb_file.write_text("\n".join(lines) + "\n")

def _prepare_dna(config, dna_dir, workdir="."):
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


class StructureBuilder:
    """Coordinate DNA preparation, HADDOCK docking, and Amber input preparation."""

    def __init__(self, structure_config, workdir=".", dna_dir=None, dye_dir=None,
                 pyedna_home=None):
        self.config = structure_config
        self.workdir = Path(workdir)
        dna_root = dna_dir or os.environ.get("DNA_DIR")
        dye_root = dye_dir or os.environ.get("DYE_DIR")
        home_root = pyedna_home or os.environ.get("PYEDNA_HOME")
        self.dna_dir = Path(dna_root) if dna_root else None
        self.dye_dir = Path(dye_root) if dye_root else None
        self.pyedna_home = Path(home_root) if home_root else None
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
        self.dna_pdb = _prepare_dna(
            self.config, dna_dir=self.dna_dir, workdir=self.workdir
        )
        return self.dna_pdb

    def prepare(self):
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
            pyedna_home=self.pyedna_home,
        )
        setup.prepare_inputs()
        return setup

    def finalize(self):
        """Convert completed HADDOCK output into final DNA–dye PDB structures."""

        from .haddock import HaddockSetup

        self._load_dyes()
        setup = HaddockSetup(
            config=self.config,
            dna_pdb=self.dna_pdb,
            instances=self.dye_instances,
            workdir=self.workdir,
            pyedna_home=self.pyedna_home,
        )
        setup.process_results()
        return setup

    def prepare_amber(self, run_tleap=True):
        """Prepare a finalized structure and run tleap by default."""

        from .amber import AmberSetup

        setup = AmberSetup.from_config(
            self.config,
            workdir=self.workdir,
            dye_dir=self.dye_dir,
        )
        setup.prepare(run_tleap=run_tleap)
        return setup


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

        
        

# TODO: Remove after trajectory, quanttools, and dye-library construction no
# longer depend on this legacy analysis representation.
class Chromophore():

    def __init__(self, Chromophore_u):
        self.chromophore_u = Chromophore_u                                       # MDAnalysis object
        # parse structure: coordinates, atom names, center of mass
        self.xyz, self.names, self.types, self.com, self.resnames = self.parseStructure() 
        self.natoms = len(self.xyz)
        self.dye_name = np.unique(self.resnames)[0]
    
    # store .pdb source and directory information
    def storeSourcePath(self, path_to_dye):
        # TODO: Retained for scripts/create_dye.py; migrate dye-library creation
        # to DyeDefinition before removing this compatibility method.
        self.path = path_to_dye


    # parse structure
    def parseStructure(self):
        xyz, names, types, com, resnames = geo.getCoords(self.chromophore_u, 'all')
        return xyz, names, types, com, resnames

    # parse attachment info for chromophore
    # TODO : generalize to different linkers
    def parseAttachment(self, change_atom_names = True):
        # TODO: Retained for scripts/create_dye.py; this is not used by
        # StructureBuilder's DNA–dye assembly workflow.
        # (0) load attachment information from file
        try:
            with open(os.path.join(self.path, f"attach_{self.dye_name}.info"), "r") as file:
                self.attach_groups = [line.strip().split() for line in file]
        except FileNotFoundError:
            print("ERROR: Attachment information for this dye does not exist yet!")

        self.attach_num = len(self.attach_groups)                                        # len(attach_points) = number of attachments to DNA
        # (1) extract names of the Os, Ps that won't be deleted
        self.O_term, self.O_conn, self.P= [], [], []
        for attach_group in self.attach_groups:
            self.O_conn.append(attach_group[0])
            self.P.append(attach_group[1])
            self.O_term.append([attach_group[2], attach_group[3]])
        
        # (1) atoms that need to be deleted:
        self.delete_atoms = []
        # (1.1) need to remove one OH group and one H (last three elements) from both OPO3 groups
        for i in range(self.attach_num):
            self.delete_atoms += self.attach_groups[i][-3:]
        # (1.2) need to delete all O_term and P from 5' end 
        self.delete_atoms.append(self.P[0])
        self.delete_atoms += self.O_term[0]
        # (1.3) build delete string for select_atoms method
        self.del_string = str()
        for atom in self.delete_atoms:
            self.del_string += f" and not name {atom}" 


        # (3) indices of atoms where Chromophore will be attached to DNA
        self.attach_idx = [np.where(self.names == P)[0][0] for P in self.P]
        self.attach_pos = self.xyz[self.attach_idx]

        # (4) rename atoms in phosphate group that are not getting deleted
        # in order to simulate them with the DNA forcefield
        DNA_O_conn = ["O3'", "O5'"]
        DNA_O_term = ['OP1', 'OP2']
        DNA_P = "P"
        self.rename_atoms = dict()                  # dictionary for renaming atoms    
        self.rename_types = dict()                  # dictionary for setting OL15 atom types for these atoms        
        for i in range(self.attach_num):
            if self.P[i] not in self.delete_atoms:  
                self.rename_atoms[self.P[i]] = DNA_P
                self.rename_types[self.P[i]] = 'P'
            if self.O_conn[i] not in self.delete_atoms: 
                self.rename_atoms[self.O_conn[i]] = DNA_O_conn[i]
                self.rename_types[self.O_conn[i]] = 'OS'
            for j in range(2):
                if self.O_term[i][j] not in self.delete_atoms: 
                    self.rename_atoms[self.O_term[i][j]] = DNA_O_term[j]
                    self.rename_types[self.O_term[i][j]] = 'O2'

        # use dict to translate old atom names
        self.names = np.array([self.rename_atoms.get(atom, atom) for atom in self.names])
        # update self.names
        if change_atom_names:
            self.chromophore_u.atoms.names = self.names
            self.names = self.chromophore_u.atoms.names


    # write attachment information
    # TODO : generalize this to different linkers
    @staticmethod
    def writeAttachmentInfo(chromophore_u, dye_name, linker_atoms = ['P1', 'P2'], linker_group = 'phosphate'):
        # TODO: Retained for scripts/create_dye.py until dye-library creation
        # writes the current .attach format directly.
        
        attachment_info = []

        if linker_group != 'phosphate':
            raise NotImplementedError("Only phosphate group as linker currently implemeted!")
        
        if linker_group == 'phosphate':

            # we store the atom names for the phosphate attachment group as follows for each linker_atom in linker_atoms
            # attachment = [O_bridge, P, O_term, OH_term, H, OH_term, H]
            # where ...-C-O_bridge-P(=O_term)(-OH_term-H)(-OH_term-H) is the -OPO3H2 linker group

            # loop through linkers
            for linker_atom in linker_atoms:

                # get (oxygen) neighbors of linkers (phosphates)
                nearest_neighbors = Chromophore.getNeighborAtoms(chromophore_u, linker_atom)

                # among (oxygen) neighbors, identify which ones are terminal (=0), terminal (-OH) or bridging (-O-)
                OH_terminal, O_terminal, O_bridge = [], [], []
                for neighbor in nearest_neighbors.atoms.names:

                    # get next nearest neighbors
                    next_nearest_neighbors = Chromophore.getNeighborAtoms(chromophore_u, neighbor).select_atoms(f"not name {linker_atom}")

                    # identify type of neighbor and store information
                    if len(next_nearest_neighbors) == 0:
                        O_terminal.append(neighbor)
                    elif len(next_nearest_neighbors) == 1:
                        # determine whether the oxygen is terminal or bridging
                        if 'C' in next_nearest_neighbors.types:
                            O_bridge.append(neighbor)
                        else:
                            OH_terminal.append(neighbor)
                            OH_terminal.append(next_nearest_neighbors.atoms.names[0])

                    else:
                        raise Warning("Incorrect number of bond-neighbors identified in linker! Check PDB. \
                                    Might want to reduce cutoff to identify only nect neighbors")
                    
                # store attachment info as follows
                attachment = O_bridge + [linker_atom] + O_terminal + OH_terminal
                attachment_info.append(attachment)

            # write attachment info to file
            with open(f"attach_{dye_name}.info", "w") as file:
                for attachment in attachment_info:
                    file.write(" ".join(attachment) + "\n")



    @staticmethod
    def getNeighborAtoms(chromophore_u, target_atom_name, search_radius = 2.0):

        from MDAnalysis.lib import NeighborSearch

        # Select all atoms for searching
        all_atoms = chromophore_u.select_atoms("all")

        # Initialize NeighborSearch with all atoms
        ns = NeighborSearch.AtomNeighborSearch(all_atoms)

        # Search around target atom
        target_atom = chromophore_u.select_atoms(f"name {target_atom_name}")[0]
        neighbors = ns.search(target_atom, search_radius).select_atoms(f"not name {target_atom_name}")

        return neighbors



    # create force field with antechamber/parmchk2
    def createFF(self, charge = 0, ff = 'gaff', file_verbosity = 0):
        # TODO: Retained for scripts/create_dye.py; AmberSetup consumes
        # already prepared GAFF files from DYE_DIR.
        # (1) write updated pdb file after deletion of groups for attachment
        fp.deleteAtomsPDB(f'{self.dye_name}' + '.pdb', f'{self.dye_name}' + '_del.pdb', self.delete_atoms)
        # (2) use antechamber 
        command = f"antechamber -i '{self.dye_name}_del.pdb' -fi pdb -o {self.dye_name}.mol2 -fo mol2 -c bcc -s 2 -nc {charge} -m 1 -at {ff}"
        run_antechamber = subprocess.Popen(command, cwd = self.path, shell = True)
        run_antechamber.wait()
        # (3) run parmchk2
        command = f"parmchk2 -i {self.dye_name}.mol2 -f mol2 -o {self.dye_name}.frcmod -s gaff"
        run_parmchk2 = subprocess.Popen(command, cwd = self.path, shell = True)
        run_parmchk2.wait()
        # (4) delete axuiliary files in ff directory
        if file_verbosity == 0:
            for prefix in ["ANTECHAMBER", "sqm", "ATOMTYPE"]:
                pattern = os.path.join(f'{self.path}', f"{prefix}*")  # Match files with the prefix
                files_to_delete = glob.glob(pattern)                    # Get list of matching files
                
                for file in files_to_delete:
                    try:
                        os.remove(file)
                        print(f"Deleted: {file}")
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
        else:
            pass
        



# clean PDB files
# TODO : do we actually need this ?? If yes, put this somewhere else
def cleanPDB(inPath, outPath, res_code='DYE', mol_title='Dye molecule', printCONNECT = False):
    from .. import fileproc
    lpdb = fileproc.PDB_DF()
    lpdb.read_file(inPath)

    # Clean existing names (if already numbered)
    atom_names = lpdb.data['HETATM']['atom_name']
    clean_atoms = fp.clean_numbers(atom_names)

    # Sort atoms in the dataframe
    hetatm = lpdb.data['HETATM']
    hetatm['atom_name'] = clean_atoms
    hetatm_sorted = hetatm.sort_values(by=['atom_name', 'atom_id'],ascending = [True, True])
    sorted_atoms = hetatm_sorted['atom_name']

    # print(hetatm.head())

    # Replace atom names by numbered names
    asymbol, acounts = np.unique(sorted_atoms, return_counts = True)
    fixed_names = fp.make_names(asymbol,acounts)
    hetatm_sorted['atom_name'] = fixed_names

    # Replace res names
    # res_names = fp.set_res(res_code, hetatm_sorted)
    # hetatm_sorted['res_name'] = res_names

    # Save pdb file
    lpdb.data['MOLECULE'] = mol_title
    lpdb.data['HETATM'] = hetatm_sorted

    lpdb.write_file(outPath, resname = res_code, print_connect = printCONNECT)
