from dataclasses import dataclass
from pathlib import Path

from .parameterization import (
    AmberSettings,
    OutputSettings,
    QMSettings,
    cleanup_outputs,
    compute_resp_esp_from_xyz,
    embed_rdkit_conformer,
    extract_mol2_subset,
    generate_ac,
    generate_frcmod,
    generate_resp_inputs,
    generate_resp_mol2,
    mol2_charge,
    optimize_classical_geometry,
    optimize_rdkit_geometry,
    read_ac_atom_names,
    resolve_output_directory,
    run_two_stage_resp,
    write_attach_file,
    write_qin,
    write_xyz,
)

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass(frozen=True)
class ChargeRestraints:
    """RESP charge-restraint settings from the linker TOML file."""

    forcefield: str

    @classmethod
    def from_config(cls, config):
        """Create RESP charge restraints from the TOML restraint table."""
        return cls(forcefield=config["forcefield"])


class LinkerDefinition:
    """Represent and parameterize a capped linker definition."""

    DEFAULT_RESIDUE_NAMES = {"three_prime": "L03", "five_prime": "L05"}

    def __init__(
        self,
        name,
        code,
        dye_cap,
        core,
        dna_cap,
        boundaries,
        amber=None,
        charge_restraints=None,
        output=None,
        qm=None,
        config=None,
    ):
        """Store linker fragments, partition boundaries, and workflow settings."""
        self.name = name
        self.code = code
        self.dye_cap = dye_cap
        self.core = core
        self.dna_cap = dna_cap
        self.boundaries = boundaries
        self.amber = amber or AmberSettings()
        self.charge_restraints = charge_restraints
        self.output = output or OutputSettings()
        self.qm = qm or QMSettings()
        self.config = config or {}
        self.mol = None

    @property
    def smiles(self):
        """Return the full capped-linker SMILES string."""
        return self.dye_cap + self.core + self.dna_cap

    @property
    def residue_names(self):
        """Return residue names for the 3' and 5' linker templates."""
        if self.code:
            return {"three_prime": f"{self.code}3", "five_prime": f"{self.code}5"}

        return self.DEFAULT_RESIDUE_NAMES

    def output_directory(self, cwd=None):
        """Return the workflow output directory for this linker."""
        return resolve_output_directory(
            self.output,
            "linker",
            self.code,
            self.amber.forcefield,
            self.charge_restraints.forcefield,
            cwd=cwd,
        )

    @classmethod
    def from_file(cls, filename):
        """Load a linker definition from a TOML configuration file."""
        path = Path(filename)

        with path.open("rb") as f:
            config = tomllib.load(f)

        component = config.get("component", config.get("linker"))
        if component is None:
            raise ValueError(f"{path}: missing [component] section")
        if component.get("type", "linker") != "linker":
            raise ValueError(f"{path}: component.type must be 'linker'")

        structure = config.get("structure")
        if structure is not None:
            smiles = structure
            # Boundary definitions remain explicit: they define residue partitioning.
            boundaries = structure["boundaries"]
        else:
            smiles = config["smiles"]
            boundaries = config["boundaries"]

        parameterization = config.get("parameterization", {})
        amber_config = dict(config.get("amber", {}))
        amber_config.update(parameterization)
        charge_method = parameterization.get("charge_method", "resp")
        if charge_method != "resp":
            raise ValueError("Only RESP charge fitting is supported.")
        restraints = parameterization.get(
            "restraints",
            config.get("charges", {}).get("restraints", {}),
        )
        if not restraints:
            raise ValueError(f"{path}: parameterization.restraints must be specified")

        return cls(
            name=component["name"],
            code=component.get("code"),
            dye_cap=smiles["dye_cap"],
            core=smiles["core"],
            dna_cap=smiles["dna_cap"],
            boundaries=boundaries,
            amber=AmberSettings.from_config(amber_config),
            charge_restraints=ChargeRestraints.from_config(restraints),
            output=OutputSettings.from_config(config.get("output", {})),
            qm=QMSettings.from_config(config.get("qm", {})),
            config=config,
        )

    def validate(self):
        """Validate the mapped SMILES and all configured boundary bonds."""
        from rdkit import Chem

        self.mol = Chem.MolFromSmiles(self.smiles)

        if self.mol is None:
            raise ValueError(f"Could not parse SMILES for linker '{self.name}'.")

        map_ids = [atom.GetAtomMapNum() for atom in self.mol.GetAtoms()]

        if any(idx == 0 for idx in map_ids):
            raise ValueError("All linker atoms must have an atom-map ID.")

        if len(map_ids) != len(set(map_ids)):
            raise ValueError("Atom-map IDs must be unique.")

        for variant, boundaries in self.boundaries.items():
            for side, boundary in boundaries.items():
                self._validate_boundary(variant, side, boundary)

    def _atom_from_map(self, map_id):
        """Return the RDKit atom with a given atom-map ID."""
        for atom in self.mol.GetAtoms():
            if atom.GetAtomMapNum() == map_id:
                return atom

        raise ValueError(f"Atom-map ID {map_id} does not exist.")

    def _validate_boundary(self, variant, side, boundary):
        """Validate that one boundary is a bonded mapped atom pair."""
        if len(boundary) != 2:
            raise ValueError(
                f"{variant}.{side} boundary must contain exactly two atom-map IDs."
            )

        atom1 = self._atom_from_map(boundary[0])
        atom2 = self._atom_from_map(boundary[1])

        if self.mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx()) is None:
            raise ValueError(
                f"{variant}.{side} boundary {boundary} does not define a bonded atom pair."
            )

    @property
    def formal_charge(self):
        """Return the formal charge of the full capped linker molecule."""
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        return Chem.GetFormalCharge(self.mol)

    def summary(self):
        """Return a compact summary of the parsed linker molecule."""
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        return {
            "name": self.name,
            "smiles": self.smiles,
            "atoms": self.mol.GetNumAtoms(),
            "formal_charge": self.formal_charge,
            "fragments": len(Chem.GetMolFrags(self.mol)),
        }

    def generate_conformer(self, output_file=None):
        """Generate and write an RDKit 3D conformer for the full linker."""
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        self.mol = embed_rdkit_conformer(
            self.mol,
            f"Could not generate 3D conformer for '{self.name}'.",
        )
        output_file = Path(output_file or f"{self.name}.sdf")
        writer = Chem.SDWriter(str(output_file))
        writer.write(self.mol)
        writer.close()

        return output_file

    def _mapped_neighbors(self):
        """Return molecular graph using atom-map IDs."""
        graph = {}

        for atom in self.mol.GetAtoms():
            map_id = atom.GetAtomMapNum()
            if map_id:
                graph[map_id] = set()

        for bond in self.mol.GetBonds():
            atom1 = bond.GetBeginAtom().GetAtomMapNum()
            atom2 = bond.GetEndAtom().GetAtomMapNum()

            if atom1 and atom2:
                graph[atom1].add(atom2)
                graph[atom2].add(atom1)

        return graph

    def _connected_component(self, graph, start):
        """Return all mapped atoms connected to start."""
        visited = set()
        stack = [start]

        while stack:
            atom = stack.pop()

            if atom in visited:
                continue

            visited.add(atom)
            stack.extend(graph[atom] - visited)

        return visited

    def partition(self, boundary_name):
        """Partition atoms including implicit hydrogen assignments."""
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        boundaries = self.boundaries[boundary_name]

        dye_boundary = boundaries["dye"]
        dna_boundary = boundaries["dna"]

        graph = self._mapped_neighbors()
        for atom1, atom2 in (dye_boundary, dna_boundary):
            graph[atom1].remove(atom2)
            graph[atom2].remove(atom1)

        dye_cap = self._connected_component(graph, dye_boundary[0])
        residue = self._connected_component(graph, dye_boundary[1])

        if dna_boundary[0] in residue:
            dna_cap = self._connected_component(graph, dna_boundary[1])
        else:
            dna_cap = self._connected_component(graph, dna_boundary[0])

        partitions = {
            "dye_cap": set(dye_cap),
            "residue": set(residue),
            "dna_cap": set(dna_cap),
        }

        atom_partition = {}
        for name, atoms in partitions.items():
            for atom in atoms:
                atom_partition[atom] = name

        expanded = {name: set(atoms) for name, atoms in partitions.items()}
        mol_h = Chem.AddHs(self.mol)

        for atom in mol_h.GetAtoms():
            if atom.GetSymbol() != "H":
                continue
            parent = atom.GetNeighbors()[0]
            parent_map = parent.GetAtomMapNum()

            if parent_map in atom_partition:
                expanded[atom_partition[parent_map]].add(atom.GetIdx() + 1)

        return {name: sorted(atoms) for name, atoms in expanded.items()}

    def _write_xyz(self, atoms, coords, filename, comment=""):
        """Write atom symbols and coordinates to an XYZ file."""
        write_xyz(atoms, coords, filename, comment)

    def build_pyscf_molecule(self):
        """Build a PySCF molecule from the current RDKit conformer."""
        from pyscf import gto

        if self.mol is None:
            raise RuntimeError("No RDKit molecule available.")

        conf = self.mol.GetConformer()

        atoms = [
            [
                atom.GetSymbol(),
                tuple(conf.GetAtomPosition(atom.GetIdx())),
            ]
            for atom in self.mol.GetAtoms()
        ]

        mol = gto.M(
            atom=atoms,
            basis="6-31g(d)",
            charge=self.formal_charge,
            spin=0,
            unit="Angstrom",
            verbose=4,
        )

        return mol

    def compute_resp_esp(self, xyz_file, output_file=None):
        """Compute and write the electrostatic potential used by RESP."""
        return compute_resp_esp_from_xyz(
            xyz_file,
            output_file,
            charge=self.formal_charge,
        )

    def optimize_geometry(self, output_dir=None):
        """Optimize the linker geometry with PySCF and write XYZ/SDF outputs."""
        if self.mol is None or self.mol.GetNumConformers() == 0:
            raise RuntimeError("Generate a 3D conformer before geometry optimization.")

        initial_comment = "RDKit starting geometry"
        if self.qm.classical_preopt:
            self.mol = optimize_classical_geometry(
                self.mol,
                num_confs=self.qm.classical_conformers,
            )
            initial_comment = "RDKit/MMFF-UFF relaxed starting geometry"

        optimized_file, self.mol = optimize_rdkit_geometry(
            self.mol,
            self.name,
            self.formal_charge,
            self.qm,
            output_dir,
            initial_comment,
        )
        return optimized_file

    def _hydrogen_neighbors(self, atom_map_id):
        """Return RESP indices of hydrogens bonded to a mapped atom."""
        from rdkit import Chem

        if self.mol is None:
            raise RuntimeError("No RDKit molecule available.")

        mol = Chem.AddHs(self.mol)
        atom = None

        for candidate in mol.GetAtoms():
            if candidate.GetAtomMapNum() == atom_map_id:
                atom = candidate
                break

        if atom is None:
            raise ValueError(f"Atom-map ID {atom_map_id} does not exist.")

        return [
            neighbor.GetIdx() + 1
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "H"
        ]

    def _ol15_inference_error(self, message):
        """Raise a consistent OL15 inference error."""
        raise ValueError(
            "DNA-cap topology cannot be automatically mapped to OL15: "
            f"{message}"
        )

    def infer_ol15_charge_map(self):
        """Infer RESP atom indices to OL15 atom names from DNA-cap topology."""
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        # SMILES atom maps define topology; OL15 atom names are inferred here.
        cap_mol = Chem.MolFromSmiles(self.dna_cap)
        if cap_mol is None:
            self._ol15_inference_error("could not parse DNA-cap SMILES")

        cap_maps = {
            atom.GetAtomMapNum()
            for atom in cap_mol.GetAtoms()
            if atom.GetAtomMapNum()
        }
        p_atoms = [
            atom for atom in self.mol.GetAtoms()
            if atom.GetSymbol() == "P" and atom.GetAtomMapNum() in cap_maps
        ]

        if len(p_atoms) != 1:
            self._ol15_inference_error(
                f"expected exactly one phosphorus, found {len(p_atoms)}"
            )

        phosphorus = p_atoms[0]
        oxygens = [
            atom for atom in phosphorus.GetNeighbors()
            if atom.GetSymbol() == "O"
        ]

        if len(oxygens) != 4:
            self._ol15_inference_error(
                f"expected four oxygen neighbors around phosphorus, found {len(oxygens)}"
            )

        mapping = {phosphorus.GetAtomMapNum(): "P"}
        terminal_carbon = None
        ester_oxygen_names = []

        for oxygen in oxygens:
            bond = self.mol.GetBondBetweenAtoms(phosphorus.GetIdx(), oxygen.GetIdx())
            carbon_neighbors = [
                atom for atom in oxygen.GetNeighbors()
                if atom.GetSymbol() == "C"
            ]

            if bond.GetBondType() == Chem.BondType.DOUBLE:
                mapping[oxygen.GetAtomMapNum()] = "OP1"
            elif oxygen.GetFormalCharge() == -1:
                mapping[oxygen.GetAtomMapNum()] = "OP2"
            elif carbon_neighbors:
                carbon = carbon_neighbors[0]
                if carbon.GetAtomMapNum() in cap_maps:
                    mapping[oxygen.GetAtomMapNum()] = "O3'"
                    mapping[carbon.GetAtomMapNum()] = "C5'"
                    terminal_carbon = carbon
                else:
                    mapping[oxygen.GetAtomMapNum()] = "O5'"
                ester_oxygen_names.append(mapping[oxygen.GetAtomMapNum()])

        required = {"P", "OP1", "OP2", "O3'", "O5'", "C5'"}
        missing = required - set(mapping.values())
        if missing:
            self._ol15_inference_error(
                f"missing inferred OL15 atoms: {', '.join(sorted(missing))}"
            )

        if sorted(ester_oxygen_names) != ["O3'", "O5'"]:
            self._ol15_inference_error(
                "expected two phosphate ester oxygens connected to carbon"
            )

        hydrogens = self._hydrogen_neighbors(terminal_carbon.GetAtomMapNum())
        if len(hydrogens) != 3:
            self._ol15_inference_error(
                f"expected three hydrogens on terminal C5' carbon, found {len(hydrogens)}"
            )

        for hydrogen in hydrogens:
            mapping[hydrogen] = "H5'"

        return mapping

    def load_charges(self):
        """Load reference forcefield charges for RESP restraints."""
        import os
        import re

        restraints = self.charge_restraints
        forcefield = restraints.forcefield

        # (1) Read the supported Amber reference library.
        if forcefield.upper() == "OL15":
            lib_file = Path(os.environ["AMBERHOME"]) / "dat/leap/lib/DNA.OL15.lib"
        else:
            raise ValueError(f"Unsupported charge forcefield: {forcefield}")

        text = lib_file.read_text()

        match = re.search(
            r'!entry\.DA\.unit\.atoms table.*?(?=!entry\.)',
            text,
            flags=re.S,
        )

        if match is None:
            raise RuntimeError("Could not find atom charge table.")

        ff_charges = {}

        # (2) Parse atom names and charges from the DA residue table.
        for line in match.group().splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                atom = fields[0].strip('"')
                ff_charges[atom] = float(fields[-1])
            except ValueError:
                continue

        charges = {}

        # (3) Infer OL15 atom names from the mapped DNA-cap topology.
        charge_map = self.infer_ol15_charge_map()

        # (4) Assign fixed RESP charges from inferred OL15 atom names.
        for resp_idx, ff_atom in charge_map.items():
            if ff_atom not in ff_charges:
                raise KeyError(f"{ff_atom} not found in {forcefield}")

            charges[resp_idx] = ff_charges[ff_atom]

        return charges

    def _read_ac_atom_names(self, ac_file):
        """Return RESP atom index -> atom name from an Amber .ac file."""
        return read_ac_atom_names(ac_file)

    def generate_charges(self, workdir=None):
        """Run Amber RESP charge fitting and write the full-linker mol2 file."""
        workdir = Path(workdir or Path.cwd())

        fit_dir = workdir / self.output.work_subdir
        fit_dir.mkdir(exist_ok=True)

        optimized = workdir / "qm_opt" / f"{self.name}_opt.sdf"
        esp = workdir / "qm_opt" / f"{self.name}.esp"

        # (1) Generate Amber AC input and charge restraints.
        ac_file = self._generate_ac(optimized, fit_dir)
        charges = self.load_charges()

        restraint_file = self.write_resp_restraints(
            charges,
            ac_file,
            fit_dir / "resp_constraints.txt",
        )

        # (2) Generate RESP inputs and run the two-stage RESP fit.
        resp1_in, resp2_in = self._generate_resp_inputs(
            ac_file,
            fit_dir,
            restraint_file,
        )

        self._run_resp(resp1_in, resp2_in, esp, fit_dir, charges)

        # (3) Convert RESP charges back into an Amber mol2.
        return self._generate_mol2(ac_file, fit_dir)

    def _generate_ac(self, sdf_file, output_dir):
        """Generate an Amber AC file from the optimized SDF structure."""
        return generate_ac(self.name, sdf_file, output_dir, self.amber, self.formal_charge)

    def write_resp_restraints(self, charges, ac_file, output_file):
        """Write fixed charge and group charge restraints for RESP."""
        ac_names = self._read_ac_atom_names(ac_file)

        with Path(output_file).open("w") as f:
            for idx, charge in sorted(charges.items()):
                f.write(f"CHARGE {charge:.6f} {idx} {ac_names[idx]}\n")

            parts = self.partition("five_prime")
            residue = parts["residue"]

            p3 = self.partition("three_prime")
            p5 = self.partition("five_prime")

            extra_3prime = set(p3["residue"]) - set(p5["residue"])
            q_extra = sum(charges[i] for i in extra_3prime)

            target_charge = (self.formal_charge - q_extra) / 2

            f.write(f"\nGROUP {len(residue)} {target_charge:.6f}\n")
            for idx in residue:
                f.write(f"ATOM {idx} {ac_names[idx]}\n")

        return output_file

    def _generate_resp_inputs(self, ac_file, output_dir, restraint_file=None):
        """Generate stage-one and stage-two RESP input files."""
        return generate_resp_inputs(ac_file, output_dir, restraint_file)

    def _write_qin(self, ac_file, charges, output_file):
        """Write initial RESP charges for restrained atom indices."""
        return write_qin(ac_file, charges, output_file)

    def _run_resp(self, resp1_in, resp2_in, esp_file, output_dir, charges):
        """Run the two-stage Amber RESP charge fit."""
        qin = output_dir / "qin"
        self._write_qin(output_dir / f"{self.name}.ac", charges, qin)
        return run_two_stage_resp(resp1_in, resp2_in, esp_file, output_dir, qin)

    def _generate_mol2(self, ac_file, output_dir):
        """Generate the charged full-linker mol2 file from RESP output."""
        return generate_resp_mol2(self.name, ac_file, output_dir, self.amber)

    def extract_residue_mol2(self, mol2_file, variant, output_file):
        """Write one linker residue mol2 by slicing the full-linker mol2."""
        partition = self.partition(variant)
        keep_atoms = set(partition["residue"])
        residue_name = self.residue_names[variant]
        return extract_mol2_subset(mol2_file, output_file, keep_atoms, residue_name)

    def connection_indices(self, variant):
        """Return linker 3'/5' connection atom indices for one residue variant."""
        partition = self.partition(variant)
        residue = set(partition["residue"])
        boundaries = self.boundaries[variant]

        def residue_atom(boundary):
            atoms = [idx for idx in boundary if idx in residue]
            if len(atoms) != 1:
                raise ValueError(
                    f"{variant} boundary {boundary} does not identify one residue atom."
                )
            return atoms[0]

        dye_atom = residue_atom(boundaries["dye"])
        dna_atom = residue_atom(boundaries["dna"])

        if variant == "three_prime":
            return {"3CONNECT": dna_atom, "5CONNECT": dye_atom}
        if variant == "five_prime":
            return {"3CONNECT": dye_atom, "5CONNECT": dna_atom}

        raise ValueError(f"Unsupported linker residue variant: {variant}")

    def write_attach_file(self, mol2_file, records, output_file):
        """Write linker attachment metadata for the final residue mol2."""
        return write_attach_file(output_file, records, mol2_file)

    def mol2_charge(self, mol2_file):
        """Return the sum of partial charges in a mol2 ATOM section."""
        return mol2_charge(mol2_file)

    def generate_residue_templates(self, mol2_file, output_dir):
        """Generate linker residue mol2/frcmod files and report charge sums."""
        output_dir = Path(output_dir)
        templates = {}

        for variant, residue_name in self.residue_names.items():
            partition = self.partition(variant)
            mol2, name_map = extract_mol2_subset(
                mol2_file,
                output_dir / f"{residue_name}.mol2",
                set(partition["residue"]),
                residue_name,
                return_mapping=True,
            )
            frcmod = self.generate_frcmod(
                mol2,
                output_dir / f"{residue_name}.frcmod",
            )
            records = []
            for label, idx in self.connection_indices(variant).items():
                if idx not in name_map:
                    raise ValueError(f"{label} atom index {idx} is absent from {mol2}")
                records.append((label, name_map[idx]))
            attach = self.write_attach_file(
                mol2,
                records,
                output_dir / f"{residue_name}.attach",
            )
            templates[residue_name] = {
                "mol2": mol2,
                "frcmod": frcmod,
                "attach": attach,
                "charge": self.mol2_charge(mol2),
            }

        return templates

    def cleanup_outputs(self, workdir=None):
        """Remove configured temporary files from the linker workflow."""
        return cleanup_outputs(
            self.name,
            self.output,
            workdir,
            extra_scratch=["qin"],
        )

    def generate_frcmod(self, mol2_file, output_file):
        """Generate a frcmod file for a linker residue mol2."""
        return generate_frcmod(mol2_file, output_file, self.amber)
