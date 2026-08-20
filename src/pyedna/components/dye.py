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
    optimize_rdkit_geometry,
    read_ac_atom_names,
    run_two_stage_resp,
    write_attach_file,
    write_xyz,
)

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class DyeDefinition:
    """Represent and parameterize a capped dye definition."""

    def __init__(
        self,
        name,
        code,
        core_smiles,
        cap_smiles,
        core_targets,
        formal_charge=None,
        amber=None,
        output=None,
        qm=None,
        config=None,
    ):
        """Store dye core, cap attachment targets, and workflow settings."""
        self.name = name
        self.code = code
        self.core_smiles = core_smiles
        self.cap_smiles = cap_smiles
        self.core_targets = core_targets
        self.formal_charge = formal_charge
        self.amber = amber or AmberSettings()
        self.output = output or OutputSettings()
        self.qm = qm or QMSettings()
        self.config = config or {}
        self.mol = None
        self.core_map_ids = set()
        self.core_atom_indices = set()
        self.cap_map_ids = set()

    @property
    def residue_name(self):
        """Return the final uncapped dye residue name."""
        return self.code or self.name

    @property
    def capped_formal_charge(self):
        """Return the formal charge of the full capped dye molecule."""
        from rdkit import Chem

        if self.mol is None:
            self.attach_caps()

        return Chem.GetFormalCharge(self.mol)

    @classmethod
    def from_file(cls, filename):
        """Load a dye definition from a TOML configuration file."""
        path = Path(filename)

        with path.open("rb") as f:
            config = tomllib.load(f)

        dye = config["dye"]
        core = config["core"]
        caps = config["caps"]

        return cls(
            name=dye["name"],
            code=dye.get("code"),
            core_smiles=core["smiles"],
            cap_smiles=caps["smiles"],
            core_targets=caps["core_targets"],
            formal_charge=core.get("formal_charge"),
            amber=AmberSettings.from_config(config.get("amber", {})),
            output=OutputSettings.from_config(config.get("output", {})),
            qm=QMSettings.from_config(config.get("qm", {})),
            config=config,
        )

    def validate(self):
        """Validate dye core, cap, and explicit attachment atom maps."""
        from rdkit import Chem

        core = Chem.MolFromSmiles(self.core_smiles)
        if core is None:
            raise ValueError(f"Could not parse dye core '{self.name}'.")

        cap = Chem.MolFromSmiles(self.cap_smiles)
        if cap is None:
            raise ValueError("Could not parse cap fragment.")
        if cap.GetNumAtoms() != 1:
            raise ValueError("Only single-atom caps are supported for now.")

        self.core_map_ids = {
            atom.GetAtomMapNum()
            for atom in core.GetAtoms()
            if atom.GetAtomMapNum()
        }
        self.core_atom_indices = set(range(core.GetNumAtoms()))
        if not self.core_map_ids:
            raise ValueError("Dye core SMILES must use atom-map IDs.")
        if len(self.core_targets) != len(set(self.core_targets)):
            raise ValueError("Cap target atom-map IDs must be unique.")

        for target in self.core_targets:
            if target not in self.core_map_ids:
                raise ValueError(f"Cap target {target} does not exist.")

        if self.formal_charge is None:
            self.formal_charge = Chem.GetFormalCharge(core)

        self.attach_caps()
        if not self.core_resp_indices():
            raise ValueError("Dye core RESP group is empty.")

    def attach_caps(self):
        """Attach cap atoms to mapped core atoms."""
        from rdkit import Chem

        if self.mol is not None and self.cap_map_ids:
            return self.cap_map_ids

        core = Chem.MolFromSmiles(self.core_smiles)
        cap = Chem.MolFromSmiles(self.cap_smiles)
        self.core_map_ids = {
            atom.GetAtomMapNum()
            for atom in core.GetAtoms()
            if atom.GetAtomMapNum()
        }
        self.core_atom_indices = set(range(core.GetNumAtoms()))

        rw = Chem.RWMol(core)
        next_map = max(max(self.core_map_ids) + 1, 1000)
        cap_atom = cap.GetAtomWithIdx(0)
        cap_maps = set()

        for target in self.core_targets:
            target_idx = next(
                atom.GetIdx()
                for atom in rw.GetAtoms()
                if atom.GetAtomMapNum() == target
            )
            atom = Chem.Atom(cap_atom)
            atom.SetAtomMapNum(next_map)
            cap_idx = rw.AddAtom(atom)
            rw.AddBond(target_idx, cap_idx, Chem.BondType.SINGLE)
            cap_maps.add(next_map)
            next_map += 1

        self.cap_map_ids = cap_maps
        self.mol = rw.GetMol()
        self.mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(self.mol)

        return self.cap_map_ids

    def generate_conformer(self, output_file=None):
        """Generate and write capped dye 3D SDF/PDB structures."""
        from rdkit import Chem

        if self.mol is None:
            self.attach_caps()

        self.mol = embed_rdkit_conformer(self.mol, "3D embedding failed.")
        output_file = Path(output_file or f"{self.name}.sdf")

        Chem.MolToMolFile(self.mol, str(output_file))
        Chem.MolToPDBFile(self.mol, str(output_file.with_suffix(".pdb")))

        return output_file

    def _write_xyz(self, atoms, coords, filename, comment=""):
        """Write atom symbols and coordinates to an XYZ file."""
        write_xyz(atoms, coords, filename, comment)

    def optimize_geometry(self, output_dir=None):
        """Optimize capped dye geometry with PySCF and write XYZ/SDF outputs."""
        if self.mol is None or self.mol.GetNumConformers() == 0:
            raise RuntimeError("Generate a 3D conformer before geometry optimization.")

        # Use [qm.geometry] in dye.toml to lower the basis for debug runs.
        optimized_file, self.mol = optimize_rdkit_geometry(
            self.mol,
            self.name,
            self.capped_formal_charge,
            self.qm,
            output_dir,
            "RDKit starting geometry",
        )
        return optimized_file

    def compute_resp_esp(self, xyz_file, output_file=None):
        """Compute and write the electrostatic potential used by RESP."""
        return compute_resp_esp_from_xyz(
            xyz_file,
            output_file,
            charge=self.capped_formal_charge,
        )

    def core_resp_indices(self):
        """Return RESP atom indices for core atoms and core hydrogens."""
        from rdkit import Chem

        if self.mol is None:
            self.attach_caps()

        mol = Chem.AddHs(self.mol)
        indices = []

        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                if atom.GetIdx() in self.core_atom_indices:
                    indices.append(atom.GetIdx() + 1)
                continue

            parent = atom.GetNeighbors()[0]
            if parent.GetIdx() in self.core_atom_indices:
                indices.append(atom.GetIdx() + 1)

        return sorted(indices)

    def attachment_indices(self):
        """Return RESP atom indices for mapped dye-linker attachment atoms."""
        if self.mol is None:
            self.attach_caps()

        indices = {}
        for atom in self.mol.GetAtoms():
            map_id = atom.GetAtomMapNum()
            if map_id in self.core_targets:
                indices[map_id] = atom.GetIdx() + 1

        missing = set(self.core_targets) - set(indices)
        if missing:
            raise ValueError(
                "Could not resolve dye attachment atom maps: "
                f"{sorted(missing)}"
            )

        return [indices[map_id] for map_id in self.core_targets]

    def _read_ac_atom_names(self, ac_file):
        """Return RESP atom index -> atom name from an Amber .ac file."""
        return read_ac_atom_names(ac_file)

    def generate_charges(self, workdir=None):
        """Run Amber RESP charge fitting and write the full capped dye mol2."""
        workdir = Path(workdir or Path.cwd())
        fit_dir = workdir / self.output.work_subdir
        fit_dir.mkdir(exist_ok=True)

        optimized = workdir / "qm_opt" / f"{self.name}_opt.sdf"
        esp = workdir / "qm_opt" / f"{self.name}.esp"

        ac_file = self._generate_ac(optimized, fit_dir)
        restraint_file = self.write_resp_restraints(
            ac_file,
            fit_dir / "resp_constraints.txt",
        )

        resp1_in, resp2_in = self._generate_resp_inputs(
            ac_file,
            fit_dir,
            restraint_file,
        )

        self._run_resp(resp1_in, resp2_in, esp, fit_dir)

        return self._generate_mol2(ac_file, fit_dir)

    def _generate_ac(self, sdf_file, output_dir):
        """Generate an Amber AC file from the optimized capped SDF structure."""
        return generate_ac(
            self.name,
            sdf_file,
            output_dir,
            self.amber,
            self.capped_formal_charge,
        )

    def write_resp_restraints(self, ac_file, output_file):
        """Write one core-only GROUP charge constraint for dye RESP."""
        ac_names = self._read_ac_atom_names(ac_file)
        core_atoms = self.core_resp_indices()

        if not core_atoms:
            raise ValueError("Dye core RESP group is empty.")

        with Path(output_file).open("w") as f:
            f.write(f"GROUP {len(core_atoms)} {self.formal_charge:.6f}\n")
            for idx in core_atoms:
                f.write(f"ATOM {idx} {ac_names[idx]}\n")

        return output_file

    def _generate_resp_inputs(self, ac_file, output_dir, restraint_file=None):
        """Generate stage-one and stage-two RESP input files."""
        return generate_resp_inputs(ac_file, output_dir, restraint_file)

    def _run_resp(self, resp1_in, resp2_in, esp_file, output_dir):
        """Run the two-stage Amber RESP charge fit."""
        return run_two_stage_resp(resp1_in, resp2_in, esp_file, output_dir)

    def _generate_mol2(self, ac_file, output_dir):
        """Generate the charged full capped dye mol2 file from RESP output."""
        return generate_resp_mol2(self.name, ac_file, output_dir, self.amber)

    def extract_residue_mol2(self, mol2_file, output_file):
        """Write the uncapped dye residue mol2 from the full capped mol2."""
        keep_atoms = set(self.core_resp_indices())
        residue_name = self.residue_name
        return extract_mol2_subset(mol2_file, output_file, keep_atoms, residue_name)

    def write_attach_file(self, mol2_file, atom_names, output_file):
        """Write dye-linker attachment metadata for the final dye mol2."""
        records = [("LINKER", atom_name) for atom_name in atom_names]
        return write_attach_file(output_file, records, mol2_file)

    def mol2_charge(self, mol2_file):
        """Return the sum of partial charges in a mol2 ATOM section."""
        return mol2_charge(mol2_file)

    def generate_residue_template(self, mol2_file, output_dir):
        """Generate final uncapped dye mol2/frcmod files."""
        output_dir = Path(output_dir)
        mol2, name_map = extract_mol2_subset(
            mol2_file,
            output_dir / f"{self.residue_name}.mol2",
            set(self.core_resp_indices()),
            self.residue_name,
            return_mapping=True,
        )
        frcmod = self.generate_frcmod(mol2, output_dir / f"{self.residue_name}.frcmod")
        charge = self.mol2_charge(mol2)
        attach_names = []
        for idx in self.attachment_indices():
            if idx not in name_map:
                raise ValueError(f"Attachment atom index {idx} is absent from {mol2}")
            attach_names.append(name_map[idx])
        attach = self.write_attach_file(
            mol2,
            attach_names,
            output_dir / f"{self.residue_name}.attach",
        )

        if abs(charge - self.formal_charge) > 0.05:
            raise ValueError(
                f"{mol2}: charge {charge:.6f} differs from target "
                f"{self.formal_charge:.6f}"
            )

        return {"mol2": mol2, "frcmod": frcmod, "attach": attach, "charge": charge}

    def generate_frcmod(self, mol2_file, output_file):
        """Generate a frcmod file for an uncapped dye mol2."""
        return generate_frcmod(mol2_file, output_file, self.amber)

    def cleanup_outputs(self, workdir=None):
        """Remove configured temporary files from the dye workflow."""
        return cleanup_outputs(
            self.name,
            self.output,
            workdir,
            extra_scratch=[f"{self.name}.pdb"],
        )
