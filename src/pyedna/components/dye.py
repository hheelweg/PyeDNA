from dataclasses import dataclass
from pathlib import Path
import subprocess

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass(frozen=True)
class AmberSettings:
    """AmberTools settings for dye parameter generation."""

    forcefield: str = "gaff2"

    @classmethod
    def from_config(cls, config):
        """Create Amber settings from an optional TOML table."""
        return cls(forcefield=config.get("forcefield", "gaff2"))


@dataclass(frozen=True)
class OutputSettings:
    """Output directory and cleanup settings for the dye workflow."""

    work_subdir: str = "resp_fit"
    cleanup: str = "none"

    @classmethod
    def from_config(cls, config):
        """Create output settings from an optional TOML table."""
        cleanup = config.get("cleanup", "none")

        # Cleanup modes:
        # none: keep all generated files for maximum debugging/reproducibility.
        # scratch: remove known transient AmberTools files and initial SDF/PDB.
        # minimal: remove scratch files plus regenerable RESP setup intermediates.
        if cleanup not in {"none", "scratch", "minimal"}:
            raise ValueError(f"Unsupported output cleanup mode: {cleanup}")

        return cls(
            work_subdir=config.get("work_subdir", "resp_fit"),
            cleanup=cleanup,
        )


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
        from rdkit.Chem import AllChem

        if self.mol is None:
            self.attach_caps()

        mol = Chem.AddHs(self.mol)

        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise RuntimeError("3D embedding failed.")

        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

        self.mol = mol
        output_file = Path(output_file or f"{self.name}.sdf")

        Chem.MolToMolFile(self.mol, str(output_file))
        Chem.MolToPDBFile(self.mol, str(output_file.with_suffix(".pdb")))

        return output_file

    def _write_xyz(self, atoms, coords, filename, comment=""):
        """Write atom symbols and coordinates to an XYZ file."""
        with Path(filename).open("w") as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom, (x, y, z) in zip(atoms, coords):
                f.write(f"{atom:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")

    def optimize_geometry(self, output_dir=None):
        """Optimize capped dye geometry with a temporary debug QM setup."""
        from pyscf import gto, scf
        from pyscf.geomopt.geometric_solver import optimize

        if self.mol is None or self.mol.GetNumConformers() == 0:
            raise RuntimeError("Generate a 3D conformer before geometry optimization.")

        # Temporary/debug QM setup: STO-3G is deliberately cheap and not
        # production-quality for final dye parameterization.
        output_dir = Path(output_dir or Path.cwd() / "qm_opt")
        output_dir.mkdir(parents=True, exist_ok=True)

        conf = self.mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        coords = [
            tuple(conf.GetAtomPosition(atom.GetIdx())) for atom in self.mol.GetAtoms()
        ]

        initial_file = output_dir / f"{self.name}_initial.xyz"
        self._write_xyz(symbols, coords, initial_file, "RDKit starting geometry")

        mol = gto.M(
            atom=list(zip(symbols, coords)),
            basis="sto-3g",
            charge=self.capped_formal_charge,
            spin=0,
            unit="Angstrom",
            verbose=4,
        )
        mf = scf.RHF(mol).density_fit()

        try:
            import cupy as cp
            import gpu4pyscf

            if cp.cuda.runtime.getDeviceCount() > 0:
                mf = mf.to_gpu()
                print("QM backend: GPU4PySCF")
            else:
                print("QM backend: PySCF CPU")
        except (ImportError, RuntimeError):
            print("QM backend: PySCF CPU")

        mol_opt = optimize(mf, maxsteps=50)
        optimized_file = output_dir / f"{self.name}_opt.xyz"
        opt_coords = mol_opt.atom_coords(unit="Angstrom")

        self._write_xyz(
            [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)],
            opt_coords,
            optimized_file,
            "RHF/STO-3G debug optimized geometry",
        )

        from rdkit import Chem

        mol_opt_rdkit = Chem.Mol(self.mol)
        conf = mol_opt_rdkit.GetConformer()
        for i, xyz in enumerate(opt_coords):
            conf.SetAtomPosition(i, xyz)

        optimized_sdf = output_dir / f"{self.name}_opt.sdf"
        writer = Chem.SDWriter(str(optimized_sdf))
        writer.write(mol_opt_rdkit)
        writer.close()

        print(f"Optimized SDF: {optimized_sdf}")

        return optimized_file

    def compute_resp_esp(self, xyz_file, output_file=None):
        """Compute and write the electrostatic potential used by RESP."""
        import cupy as cp
        from pyscf import gto, scf
        from pyscf.data import radii
        from gpu4pyscf.pop import esp
        from gpu4pyscf.gto.int3c1e import int1e_grids
        from gpu4pyscf.lib.cupy_helper import dist_matrix

        xyz_file = Path(xyz_file)
        output_file = Path(output_file or xyz_file.with_suffix(".esp"))

        lines = xyz_file.read_text().splitlines()
        natoms = int(lines[0])
        atoms = []
        for line in lines[2:2 + natoms]:
            e, x, y, z = line.split()[:4]
            atoms.append((e, (float(x), float(y), float(z))))

        mol = gto.M(
            atom=atoms,
            basis="6-31G*",
            charge=self.capped_formal_charge,
            spin=0,
            unit="Angstrom",
        )
        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF calculation did not converge.")

        dm = mf.make_rdm1()
        points = esp.vdw_surface(
            mol,
            scales=[1.4, 1.6, 1.8, 2.0],
            density=1.0 * radii.BOHR**2,
        )

        coords = cp.asarray(mol.atom_coords(unit="B"))
        charges = cp.asarray(mol.atom_charges())
        points_gpu = cp.asarray(points)

        rinv = 1.0 / dist_matrix(coords, points_gpu)
        v_nuc = cp.dot(charges, rinv)
        v_elec = int1e_grids(mol, points, dm=dm, direct_scf_tol=1e-14)
        values = cp.asnumpy(v_nuc - v_elec)

        with output_file.open("w") as f:
            f.write(f"{mol.natm:5d}{len(points):6d}\n")
            for x, y, z in mol.atom_coords(unit="B"):
                f.write(f"{'':17s}{x:16.7E}{y:16.7E}{z:16.7E}\n")
            for value, (x, y, z) in zip(values, points):
                f.write(f" {value:16.7E}{x:16.7E}{y:16.7E}{z:16.7E}\n")

        return output_file

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

    def _read_ac_atom_names(self, ac_file):
        """Return RESP atom index -> atom name from an Amber .ac file."""
        names = {}

        with Path(ac_file).open() as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                fields = line.split()
                names[int(fields[1])] = fields[2]

        return names

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
        ac_file = output_dir / f"{self.name}.ac"

        cmd = [
            "antechamber",
            "-i", str(sdf_file),
            "-fi", "sdf",
            "-o", str(ac_file),
            "-fo", "ac",
            "-at", self.amber.forcefield,
            "-nc", str(self.capped_formal_charge),
        ]
        subprocess.run(cmd, check=True)

        return ac_file

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
        resp1 = output_dir / "resp1.in"
        resp2 = output_dir / "resp2.in"

        cmd1 = ["respgen", "-i", str(ac_file), "-o", str(resp1), "-f", "resp1"]
        if restraint_file:
            cmd1 += ["-a", str(restraint_file)]
        subprocess.run(cmd1, check=True)

        cmd2 = ["respgen", "-i", str(ac_file), "-o", str(resp2), "-f", "resp2"]
        if restraint_file:
            cmd2 += ["-a", str(restraint_file)]
        subprocess.run(cmd2, check=True)

        return resp1, resp2

    def _run_resp(self, resp1_in, resp2_in, esp_file, output_dir):
        """Run the two-stage Amber RESP charge fit."""
        resp1_charges = output_dir / "resp1_charges"
        resp2_charges = output_dir / "resp2_charges"
        esp_rel = Path("../qm_opt") / esp_file.name

        cmd1 = [
            "resp",
            "-O",
            "-i", resp1_in.name,
            "-e", str(esp_rel),
            "-o", "resp1.out",
            "-t", "resp1_charges",
        ]
        result = subprocess.run(cmd1, cwd=output_dir, capture_output=True, text=True)

        print("RESP1 stdout:")
        print(result.stdout)
        print("RESP1 stderr:")
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("RESP1 failed")
        if not resp1_charges.exists():
            raise RuntimeError("RESP1 did not create charges")

        cmd2 = [
            "resp",
            "-O",
            "-i", resp2_in.name,
            "-e", str(esp_rel),
            "-q", "resp1_charges",
            "-o", "resp2.out",
            "-t", "resp2_charges",
        ]
        subprocess.run(cmd2, cwd=output_dir, check=True)

        return resp2_charges

    def _generate_mol2(self, ac_file, output_dir):
        """Generate the charged full capped dye mol2 file from RESP output."""
        mol2 = output_dir / f"{self.name}.mol2"

        cmd = [
            "antechamber",
            "-i", str(ac_file),
            "-fi", "ac",
            "-o", str(mol2),
            "-fo", "mol2",
            "-c", "rc",
            "-cf", str(output_dir / "resp2_charges"),
            "-at", self.amber.forcefield,
        ]
        subprocess.run(cmd, check=True)

        return mol2

    def extract_residue_mol2(self, mol2_file, output_file):
        """Write the uncapped dye residue mol2 from the full capped mol2."""
        keep_atoms = set(self.core_resp_indices())
        residue_name = self.residue_name
        lines = Path(mol2_file).read_text().splitlines()

        atoms = []
        bonds = []
        section = None

        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                section = "atom"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "bond"
                continue
            if line.startswith("@<TRIPOS>"):
                section = None
                continue
            if section == "atom":
                atoms.append(line)
            elif section == "bond":
                bonds.append(line)

        old_to_new = {}
        new_atoms = []

        for line in atoms:
            fields = line.split()
            idx = int(fields[0])
            if idx in keep_atoms:
                new_idx = len(new_atoms) + 1
                old_to_new[idx] = new_idx
                new_atoms.append(
                    {
                        "id": new_idx,
                        "name": fields[1],
                        "x": float(fields[2]),
                        "y": float(fields[3]),
                        "z": float(fields[4]),
                        "type": fields[5],
                        "charge": float(fields[8]),
                    }
                )

        new_bonds = []

        for line in bonds:
            fields = line.split()
            a1 = int(fields[1])
            a2 = int(fields[2])
            btype = fields[3]
            if a1 in old_to_new and a2 in old_to_new:
                new_bonds.append(
                    {
                        "id": len(new_bonds) + 1,
                        "a1": old_to_new[a1],
                        "a2": old_to_new[a2],
                        "type": btype,
                    }
                )

        with Path(output_file).open("w") as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{residue_name}\n")
            f.write(
                f"{len(new_atoms):5d}{len(new_bonds):6d}{1:6d}{0:6d}{0:6d}\n"
            )
            f.write("SMALL\n")
            f.write("rc\n\n")
            f.write("@<TRIPOS>ATOM\n")

            for atom in new_atoms:
                f.write(
                    f"{atom['id']:7d}"
                    f" {atom['name']:<8s}"
                    f"{atom['x']:10.4f}"
                    f"{atom['y']:10.4f}"
                    f"{atom['z']:10.4f}"
                    f" {atom['type']:<8s}"
                    f" {1:2d}"
                    f" {residue_name:<8s}"
                    f"{atom['charge']:12.6f}\n"
                )

            f.write("@<TRIPOS>BOND\n")

            for bond in new_bonds:
                f.write(
                    f"{bond['id']:6d}"
                    f"{bond['a1']:6d}"
                    f"{bond['a2']:6d}"
                    f" {bond['type']:<3s}\n"
                )

            f.write("@<TRIPOS>SUBSTRUCTURE\n")
            f.write(
                f"{1:6d}"
                f" {residue_name:<8s}"
                f"{1:5d}"
                f" TEMP              0 ****  ****    0 ROOT\n"
            )

        return Path(output_file)

    def mol2_charge(self, mol2_file):
        """Return the sum of partial charges in a mol2 ATOM section."""
        charge = 0.0
        in_atoms = False

        for line in Path(mol2_file).read_text().splitlines():
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if line.startswith("@<TRIPOS>"):
                if in_atoms:
                    break
                continue
            if not in_atoms:
                continue

            fields = line.split()
            if len(fields) >= 9:
                charge += float(fields[8])

        return charge

    def generate_residue_template(self, mol2_file, output_dir):
        """Generate final uncapped dye mol2/frcmod files."""
        output_dir = Path(output_dir)
        mol2 = self.extract_residue_mol2(
            mol2_file,
            output_dir / f"{self.residue_name}.mol2",
        )
        frcmod = self.generate_frcmod(mol2, output_dir / f"{self.residue_name}.frcmod")
        charge = self.mol2_charge(mol2)

        if abs(charge - self.formal_charge) > 0.05:
            raise ValueError(
                f"{mol2}: charge {charge:.6f} differs from target "
                f"{self.formal_charge:.6f}"
            )

        return {"mol2": mol2, "frcmod": frcmod, "charge": charge}

    def generate_frcmod(self, mol2_file, output_file):
        """Generate a frcmod file for an uncapped dye mol2."""
        cmd = [
            "parmchk2",
            "-i", str(mol2_file),
            "-f", "mol2",
            "-o", str(output_file),
            "-s", self.amber.forcefield,
        ]
        subprocess.run(cmd, check=True)

        return output_file

    def cleanup_outputs(self, workdir=None):
        """Remove configured temporary files from the dye workflow."""
        workdir = Path(workdir or Path.cwd())
        mode = self.output.cleanup

        if mode == "none":
            return []

        scratch_patterns = [
            "ANTECHAMBER_*",
            "ATOMTYPE.INF",
            "NEWPDB.PDB",
            "PREP.INF",
            "QIN",
            f"{self.name}.sdf",
            f"{self.name}.pdb",
            "sqm.*",
        ]
        minimal_files = [
            f"{self.name}.ac",
            "resp1.in",
            "resp2.in",
            "resp1_charges",
        ]

        removed = []
        cleanup_dirs = [workdir, workdir / self.output.work_subdir]

        for directory in cleanup_dirs:
            if not directory.exists():
                continue
            for pattern in scratch_patterns:
                for path in directory.glob(pattern):
                    if path.is_file():
                        path.unlink()
                        removed.append(path)

        if mode == "minimal":
            fit_dir = workdir / self.output.work_subdir
            for filename in minimal_files:
                path = fit_dir / filename
                if path.is_file():
                    path.unlink()
                    removed.append(path)

        return removed
