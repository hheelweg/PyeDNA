from pathlib import Path
import subprocess

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class LinkerDefinition:

    def __init__(self, name, dye_cap, core, dna_cap, boundaries, config=None):
        self.name = name
        self.dye_cap = dye_cap
        self.core = core
        self.dna_cap = dna_cap
        self.boundaries = boundaries
        self.config = config or {}
        self.mol = None

    @property
    def smiles(self):
        return self.dye_cap + self.core + self.dna_cap

    @classmethod
    def from_file(cls, filename):
        path = Path(filename)

        with path.open("rb") as f:
            config = tomllib.load(f)

        linker = config["linker"]
        smiles = config["smiles"]
        boundaries = config["boundaries"]

        return cls(
                    name=linker["name"],
                    dye_cap=smiles["dye_cap"],
                    core=smiles["core"],
                    dna_cap=smiles["dna_cap"],
                    boundaries=boundaries,
                    config=config,
                )

    def validate(self):
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
        for atom in self.mol.GetAtoms():
            if atom.GetAtomMapNum() == map_id:
                return atom

        raise ValueError(f"Atom-map ID {map_id} does not exist.")

    def _validate_boundary(self, variant, side, boundary):
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
        from rdkit import Chem

        if self.mol is None:
            self.validate()

        return Chem.GetFormalCharge(self.mol)

    def summary(self):
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
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if self.mol is None:
            self.validate()

        mol = Chem.AddHs(self.mol)

        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise RuntimeError(f"Could not generate 3D conformer for '{self.name}'.")

        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)

        self.mol = mol

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

        dye_cap = self._connected_component(
            graph,
            dye_boundary[0]
        )

        residue = self._connected_component(
            graph,
            dye_boundary[1]
        )

        if dna_boundary[0] in residue:
            dna_cap = self._connected_component(
                graph,
                dna_boundary[1]
            )
        else:
            dna_cap = self._connected_component(
                graph,
                dna_boundary[0]
            )

        partitions = {
            "dye_cap": set(dye_cap),
            "residue": set(residue),
            "dna_cap": set(dna_cap),
        }

        # --------------------------------------------------
        # Assign hydrogens to the partition of their parent
        # --------------------------------------------------

        atom_partition = {}

        for name, atoms in partitions.items():
            for atom in atoms:
                atom_partition[atom] = name

        expanded = {
            name: set(atoms)
            for name, atoms in partitions.items()
        }

        mol_h = Chem.AddHs(self.mol)

        for atom in mol_h.GetAtoms():

            if atom.GetSymbol() != "H":
                continue

            parent = atom.GetNeighbors()[0]

            parent_map = parent.GetAtomMapNum()

            if parent_map in atom_partition:
                expanded[
                    atom_partition[parent_map]
                ].add(atom.GetIdx()+1)

        return {
            name: sorted(atoms)
            for name, atoms in expanded.items()
        }

    def _write_xyz(self, atoms, coords, filename, comment=""):
        with Path(filename).open("w") as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"{comment}\n")
            for atom, (x, y, z) in zip(atoms, coords):
                f.write(f"{atom:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")

    def build_pyscf_molecule(self):
        from pyscf import gto

        if self.mol is None:
            raise RuntimeError("No RDKit molecule available.")

        conf = self.mol.GetConformer()

        atoms = [
            [
                atom.GetSymbol(),
                tuple(conf.GetAtomPosition(atom.GetIdx()))
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
        from pathlib import Path
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
        for line in lines[2:2+natoms]:
            e, x, y, z = line.split()[:4]
            atoms.append((e, (float(x), float(y), float(z))))

        mol = gto.M(atom=atoms, basis="6-31G*", charge=self.formal_charge, spin=0, unit="Angstrom")
        mf = scf.RHF(mol).density_fit().to_gpu()
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF calculation did not converge.")

        dm = mf.make_rdm1()
        points = esp.vdw_surface(mol, scales=[1.4, 1.6, 1.8, 2.0], density=1.0*radii.BOHR**2)

        coords = cp.asarray(mol.atom_coords(unit="B"))
        charges = cp.asarray(mol.atom_charges())
        points_gpu = cp.asarray(points)

        rinv = 1.0 / dist_matrix(coords, points_gpu)
        v_nuc = cp.dot(charges, rinv)
        v_elec = int1e_grids(mol, points, dm=dm, direct_scf_tol=1e-14)
        values = cp.asnumpy(v_nuc - v_elec)

        atom_coords = mol.atom_coords(unit="B")

        with output_file.open("w") as f:
            f.write(f"{mol.natm:5d}{len(points):6d}\n")
            for x, y, z in atom_coords:
                f.write(f"{'':17s}{x:16.7E}{y:16.7E}{z:16.7E}\n")
            for value, (x, y, z) in zip(values, points):
                f.write(f" {value:16.7E}{x:16.7E}{y:16.7E}{z:16.7E}\n")

        return output_file

    def optimize_geometry(self, output_dir=None):
        from pyscf import gto, scf
        from pyscf.geomopt.geometric_solver import optimize

        if self.mol is None or self.mol.GetNumConformers() == 0:
            raise RuntimeError("Generate a 3D conformer before geometry optimization.")

        output_dir = Path(output_dir or Path.cwd() / "qm_opt")
        output_dir.mkdir(parents=True, exist_ok=True)

        conf = self.mol.GetConformer()
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        coords = [
            tuple(conf.GetAtomPosition(atom.GetIdx()))
            for atom in self.mol.GetAtoms()
        ]

        initial_file = output_dir / f"{self.name}_initial.xyz"
        self._write_xyz(
            symbols,
            coords,
            initial_file,
            "RDKit/MMFF starting geometry",
        )

        mol = gto.M(
            atom=list(zip(symbols, coords)),
            basis="6-31g(d)",
            charge=self.formal_charge,
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

        mol_opt = optimize(mf, maxsteps=100)

        optimized_file = output_dir / f"{self.name}_opt.xyz"
        opt_coords = mol_opt.atom_coords(unit="Angstrom")

        self._write_xyz(
            [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)],
            opt_coords,
            optimized_file,
            "RHF/6-31G(d) optimized geometry",
        )

        # -------------------------------------------------
        # Write optimized RDKit SDF
        # -------------------------------------------------

        from rdkit import Chem

        mol_opt_rdkit = Chem.Mol(self.mol)

        conf = mol_opt_rdkit.GetConformer()

        for i, xyz in enumerate(opt_coords):
            conf.SetAtomPosition(
                i,
                xyz,
            )

        optimized_sdf = output_dir / f"{self.name}_opt.sdf"

        writer = Chem.SDWriter(str(optimized_sdf))
        writer.write(mol_opt_rdkit)
        writer.close()

        print(f"Optimized SDF: {optimized_sdf}")

        return optimized_file

    def _hydrogen_neighbors(self, resp_index):
        """Return RESP indices of hydrogens bonded to an atom."""

        from rdkit import Chem

        if self.mol is None:
            raise RuntimeError("No RDKit molecule available.")

        mol = Chem.AddHs(self.mol)

        atom = mol.GetAtomWithIdx(resp_index - 1)

        return [
            neighbor.GetIdx() + 1
            for neighbor in atom.GetNeighbors()
            if neighbor.GetSymbol() == "H"
        ]

    def load_charges(self):
        """Load reference forcefield charges for RESP restraints."""

        import os
        import re
        from pathlib import Path

        cfg = self.config["charges"]["restraints"]

        forcefield = cfg["forcefield"]

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

        # heavy atom restraints
        for resp_idx, ff_atom in cfg["atoms"].items():

            resp_idx = int(resp_idx)

            if ff_atom not in ff_charges:
                raise KeyError(
                    f"{ff_atom} not found in {forcefield}"
                )

            charges[resp_idx] = ff_charges[ff_atom]

        # hydrogen restraints inferred from connectivity
        for parent_idx, ff_hydrogen in cfg.get("hydrogens", {}).items():

            parent_idx = int(parent_idx)

            h_indices = self._hydrogen_neighbors(parent_idx)

            if ff_hydrogen not in ff_charges:
                raise KeyError(
                    f"{ff_hydrogen} not found in {forcefield}"
                )

            for resp_idx in h_indices:
                charges[resp_idx] = ff_charges[ff_hydrogen]

        return charges

    def _read_ac_atom_names(self, ac_file):
        """Return RESP atom index -> atom name from an Amber .ac file."""

        names = {}

        with Path(ac_file).open() as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                fields = line.split()

                idx = int(fields[1])
                name = fields[2]

                names[idx] = name

        return names


    def generate_charges(self, workdir=None):
        workdir = Path(workdir or Path.cwd())

        amber_dir = workdir / "amber"
        amber_dir.mkdir(exist_ok=True)

        optimized = workdir / "qm_opt" / f"{self.name}_opt.sdf"
        esp = workdir / "qm_opt" / f"{self.name}.esp"

        ac_file = self._generate_ac(
            optimized,
            amber_dir
        )


        charges = self.load_charges()


        restraint_file = self.write_resp_restraints(
            charges,
            ac_file,
            amber_dir/"resp_constraints.txt"
        )

        resp1_in, resp2_in = self._generate_resp_inputs(
            ac_file,
            amber_dir,
            restraint_file,
        )

        self._run_resp(
            resp1_in,
            resp2_in,
            esp,
            amber_dir
        )

        mol2 = self._generate_mol2(
            ac_file,
            amber_dir
        )

        return mol2


    def _generate_ac(self, sdf_file, output_dir):

        ac_file = output_dir / f"{self.name}.ac"

        cmd = [
            "antechamber",
            "-i", str(sdf_file),
            "-fi", "sdf",
            "-o", str(ac_file),
            "-fo", "ac",
            "-at", "gaff2",
            "-nc", str(self.formal_charge),
        ]

        subprocess.run(
            cmd,
            check=True
        )

        return ac_file


    # def write_resp_restraints(self, charges, ac_file, output_file):


    #     ac_names = self._read_ac_atom_names(ac_file)

    #     print("\nRESP restraint charges:")
    #     for idx, charge in sorted(charges.items()):
    #         print(f"{idx:5d} {ac_names[idx]:5s} {charge: .6f}")

    #     with Path(output_file).open("w") as f:
    #         for idx, charge in sorted(charges.items()):
    #             f.write(
    #                 f"CHARGE {charge:.6f} {idx} {ac_names[idx]}\n"
    #             )

    #     return output_file

    def write_resp_restraints(self, charges, ac_file, output_file):
        ac_names = self._read_ac_atom_names(ac_file)

        with Path(output_file).open("w") as f:
            for idx, charge in sorted(charges.items()):
                f.write(f"CHARGE {charge:.6f} {idx} {ac_names[idx]}\n")

            parts = self.partition("five_prime")
            residue = parts["residue"]
            dna_cap = parts["dna_cap"]

            p3 = self.partition("three_prime")
            p5 = self.partition("five_prime")

            extra_3prime = set(p3["residue"]) - set(p5["residue"])
            q_extra = sum(charges[i] for i in extra_3prime)

            target_charge = (self.formal_charge - q_extra) / 2

            #target_charge = self.formal_charge - sum(charges[i] for i in dna_cap)

            f.write(f"\nGROUP {len(residue)} {target_charge:.6f}\n")
            for idx in residue:
                f.write(f"ATOM {idx} {ac_names[idx]}\n")

        return output_file
    

    def _generate_resp_inputs(self, ac_file, output_dir, restraint_file=None):

        resp1 = output_dir / "resp1.in"
        resp2 = output_dir / "resp2.in"

        cmd1 = [
            "respgen",
            "-i", str(ac_file),
            "-o", str(resp1),
            "-f", "resp1",
        ]

        if restraint_file:
            cmd1 += ["-a", str(restraint_file)]

        subprocess.run(cmd1, check=True)

        cmd2 = [
            "respgen",
            "-i", str(ac_file),
            "-o", str(resp2),
            "-f", "resp2",
        ]

        if restraint_file:
            cmd2 += ["-a", str(restraint_file)]

        subprocess.run(cmd2, check=True)

        return resp1, resp2


    def _write_qin(self, ac_file, charges, output_file):

        natoms = sum(
            1 for line in Path(ac_file).open()
            if line.startswith("ATOM")
        )

        q = [0.0] * natoms

        for idx, charge in charges.items():
            q[idx-1] = charge

        with Path(output_file).open("w") as f:
            for i in range(0, natoms, 8):
                f.write(
                    "".join(f"{x:10.6f}" for x in q[i:i+8])
                    + "\n"
                )

        return output_file

    def _run_resp(self, resp1_in, resp2_in, esp_file, output_dir):

        resp1_charges = output_dir / "resp1_charges"
        resp2_charges = output_dir / "resp2_charges"

        esp_rel = Path("../qm_opt") / esp_file.name

        qin = output_dir / "qin"
        charges = self.load_charges()

        self._write_qin(
            output_dir / f"{self.name}.ac",
            charges,
            qin
        )

        cmd1 = [
            "resp",
            "-O",
            "-i", "resp1.in",
            "-e", str(esp_rel),
            "-q", "qin",
            "-o", "resp1.out",
            "-t", "resp1_charges",
        ]

        print("RESP1 command:")
        print(" ".join(cmd1), flush=True)

        result = subprocess.run(
            cmd1,
            cwd=output_dir,
            capture_output=True,
            text=True,
        )

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
            "-i", "resp2.in",
            "-e", str(esp_rel),
            "-q", "resp1_charges",
            "-o", "resp2.out",
            "-t", "resp2_charges",
        ]

        subprocess.run(
            cmd2,
            cwd=output_dir,
            check=True,
        )

        return resp2_charges

    def _generate_mol2(self, ac_file, output_dir):

        mol2 = output_dir / f"{self.name}.mol2"

        subprocess.run([
            "antechamber",
            "-i", str(ac_file),
            "-fi", "ac",
            "-o", str(mol2),
            "-fo", "mol2",
            "-c", "rc",
            "-cf", str(output_dir/"resp2_charges"),
            "-at", "gaff2",
        ], check=True)

        return mol2


    def extract_residue_mol2(self, mol2_file, variant, output_file):

        from pathlib import Path

        partition = self.partition(variant)
        keep_atoms = set(partition["residue"])

        residue_name = "L03" if variant == "three_prime" else "L05"

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

                new_atoms.append({
                    "id": new_idx,
                    "name": fields[1],
                    "x": float(fields[2]),
                    "y": float(fields[3]),
                    "z": float(fields[4]),
                    "type": fields[5],
                    "charge": float(fields[8]),
                })


        new_bonds = []

        for line in bonds:

            fields = line.split()

            bond_id = int(fields[0])
            a1 = int(fields[1])
            a2 = int(fields[2])
            btype = fields[3]

            if a1 in old_to_new and a2 in old_to_new:

                new_bonds.append({
                    "id": len(new_bonds)+1,
                    "a1": old_to_new[a1],
                    "a2": old_to_new[a2],
                    "type": btype,
                })


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

    def generate_frcmod(self, mol2_file, output_file):
        subprocess.run([
            "parmchk2",
            "-i", str(mol2_file),
            "-f", "mol2",
            "-o", str(output_file),
            "-s", "gaff2",
        ], check=True)

        return output_file

    
    def read_mol2_charges(self, mol2_file):
        """Read atom-map-indexed charges from a mol2 file."""

        charges = {}

        with Path(mol2_file).open() as f:
            lines = f.readlines()

        in_atoms = False

        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue

            if line.startswith("@<TRIPOS>"):
                if in_atoms:
                    break

            if in_atoms:
                fields = line.split()

                if len(fields) < 9:
                    continue

                idx = int(fields[0])
                charge = float(fields[-1])

                charges[idx] = charge

        return charges


    def print_charge_partition(self, mol2_file):

        charges = self.read_mol2_charges(mol2_file)

        print("\nCharge partition")
        print("================")

        total = sum(charges.values())

        print(f"Total molecule: {total: .6f}")

        for name in ["three_prime", "five_prime"]:

            parts = self.partition(name)

            print(f"\n{name}")
            print("-" * len(name))

            residue_charge = 0.0

            for part_name, atoms in parts.items():

                q = sum(charges[i] for i in atoms)

                print(
                    f"{part_name:10s}: {q: .6f}"
                )

                residue_charge += q

            print(
                f"{'sum':10s}: {residue_charge: .6f}"
            )

            print(
                f"{'core only':10s}: "
                f"{sum(charges[i] for i in parts['residue']): .6f}"
            )

    def print_partition_atoms(self):

        print("\nPartition atoms")
        print("================")

        for name in ["three_prime", "five_prime"]:

            print(f"\n{name}")
            print("-" * len(name))

            parts = self.partition(name)

            for part, atoms in parts.items():
                print(f"\n{part}")

                for idx in atoms:
                    print(idx)

    def print_partition_charges(self, mol2_file):

        charges = self.read_mol2_charges(mol2_file)

        print("\nRESP partition charges")
        print("=====================")

        for name in ["three_prime", "five_prime"]:

            print(f"\n{name}")

            parts = self.partition(name)

            for part, atoms in parts.items():

                q = sum(charges[i] for i in atoms)

                print(
                    f"{part:10s} {q: .6f}"
                )
                

    def compare_boundary_difference(self):

        p3 = self.partition("three_prime")
        p5 = self.partition("five_prime")

        print("\nBoundary atom differences")
        print("========================")

        l03_only = set(p3["residue"]) - set(p5["residue"])
        l05_only = set(p5["residue"]) - set(p3["residue"])

        print("3' residue only:")
        print(sorted(l03_only))

        print("\n5' residue only:")
        print(sorted(l05_only))