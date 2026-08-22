from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess


@dataclass(frozen=True)
class AmberSettings:
    """AmberTools settings for component parameter generation."""

    forcefield: str = "gaff2"

    @classmethod
    def from_config(cls, config):
        """Create Amber settings from an optional TOML table."""
        return cls(
            forcefield=config.get("amber_forcefield", config.get("forcefield", "gaff2"))
        )


@dataclass(frozen=True)
class OutputSettings:
    """Output directory and cleanup settings for parameterization workflows."""

    directory: str = "cwd"
    work_subdir: str = "resp_fit"
    cleanup: str = "scratch"

    @classmethod
    def from_config(cls, config):
        """Create output settings from an optional TOML table."""
        directory = config.get("directory", "cwd")
        cleanup = config.get("cleanup", "scratch")

        if directory not in {"cwd", "library"}:
            raise ValueError(f"Unsupported output directory mode: {directory}")
        # Cleanup modes:
        # none: keep all generated files for maximum debugging/reproducibility.
        # scratch: remove known transient AmberTools files and initial structures.
        # minimal: remove scratch files plus regenerable RESP setup intermediates.
        # library: keep only final mol2/frcmod/attach files in a library output.
        if cleanup not in {"none", "scratch", "minimal", "library"}:
            raise ValueError(f"Unsupported output cleanup mode: {cleanup}")
        if cleanup == "library" and directory != "library":
            raise ValueError("cleanup='library' requires output.directory='library'")

        return cls(
            directory=directory,
            work_subdir=config.get("work_subdir", "resp_fit"),
            cleanup=cleanup,
        )


def resolve_output_directory(
    output,
    component,
    code,
    amber_forcefield,
    dna_forcefield=None,
    cwd=None,
):
    """Return the working output directory for cwd or library generation."""
    cwd = Path(cwd or Path.cwd())
    if output.directory == "cwd":
        return cwd

    if not code:
        raise ValueError(f"{component}: code is required for library output")

    if component == "dye":
        root_name = "DYE_DIR"
        parts = [code, amber_forcefield]
    elif component == "linker":
        root_name = "LNK_DIR"
        if not dna_forcefield:
            raise ValueError("linker: DNA restraint forcefield is required")
        parts = [code, amber_forcefield, dna_forcefield]
    else:
        raise ValueError(f"Unsupported library component: {component}")

    root = os.environ.get(root_name)
    if not root:
        raise EnvironmentError(f"{root_name} is not set")

    output_dir = Path(root).joinpath(*parts)
    if output_dir.exists():
        raise FileExistsError(f"Library output already exists: {output_dir}")

    output_dir.mkdir(parents=True)
    return output_dir


@dataclass(frozen=True)
class QMSettings:
    """QM settings for geometry optimization."""

    basis: str = "6-31g(d)"
    maxsteps: int = 100
    classical_preopt: bool = False
    classical_conformers: int = 20

    @classmethod
    def from_config(cls, config):
        """Create QM settings from an optional TOML table."""
        geometry = config.get("geometry", config)
        classical_conformers = geometry.get("classical_conformers", 20)
        if classical_conformers < 1:
            raise ValueError("qm.classical_conformers must be at least 1")
        return cls(
            basis=geometry.get("basis", "6-31g(d)"),
            maxsteps=geometry.get("maxsteps", 100),
            classical_preopt=geometry.get("classical_preopt", False),
            classical_conformers=classical_conformers,
        )


def embed_rdkit_conformer(mol, error_message):
    """Add hydrogens and generate an RDKit 3D conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.randomSeed = 42

    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(error_message)

    return mol


def optimize_classical_geometry(mol, num_confs=20):
    """Generate and optimize multiple RDKit conformers, returning the lowest-energy one."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if mol is None:
        raise RuntimeError("No molecule provided.")

    params = AllChem.ETKDGv3()
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.randomSeed = 42

    mol.RemoveAllConformers()

    conf_ids = list(
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            params=params,
        )
    )

    if not conf_ids:
        raise RuntimeError("Could not generate conformers.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=500,
        )
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(
            mol,
            maxIters=500,
        )

    energies = [energy for _, energy in results]
    best_id = conf_ids[energies.index(min(energies))]

    best_conf = mol.GetConformer(best_id)

    new_conf = Chem.Conformer(mol.GetNumAtoms())

    for i in range(mol.GetNumAtoms()):
        new_conf.SetAtomPosition(
            i,
            best_conf.GetAtomPosition(i),
        )

    mol.RemoveAllConformers()
    mol.AddConformer(new_conf)

    return mol


def write_xyz(atoms, coords, filename, comment=""):
    """Write atom symbols and coordinates to an XYZ file."""
    with Path(filename).open("w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n")


def optimize_rdkit_geometry(
    mol,
    name,
    charge,
    qm,
    output_dir=None,
    initial_comment="RDKit starting geometry",
):
    """Optimize an RDKit conformer with PySCF/geomeTRIC and write XYZ/SDF outputs."""
    from pyscf import gto, scf
    from pyscf.geomopt.geometric_solver import optimize
    from rdkit import Chem

    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError("Generate a 3D conformer before geometry optimization.")

    output_dir = Path(output_dir or Path.cwd() / "qm_opt")
    output_dir.mkdir(parents=True, exist_ok=True)

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [tuple(conf.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms()]

    initial_file = output_dir / f"{name}_initial.xyz"
    write_xyz(symbols, coords, initial_file, initial_comment)

    pyscf_mol = gto.M(
        atom=list(zip(symbols, coords)),
        basis=qm.basis,
        charge=charge,
        spin=0,
        unit="Angstrom",
        verbose=4,
    )
    mf = scf.RHF(pyscf_mol).density_fit()

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

    mol_opt = optimize(mf, maxsteps=qm.maxsteps)
    optimized_file = output_dir / f"{name}_opt.xyz"
    opt_coords = mol_opt.atom_coords(unit="Angstrom")

    write_xyz(
        [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)],
        opt_coords,
        optimized_file,
        f"RHF/{qm.basis} optimized geometry",
    )

    mol_opt_rdkit = Chem.Mol(mol)
    conf = mol_opt_rdkit.GetConformer()
    for i, xyz in enumerate(opt_coords):
        conf.SetAtomPosition(i, xyz)

    optimized_sdf = output_dir / f"{name}_opt.sdf"
    writer = Chem.SDWriter(str(optimized_sdf))
    writer.write(mol_opt_rdkit)
    writer.close()

    print(f"Optimized SDF: {optimized_sdf}")

    return optimized_file, mol_opt_rdkit


def compute_resp_esp_from_xyz(xyz_file, output_file=None, charge=0):
    """Compute and write an Amber RESP electrostatic-potential file."""
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
        charge=charge,
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


def generate_ac(name, sdf_file, output_dir, amber, charge):
    """Generate an Amber AC file from an optimized SDF structure."""
    output_dir = Path(output_dir).resolve()
    sdf_file = Path(sdf_file).resolve()
    ac_file = output_dir / f"{name}.ac"

    cmd = [
        "antechamber",
        "-i", str(sdf_file),
        "-fi", "sdf",
        "-o", str(ac_file),
        "-fo", "ac",
        "-at", amber.forcefield,
        "-nc", str(charge),
    ]
    subprocess.run(cmd, check=True, cwd=output_dir)

    return ac_file


def read_ac_atom_names(ac_file):
    """Return RESP atom index -> atom name from an Amber .ac file."""
    names = {}

    with Path(ac_file).open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            fields = line.split()
            names[int(fields[1])] = fields[2]

    return names


def generate_resp_inputs(ac_file, output_dir, restraint_file=None):
    """Generate stage-one and stage-two RESP input files."""
    output_dir = Path(output_dir).resolve()
    ac_file = Path(ac_file).resolve()
    restraint_file = Path(restraint_file).resolve() if restraint_file else None
    resp1 = output_dir / "resp1.in"
    resp2 = output_dir / "resp2.in"

    cmd1 = ["respgen", "-i", str(ac_file), "-o", str(resp1), "-f", "resp1"]
    if restraint_file:
        cmd1 += ["-a", str(restraint_file)]
    subprocess.run(cmd1, check=True, cwd=output_dir)

    cmd2 = ["respgen", "-i", str(ac_file), "-o", str(resp2), "-f", "resp2"]
    if restraint_file:
        cmd2 += ["-a", str(restraint_file)]
    subprocess.run(cmd2, check=True, cwd=output_dir)

    return resp1, resp2


def write_qin(ac_file, charges, output_file):
    """Write initial RESP charges for restrained atom indices."""
    natoms = sum(1 for line in Path(ac_file).open() if line.startswith("ATOM"))
    q = [0.0] * natoms

    for idx, charge in charges.items():
        q[idx - 1] = charge

    with Path(output_file).open("w") as f:
        for i in range(0, natoms, 8):
            f.write("".join(f"{x:10.6f}" for x in q[i:i + 8]) + "\n")

    return output_file


def run_two_stage_resp(resp1_in, resp2_in, esp_file, output_dir, qin_file=None):
    """Run the two-stage Amber RESP charge fit."""
    output_dir = Path(output_dir)
    resp1_charges = output_dir / "resp1_charges"
    resp2_charges = output_dir / "resp2_charges"
    esp_rel = Path("../qm_opt") / Path(esp_file).name

    cmd1 = [
        "resp",
        "-O",
        "-i", Path(resp1_in).name,
        "-e", str(esp_rel),
        "-o", "resp1.out",
        "-t", "resp1_charges",
    ]
    if qin_file is not None:
        cmd1[6:6] = ["-q", Path(qin_file).name]

    print("RESP1 command:")
    print(" ".join(cmd1), flush=True)

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
        "-i", Path(resp2_in).name,
        "-e", str(esp_rel),
        "-q", "resp1_charges",
        "-o", "resp2.out",
        "-t", "resp2_charges",
    ]
    subprocess.run(cmd2, cwd=output_dir, check=True)

    return resp2_charges


def generate_resp_mol2(name, ac_file, output_dir, amber):
    """Generate a charged mol2 file from RESP output."""
    output_dir = Path(output_dir).resolve()
    ac_file = Path(ac_file).resolve()
    mol2 = output_dir / f"{name}.mol2"

    cmd = [
        "antechamber",
        "-i", str(ac_file),
        "-fi", "ac",
        "-o", str(mol2),
        "-fo", "mol2",
        "-c", "rc",
        "-cf", str(output_dir / "resp2_charges"),
        "-at", amber.forcefield,
    ]
    subprocess.run(cmd, check=True, cwd=output_dir)

    return mol2


def extract_mol2_subset(
    mol2_file,
    output_file,
    keep_atoms,
    residue_name,
    return_mapping=False,
):
    """Write a single-residue mol2 subset with clean atom and bond numbering."""
    lines = Path(mol2_file).read_text().splitlines()
    keep_atoms = set(keep_atoms)
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
    old_to_name = {}
    new_atoms = []
    for line in atoms:
        fields = line.split()
        idx = int(fields[0])
        if idx in keep_atoms:
            new_idx = len(new_atoms) + 1
            old_to_new[idx] = new_idx
            old_to_name[idx] = fields[1]
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
        f.write(f"{len(new_atoms):5d}{len(new_bonds):6d}{1:6d}{0:6d}{0:6d}\n")
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

    output_file = Path(output_file)
    if return_mapping:
        return output_file, old_to_name

    return output_file


def mol2_atom_names(mol2_file):
    """Return atom names present in a mol2 ATOM section."""
    names = set()
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
        if len(fields) >= 2:
            names.add(fields[1])

    return names


def write_attach_file(output_file, records, mol2_file=None):
    """Write attachment metadata and validate names against a final mol2."""
    output_file = Path(output_file)
    records = [(label, atom_name) for label, atom_name in records]

    if mol2_file is not None:
        names = mol2_atom_names(mol2_file)
        for _, atom_name in records:
            if atom_name not in names:
                raise ValueError(
                    f"{output_file}: atom '{atom_name}' is not present in {mol2_file}"
                )

    with output_file.open("w") as f:
        for label, atom_name in records:
            f.write(f"{label} {atom_name}\n")

    return output_file


def mol2_charge(mol2_file):
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


def generate_frcmod(mol2_file, output_file, amber):
    """Generate a frcmod file for a mol2 residue template."""
    mol2_file = Path(mol2_file).resolve()
    output_file = Path(output_file).resolve()
    cmd = [
        "parmchk2",
        "-i", str(mol2_file),
        "-f", "mol2",
        "-o", str(output_file),
        "-s", amber.forcefield,
    ]
    subprocess.run(cmd, check=True, cwd=output_file.parent)

    return output_file


def cleanup_outputs(name, output, workdir=None, extra_scratch=()):
    """Remove configured temporary files from a parameterization workflow."""
    workdir = Path(workdir or Path.cwd())
    mode = output.cleanup

    if mode == "none":
        return []
    if mode == "library" and output.directory != "library":
        raise ValueError("cleanup='library' is only supported for library output")
    if mode == "library":
        keep_suffixes = {".mol2", ".frcmod", ".attach"}
        removed = []

        for path in workdir.iterdir():
            if path.is_file() and path.suffix in keep_suffixes:
                continue
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(path)
            elif path.is_file():
                path.unlink()
                removed.append(path)

        return removed

    scratch_patterns = [
        "ANTECHAMBER_*",
        "ATOMTYPE.INF",
        "NEWPDB.PDB",
        "PREP.INF",
        "QIN",
        f"{name}.sdf",
        "sqm.*",
    ]
    scratch_patterns.extend(extra_scratch)
    minimal_files = [
        f"{name}.ac",
        "resp1.in",
        "resp2.in",
        "resp1_charges",
    ]

    removed = []
    cleanup_dirs = [workdir, workdir / output.work_subdir]

    for directory in cleanup_dirs:
        if not directory.exists():
            continue
        for pattern in scratch_patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()
                    removed.append(path)

    if mode == "minimal":
        fit_dir = workdir / output.work_subdir
        for filename in minimal_files:
            path = fit_dir / filename
            if path.is_file():
                path.unlink()
                removed.append(path)

    return removed
