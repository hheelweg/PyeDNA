from pathlib import Path
import re
import shutil
import subprocess


def get_mol2_charge(mol2, tolerance=0.05):
    in_atoms, charge = False, 0.0

    for line in Path(mol2).read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>BOND"):
            break
        if in_atoms and line.strip():
            charge += float(line.split()[8])

    integer_charge = round(charge)
    if abs(charge - integer_charge) > tolerance:
        raise ValueError(f"{mol2}: partial charges sum to {charge:.6f}, not sufficiently close to an integer")

    return integer_charge


def prepare_dye_topology(instance, workdir, script):
    workdir, script = Path(workdir), Path(script)
    haddock_dir = workdir / "haddock"
    instance_dir = haddock_dir / instance.name
    working_mol2 = haddock_dir / f"{instance.name}.mol2"

    if not script.exists():
        raise FileNotFoundError(f"Missing topology script: {script}")

    charge = get_mol2_charge(instance.definition.mol2)
    resname = instance.definition.name[:3].upper()

    if not re.fullmatch(r"[A-Za-z0-9]{1,3}", resname):
        raise ValueError(f"{instance.definition.name}: invalid RESNAME {resname!r}")

    haddock_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(instance.definition.mol2, working_mol2)

    try:
        subprocess.run(["bash", str(script), instance.name, str(charge), resname, instance.segid], cwd=haddock_dir, check=True)
    finally:
        working_mol2.unlink(missing_ok=True)

    pdb = instance_dir / f"{instance.name}_haddock.pdb"
    top = instance_dir / f"{instance.name}_haddock.top"
    par = instance_dir / f"{instance.name}_haddock.par"

    missing = [str(path) for path in (pdb, top, par) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{instance.name}: missing generated files: {missing}")

    shutil.copy2(instance.definition.attach, instance_dir / f"{instance.name}.attach")

    print(f"{instance.name}: charge={charge:+d}, resname={resname}, segid={instance.segid}")

    return pdb, top, par

def prepare_dye_topologies(instances, workdir, script):
    return {instance.name: prepare_dye_topology(instance, workdir, script) for instance in instances}

