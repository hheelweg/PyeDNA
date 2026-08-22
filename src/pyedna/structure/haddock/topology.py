"""Prepare dye topology artifacts for HADDOCK."""

from pathlib import Path
import csv
import re
import shutil
import subprocess

import MDAnalysis as mda
import numpy as np
from scipy.optimize import linear_sum_assignment


def _make_cns_resname(definition, registry):
    """
    Generate deterministic unique CNS residue names.
    """

    if definition in registry:
        return registry[definition]

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = abs(hash(definition))

    code = (alphabet[(value // 26**3) % 26]
            + alphabet[(value // 26**2) % 26]
            + alphabet[(value // 26) % 26]
            + alphabet[value % 26])

    if code in registry.values():
        raise ValueError(f"CNS residue hash collision: {definition} -> {code}")

    registry[definition] = code
    return code


def _get_mol2_charge(mol2, tolerance=0.05):
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


def _get_elements(universe):
    try:
        return np.asarray(universe.atoms.elements)
    except (AttributeError, mda.exceptions.NoDataError):
        from MDAnalysis.topology.guessers import guess_types
        return np.asarray(guess_types(universe.atoms.names))


def _write_mapping(original_mol2, haddock_pdb, output_csv, max_distance=0.1):
    original = mda.Universe(original_mol2)
    haddock = mda.Universe(haddock_pdb)

    if len(original.atoms) != len(haddock.atoms):
        raise ValueError(f"{haddock_pdb}: atom counts differ ({len(original.atoms)} vs {len(haddock.atoms)})")

    cost = np.linalg.norm(original.atoms.positions[:, None] - haddock.atoms.positions[None], axis=2)
    original_elements, haddock_elements = _get_elements(original), _get_elements(haddock)
    cost[original_elements[:, None] != haddock_elements[None]] = 1e6

    original_idx, haddock_idx = linear_sum_assignment(cost)
    distances = cost[original_idx, haddock_idx]

    if distances.max() > max_distance:
        raise ValueError(f"{haddock_pdb}: poor atom mapping (max displacement {distances.max():.3f} Å)")

    with Path(output_csv).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["haddock_index", "haddock_serial", "haddock_name", "original_index", "original_serial", "original_name",
                         "original_resname", "original_resid", "original_chain", "distance_A"])

        for i, j, distance in sorted(zip(original_idx, haddock_idx, distances), key=lambda x: x[1]):
            original_atom, haddock_atom = original.atoms[i], haddock.atoms[j]
            writer.writerow([j, haddock_atom.id, haddock_atom.name, i, original_atom.id, original_atom.name,
                             original_atom.resname, original_atom.resid, getattr(original_atom, "chainID", ""), f"{distance:.6f}"])

    print(f"Wrote {output_csv} (max displacement {distances.max():.6f} Å)")


def _prepare_dye_topology(instance, workdir, script, resname=None):
    workdir, script = Path(workdir), Path(script)
    haddock_dir = workdir / "haddock"
    instance.set_prepared_paths(workdir)
    working_mol2 = haddock_dir / f"{instance.name}.mol2"

    if not script.exists():
        raise FileNotFoundError(f"Missing topology script: {script}")

    charge = _get_mol2_charge(instance.definition.mol2)
    #resname = instance.definition.name[:3].upper()
    if resname is None:
        resname = _make_cns_resname(instance.definition.name)

    if not re.fullmatch(r"[A-Za-z0-9]{1,4}", resname):
        raise ValueError(f"{instance.definition.name}: invalid RESNAME {resname!r}")

    haddock_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(instance.definition.mol2, working_mol2)

    try:
        subprocess.run(["bash", str(script), instance.name, str(charge), resname, instance.segid],
                       cwd=haddock_dir, check=True)
    finally:
        working_mol2.unlink(missing_ok=True)

    pdb, top, par = instance.pdb, instance.top, instance.par
    attach = instance.attach

    missing = [str(path) for path in (pdb, top, par) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{instance.name}: missing generated files: {missing}")

    if instance.definition.attach is not None:
        shutil.copy2(instance.definition.attach, attach)
    else:
        instance.definition.write_attachment(attach)

    instance.charge = charge
    instance.resname = resname

    _write_mapping(instance.definition.mol2, instance.pdb, instance.mapping)

    print(f"{instance.name}: charge={charge:+d}, resname={resname}, segid={instance.segid}")
    return instance


def _prepare_dye_topologies(instances, workdir, script):
    resname_registry = {}

    for instance in instances:
        resname = _make_cns_resname(
            instance.definition.name,
            resname_registry,
        )

        _prepare_dye_topology(
            instance,
            workdir,
            script,
            resname=resname,
        )

    # optional: write mapping file
    mapping_file = Path(workdir) / "haddock" / "resname_mapping.csv"

    with mapping_file.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["definition", "cns_resname"])

        for definition, resname in resname_registry.items():
            writer.writerow([definition, resname])

    print(f"Wrote {mapping_file}")

    return instances


def _combine_ligand_topologies(instances, workdir):
    workdir = Path(workdir)
    haddock_dir = workdir / "haddock"
    top_out = haddock_dir / "dyes_haddock.top"
    par_out = haddock_dir / "dyes_haddock.par"

    unique = {}
    for instance in instances:
        unique.setdefault(instance.definition.name, instance)

    top_files = [instance.top for instance in unique.values()]
    par_files = [instance.par for instance in unique.values()]

    missing = [str(path) for path in top_files + par_files if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing ligand topology/parameter files: {missing}")

    tops = [Path(path).read_text().strip() for path in top_files]
    pars = [Path(path).read_text().strip() for path in par_files]

    residues = []
    for path, text in zip(top_files, tops):
        found = re.findall(r"(?im)^\s*RESI(?:DUE)?\s+(\S+)", text)
        if not found:
            raise ValueError(f"{path}: no CNS residue definition found")
        residues.extend(found)

    duplicates = sorted({residue for residue in residues if residues.count(residue) > 1})
    if duplicates:
        raise ValueError(f"Duplicate CNS residue names: {duplicates}")

    top_out.write_text("\n\n".join(
        f"! ===== {name} topology =====\n{text}"
        for name, text in zip(unique, tops)
    ) + "\n")

    par_out.write_text("\n\n".join(
        f"! ===== {name} parameters =====\n{text}"
        for name, text in zip(unique, pars)
    ) + "\n")

    print(f"Wrote {top_out}")
    print(f"Wrote {par_out}")

    return top_out, par_out
