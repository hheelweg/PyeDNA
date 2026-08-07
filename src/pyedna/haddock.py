from pathlib import Path
import re
import shutil
import subprocess
import csv
import pandas as pd
import numpy as np
import MDAnalysis as mda
from scipy.optimize import linear_sum_assignment
try:
    import tomllib
except ImportError:
    import tomli as tomllib


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


def get_elements(universe):
    try:
        return np.asarray(universe.atoms.elements)
    except (AttributeError, mda.exceptions.NoDataError):
        from MDAnalysis.topology.guessers import guess_types
        return np.asarray(guess_types(universe.atoms.names))


def write_mapping(original_mol2, haddock_pdb, output_csv, max_distance=0.1):
    original = mda.Universe(original_mol2)
    haddock = mda.Universe(haddock_pdb)

    if len(original.atoms) != len(haddock.atoms):
        raise ValueError(f"{haddock_pdb}: atom counts differ ({len(original.atoms)} vs {len(haddock.atoms)})")

    cost = np.linalg.norm(original.atoms.positions[:, None] - haddock.atoms.positions[None], axis=2)
    original_elements, haddock_elements = get_elements(original), get_elements(haddock)
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
        subprocess.run(["bash", str(script), instance.name, str(charge), resname, instance.segid],
                       cwd=haddock_dir, check=True)
    finally:
        working_mol2.unlink(missing_ok=True)

    pdb = instance_dir / f"{instance.name}_haddock.pdb"
    top = instance_dir / f"{instance.name}_haddock.top"
    par = instance_dir / f"{instance.name}_haddock.par"
    attach = instance_dir / f"{instance.name}.attach"
    mapping = instance_dir / f"{instance.name}_mapping.csv"

    missing = [str(path) for path in (pdb, top, par) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{instance.name}: missing generated files: {missing}")

    shutil.copy2(instance.definition.attach, attach)

    instance.charge = charge
    instance.resname = resname
    instance.directory = instance_dir
    instance.pdb = pdb
    instance.top = top
    instance.par = par
    instance.attach = attach
    instance.mapping = mapping

    write_mapping(instance.definition.mol2, instance.pdb, instance.mapping)

    print(f"{instance.name}: charge={charge:+d}, resname={resname}, segid={instance.segid}")
    return instance


def prepare_dye_topologies(instances, workdir, script):
    for instance in instances:
        prepare_dye_topology(instance, workdir, script)
    return instances

def combine_ligand_topologies(instances, workdir):
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

def prepare_dna_for_haddock(dna_pdb, instances, workdir):
    dna_pdb, workdir = Path(dna_pdb), Path(workdir)
    haddock_dir = workdir / "haddock"
    output_pdb = haddock_dir / f"{dna_pdb.stem}_haddock.pdb"
    bonding_csv = haddock_dir / f"{dna_pdb.stem}_bonding.csv"

    if not dna_pdb.exists():
        raise FileNotFoundError(f"DNA PDB not found: {dna_pdb}")

    remove_resids = {resid for instance in instances for resid in instance.residues}
    haddock_dir.mkdir(parents=True, exist_ok=True)

    kept, ter_after, last_resid = [], [], None

    for line in dna_pdb.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            resid = int(line[22:26])
            if resid in remove_resids:
                continue
            kept.append(line)
            last_resid = resid
        elif line.startswith("TER"):
            if last_resid is not None:
                kept.append(line)
                ter_after.append(last_resid)
        else:
            kept.append(line)

    output_pdb.write_text("\n".join(kept) + "\n")

    with bonding_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ter_after_resid"])
        writer.writerows([[resid] for resid in ter_after])

    print(f"Wrote {output_pdb}")
    print(f"Wrote {bonding_csv}")

    return output_pdb, bonding_csv


def read_attachment(instance):
    data = {}

    for line in Path(instance.attach).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        end, resname, resid, atom = line.split()
        if end not in {"5'", "3'"}:
            raise ValueError(f"{instance.attach}: unknown end {end!r}")

        data[end] = {"resname": resname, "resid": int(resid), "atom": atom}

    if set(data) != {"5'", "3'"}:
        raise ValueError(f"{instance.attach}: must define exactly 5' and 3'")

    return data


def write_bond_restraints(instances, dna_pdb, output="haddock/bond_restraint.tbl",
                          dna_segid="A", target=1.5, lower_tol=0.2, upper_tol=0.2):
    dna_pdb, output = Path(dna_pdb), Path(output)

    if not dna_pdb.exists():
        raise FileNotFoundError(f"Missing DNA PDB: {dna_pdb}")

    dna_atoms = {
        (int(line[22:26]), line[12:16].strip())
        for line in dna_pdb.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and (not line[72:76].strip() or line[72:76].strip() == dna_segid)
    }

    docking_data = []

    for instance in instances:
        if instance.mapping is None or not Path(instance.mapping).exists():
            raise FileNotFoundError(f"{instance.name}: missing mapping file: {instance.mapping}")
        if instance.attach is None or not Path(instance.attach).exists():
            raise FileNotFoundError(f"{instance.name}: missing attachment file: {instance.attach}")

        attachment = read_attachment(instance)
        mapping = pd.read_csv(instance.mapping)
        haddock_names = {}

        for end, atom in attachment.items():
            match = mapping[
                (mapping["original_resname"].astype(str) == atom["resname"])
                & (mapping["original_resid"].astype(int) == atom["resid"])
                & (mapping["original_name"].astype(str) == atom["atom"])
            ]

            if len(match) != 1:
                raise ValueError(
                    f"{instance.name} {end}: expected one mapping for "
                    f"{atom['resname']} {atom['resid']} {atom['atom']}, found {len(match)}"
                )

            haddock_names[end] = str(match.iloc[0]["haddock_name"])

        docking_data.append({
            "instance": instance,
            "start": min(instance.residues),
            "end": max(instance.residues),
            "atom5": haddock_names["5'"],
            "atom3": haddock_names["3'"],
        })

    ordered = sorted(docking_data, key=lambda x: x["start"])
    blocks = []

    for i, current in enumerate(ordered):
        instance = current["instance"]
        current_segid = instance.segid
        current_atom5 = current["atom5"]
        current_atom3 = current["atom3"]

        previous = ordered[i - 1] if i else None
        adjacent_previous = previous is not None and previous["end"] + 1 == current["start"]

        if adjacent_previous:
            previous_instance = previous["instance"]
            previous_segid = previous_instance.segid
            previous_atom3 = previous["atom3"]

            blocks.append(
                f"! {previous_instance.name} 3' to {instance.name} 5'\n"
                f"assign (segid {previous_segid} and resid 1 and name {previous_atom3})\n"
                f"       (segid {current_segid} and resid 1 and name {current_atom5})\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

        else:
            left = current["start"] - 1

            if (left, "O3'") not in dna_atoms:
                raise ValueError(
                    f"{instance.name}: DNA atom resid {left} name \"O3'\" "
                    f"not found in {dna_pdb}"
                )

            blocks.append(
                f"! DNA {left} to {instance.name} 5'\n"
                f"assign (segid {dna_segid} and resid {left} and name O3')\n"
                f"       (segid {current_segid} and resid 1 and name {current_atom5})\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

        following = ordered[i + 1] if i + 1 < len(ordered) else None
        adjacent_following = following is not None and current["end"] + 1 == following["start"]

        if not adjacent_following:
            right = current["end"] + 1

            if (right, "P") not in dna_atoms:
                raise ValueError(
                    f"{instance.name}: DNA atom resid {right} name 'P' "
                    f"not found in {dna_pdb}"
                )

            blocks.append(
                f"! {instance.name} 3' to DNA {right}\n"
                f"assign (segid {current_segid} and resid 1 and name {current_atom3})\n"
                f"       (segid {dna_segid} and resid {right} and name P)\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n")

    print(f"Wrote {output}")
    return output

# Haddock3 .cfg file

DEFAULT_DOCKING_CONFIG = {
    "run_dir": "haddock/run",
    "mode": "local",
    "ncores": 32,
    "clean": False,
    "postprocess": False,

    "delenph": True,
    "autohis": False,

    "rigidbody_sampling": 10,
    "rigidbody_ntrials": 10,
    "rigidbody_randremoval": False,
    "rigidbody_unambig_scale": 800,
    "rigidbody_inter_rigid": 0.001,
    "rigidbody_elecflag": True,
    "rigidbody_w_air": 9999.0,
    "rigidbody_w_vdw": 1.0,
    "rigidbody_w_elec": 1.0,
    "rigidbody_w_desolv": 0.0,
    "rigidbody_w_bsa": 0.0,
    "rigidbody_w_dist": 9999.0,
    "rigidbody_cmrest": False,
    "rigidbody_surfrest": False,
    "rigidbody_ranair": False,
    "rigidbody_rigidtrans": True,

    "seletop_select": 10,

    "flexref_randremoval": False,
    "flexref_unambig_hot": 1000,
    "flexref_unambig_cool1": 1000,
    "flexref_unambig_cool2": 1000,
    "flexref_unambig_cool3": 1000,
    "flexref_w_air": 9999.0,
    "flexref_w_vdw": 1.0,
    "flexref_w_elec": 1.0,
    "flexref_w_desolv": 0.0,
    "flexref_w_bsa": 0.0,
    "flexref_mdsteps_rigid": 0,
    "flexref_mdsteps_cool1": 0,
    "flexref_mdsteps_cool2": 2000,
    "flexref_mdsteps_cool3": 2000,
    "flexref_dnarest_on": True,
    "flexref_tadfactor": 1,
    "flexref_temp_cool3_init": 300,
    "flexref_elecflag": True,

    "caprieval_allatoms": True,
}

def read_user_docking_config(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing user docking config: {path}")

    sections = tomllib.loads(path.read_text())
    prefixes = {
        "general": "",
        "topoaa": "",
        "rigidbody": "rigidbody_",
        "seletop": "seletop_",
        "flexref": "flexref_",
        "caprieval": "caprieval_",
    }

    values = {}

    for section, params in sections.items():
        if section not in prefixes:
            raise ValueError(f"{path}: unknown section [{section}]")
        if not isinstance(params, dict):
            raise ValueError(f"{path}: [{section}] must contain key-value pairs")

        prefix = prefixes[section]
        values.update({f"{prefix}{key}": value for key, value in params.items()})

    unknown = sorted(set(values) - set(DEFAULT_DOCKING_CONFIG))
    if unknown:
        raise KeyError(f"{path}: unknown configuration parameters: {unknown}")

    return values

def write_docking_config(dna_pdb, instances, top_file, par_file, restraint_file, workdir=".", user_config=None, template=None):
    workdir = Path(workdir)
    output = workdir / "docking_config.cfg"

    if template is None:
        template = Path(__file__).resolve().parent / "data" / "haddock_templates" / "docking_config.cfg"
        print(str(Path))
    else:
        template = Path(template)

    dna_pdb, top_file, par_file, restraint_file = map(Path, (dna_pdb, top_file, par_file, restraint_file))

    required = [template, dna_pdb, top_file, par_file, restraint_file]
    required += [instance.pdb for instance in instances]
    missing = [str(path) for path in required if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required HADDOCK files: {missing}")

    values = dict(DEFAULT_DOCKING_CONFIG)

    if user_config is not None:
        values.update(read_user_docking_config(user_config))

    molecules = [dna_pdb] + [instance.pdb for instance in instances]

    values.update(
        topology_file=str(top_file),
        parameter_file=str(par_file),
        restraint_file=str(restraint_file),
        molecule_lines=",\n".join(f'    "{path}"' for path in molecules),
        flexibility_lines="\n\n".join(
            f'fle_seg_{i} = "{instance.segid}"\nfle_sta_{i} = 1\nfle_end_{i} = 1'
            for i, instance in enumerate(instances, start=1)
        ),
    )

    def format_value(value):
        return str(value).lower() if isinstance(value, bool) else str(value)

    text = template.read_text()
    for key, value in values.items():
        text = text.replace(f"{{{{ {key} }}}}", format_value(value))

    unresolved = sorted(set(re.findall(r"\{\{\s*(.*?)\s*\}\}", text)))
    if unresolved:
        raise KeyError(f"Missing docking template values: {unresolved}")

    output.write_text(text)
    print(f"Wrote {output}")

    return output
