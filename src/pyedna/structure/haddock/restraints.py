"""Prepare DNA inputs and bond restraints for HADDOCK."""

from pathlib import Path
import csv

import pandas as pd


def _prepare_dna_for_haddock(dna_pdb, instances, workdir):
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


def _write_bond_restraints(instances, dna_pdb, output="haddock/bond_restraint.tbl",
                          bond_output="haddock/bonds.csv", dna_segid="A",
                          target=1.5, lower_tol=0.2, upper_tol=0.2):
    dna_pdb, output, bond_output = map(Path, (dna_pdb, output, bond_output))

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

        attachment = instance.definition.read_attachment()
        mapping = pd.read_csv(instance.mapping)
        haddock_names = {}

        for end, atom in attachment.items():
            match = mapping[
                (mapping["original_resname"].astype(str) == atom.resname)
                & (mapping["original_resid"].astype(int) == atom.resid)
                & (mapping["original_name"].astype(str) == atom.atom)
            ]

            if len(match) != 1:
                raise ValueError(
                    f"{instance.name} {end}: expected one mapping for "
                    f"{atom.resname} {atom.resid} {atom.atom}, found {len(match)}"
                )

            haddock_names[end] = str(match.iloc[0]["haddock_name"])

        docking_data.append({
            "instance": instance,
            "start": min(instance.residues),
            "end": max(instance.residues),
            "attach5": attachment["5'"],
            "attach3": attachment["3'"],
            "haddock5": haddock_names["5'"],
            "haddock3": haddock_names["3'"],
        })

    ordered = sorted(docking_data, key=lambda x: x["start"])
    blocks, bonds = [], []

    for i, current in enumerate(ordered):
        instance = current["instance"]
        previous = ordered[i - 1] if i else None
        following = ordered[i + 1] if i + 1 < len(ordered) else None

        adjacent_previous = previous is not None and previous["end"] + 1 == current["start"]
        adjacent_following = following is not None and current["end"] + 1 == following["start"]

        if adjacent_previous:
            previous_instance = previous["instance"]

            blocks.append(
                f"! {previous_instance.name} 3' to {instance.name} 5'\n"
                f"assign (segid {previous_instance.segid} and resid 1 and name {previous['haddock3']})\n"
                f"       (segid {instance.segid} and resid 1 and name {current['haddock5']})\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

            bonds.append({
                "left_type": "dye", "left_instance": previous_instance.name,
                "left_resname": previous["attach3"].resname,
                "left_resid": previous["attach3"].resid,
                "left_atom": previous["attach3"].atom,
                "right_type": "dye", "right_instance": instance.name,
                "right_resname": current["attach5"].resname,
                "right_resid": current["attach5"].resid,
                "right_atom": current["attach5"].atom,
            })

        else:
            left = current["start"] - 1

            if (left, "O3'") not in dna_atoms:
                raise ValueError(
                    f"{instance.name}: DNA atom resid {left} name \"O3'\" not found in {dna_pdb}"
                )

            blocks.append(
                f"! DNA {left} to {instance.name} 5'\n"
                f"assign (segid {dna_segid} and resid {left} and name O3')\n"
                f"       (segid {instance.segid} and resid 1 and name {current['haddock5']})\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

            bonds.append({
                "left_type": "dna", "left_instance": "", "left_resname": "",
                "left_resid": left, "left_atom": "O3'",
                "right_type": "dye", "right_instance": instance.name,
                "right_resname": current["attach5"].resname,
                "right_resid": current["attach5"].resid,
                "right_atom": current["attach5"].atom,
            })

        if not adjacent_following:
            right = current["end"] + 1

            if (right, "P") not in dna_atoms:
                raise ValueError(
                    f"{instance.name}: DNA atom resid {right} name 'P' not found in {dna_pdb}"
                )

            blocks.append(
                f"! {instance.name} 3' to DNA {right}\n"
                f"assign (segid {instance.segid} and resid 1 and name {current['haddock3']})\n"
                f"       (segid {dna_segid} and resid {right} and name P)\n"
                f"       {target} {lower_tol} {upper_tol}"
            )

            bonds.append({
                "left_type": "dye", "left_instance": instance.name,
                "left_resname": current["attach3"].resname,
                "left_resid": current["attach3"].resid,
                "left_atom": current["attach3"].atom,
                "right_type": "dna", "right_instance": "", "right_resname": "",
                "right_resid": right, "right_atom": "P",
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(blocks) + "\n")
    pd.DataFrame(bonds).to_csv(bond_output, index=False)

    print(f"Wrote {output}")
    print(f"Wrote {bond_output}")

    return output, bond_output
