"""Post-process HADDOCK docking output into final structure files."""

from pathlib import Path
import shutil

import pandas as pd

from ... import fileproc as fp


def _select_best_models(run_dir, output_dir, top=5, structure_name="dna_dyes"):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    flexref_dir = run_dir / "3_flexref"
    capri_file = run_dir / "4_caprieval" / "capri_ss.tsv"

    if not capri_file.exists():
        raise FileNotFoundError(f"Missing CAPRI file: {capri_file}")

    df = pd.read_csv(capri_file, sep="\t")
    columns = ["vdw", "elec", "bonds", "angles", "dihe", "improper"]
    missing = [column for column in columns + ["model"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing CAPRI columns: {missing}")

    df["geometry_score"] = df[columns].sum(axis=1)
    ranked = df.sort_values("geometry_score").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob(f"{structure_name}_*.pdb"):
        old.unlink()

    nmodels = min(top, len(ranked))

    for i, model in enumerate(ranked["model"].iloc[:nmodels], start=1):
        src = flexref_dir / model
        dst = output_dir / f"{structure_name}_{i}.pdb"

        if not src.exists():
            raise FileNotFoundError(f"Missing flexref model: {src}")

        shutil.copy2(src, dst)

    print(f"Selected {nmodels} models in {output_dir}")
    return ranked.iloc[:nmodels].copy()


def _atom_key(line):
    return line[21].strip(), int(line[22:26]), line[17:20].strip(), line[12:16].strip()


def _set_atom_name(line, name):
    return line[:12] + f"{name:>4s}" + line[16:]


def _set_resname(line, name):
    return line[:17] + f"{name:>3s}" + line[20:]


def _set_resid(line, resid):
    return line[:22] + f"{resid:4d}" + line[26:]


def _set_serial(line, serial):
    return f"{line[:6]}{serial:5d}{line[11:]}"


def _make_ter(last_atom, serial):
    return (
        f"TER   {serial:5d}      "
        f"{last_atom[17:20]} "
        f"{last_atom[21]}"
        f"{last_atom[22:26]}"
        f"{last_atom[26]}"
    )


def _group_template_residues(dna_template):
    groups, current_key, current_lines = [], None, []

    for line in Path(dna_template).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue

        key = line[21].strip(), int(line[22:26]), line[26], line[17:20].strip()

        if current_key is not None and key != current_key:
            groups.append((current_key, current_lines))
            current_lines = []

        current_key = key
        current_lines.append(line)

    if current_lines:
        groups.append((current_key, current_lines))

    return groups


def _write_final_bonds(bond_file, output, residue_map, instances):
    bond_file, output = Path(bond_file), Path(output)
    bonds = pd.read_csv(bond_file, keep_default_na=False)
    final = []

    for _, bond in bonds.iterrows():
        left_key = (bond["left_type"], bond["left_instance"], int(bond["left_resid"]))
        right_key = (bond["right_type"], bond["right_instance"], int(bond["right_resid"]))

        if left_key not in residue_map:
            raise KeyError(f"Could not map bond residue {left_key}")
        if right_key not in residue_map:
            raise KeyError(f"Could not map bond residue {right_key}")

        final.append({
            "resid1": residue_map[left_key],
            "resname1": bond["left_resname"],
            "original_resid1": int(bond["left_resid"]),
            "atom1": bond["left_atom"],
            "resid2": residue_map[right_key],
            "resname2": bond["right_resname"],
            "original_resid2": int(bond["right_resid"]),
            "atom2": bond["right_atom"],
            "source1": bond["left_instance"] or "DNA",
            "source2": bond["right_instance"] or "DNA",
        })

    # Add bonds between different residues within composite dyes
    for instance in instances:
        for bond in instance.definition.read_inter_residue_bonds():
            left_key = ("dye", instance.name, bond["resid1"])
            right_key = ("dye", instance.name, bond["resid2"])

            if left_key not in residue_map or right_key not in residue_map:
                raise KeyError(
                    f"{instance.name}: could not map internal bond "
                    f"{bond['resid1']}:{bond['atom1']} - {bond['resid2']}:{bond['atom2']}"
                )

            final.append({
                "resid1": residue_map[left_key],
                "resname1": bond["resname1"],
                "original_resid1": bond["resid1"],
                "atom1": bond["atom1"],
                "resid2": residue_map[right_key],
                "resname2": bond["resname2"],
                "original_resid2": bond["resid2"],
                "atom2": bond["atom2"],
                "source1": instance.name,
                "source2": instance.name,
            })

    pd.DataFrame(final).drop_duplicates(
        subset=["resid1", "atom1", "resid2", "atom2"]
    ).to_csv(output, index=False)

    print(f"Wrote {output}")
    return output


def _reformat_docked_models(instances, dna_template, bonding_csv, structure_dir,
                           bond_file="haddock/bonds.csv", model_pattern="*.pdb"):
    dna_template, bonding_csv, structure_dir, bond_file = map(
        Path, (dna_template, bonding_csv, structure_dir, bond_file))

    for path in (dna_template, bonding_csv, bond_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    ter_after = set(pd.read_csv(bonding_csv)["ter_after_resid"].astype(int))
    template_groups = _group_template_residues(dna_template)

    ordered_instances = sorted(instances, key=lambda x: min(x.residues))
    insertion_blocks = []

    for instance in ordered_instances:
        start, end = min(instance.residues), max(instance.residues)

        if insertion_blocks and insertion_blocks[-1]["end"] + 1 == start:
            insertion_blocks[-1]["instances"].append(instance)
            insertion_blocks[-1]["end"] = end
        else:
            insertion_blocks.append({
                "insert_after": start - 1,
                "end": end,
                "instances": [instance],
            })

    insertions = {block["insert_after"]: block["instances"] for block in insertion_blocks}
    final_residue_map = None

    model_files = sorted(structure_dir.glob(model_pattern))
    if not model_files:
        raise FileNotFoundError(
            f"No selected HADDOCK models matching {model_pattern!r} in {structure_dir}"
        )

    for pdb in model_files:
        coordinates = [
            line for line in pdb.read_text().splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        ]

        docked_dna = {
            _atom_key(line): line for line in coordinates
            if line[72:76].strip() == "A"
        }

        dye_groups = {}

        for instance in instances:
            raw_dye = [line for line in coordinates if line[72:76].strip() == instance.segid]
            mapping = pd.read_csv(instance.mapping).reset_index(names="map_order")

            required = {"haddock_name", "original_name", "original_resname", "original_resid"}
            missing = required - set(mapping.columns)
            if missing:
                raise ValueError(f"{instance.mapping}: missing columns {sorted(missing)}")
            if mapping["haddock_name"].duplicated().any():
                raise ValueError(f"{instance.mapping}: haddock_name must be unique")
            if len(raw_dye) != len(mapping):
                raise ValueError(
                    f"{pdb.name}: {instance.name} has {len(raw_dye)} atoms; "
                    f"expected {len(mapping)}"
                )

            atom_map = mapping.set_index("haddock_name").to_dict("index")
            restored = []

            for line in raw_dye:
                name = line[12:16].strip()

                if name not in atom_map:
                    raise KeyError(f"{pdb.name}: no mapping for {instance.name} atom {name!r}")

                entry = atom_map[name]
                line = _set_atom_name(line, str(entry["original_name"]))
                line = _set_resname(line, str(entry["original_resname"]))
                line = _set_resid(line, int(entry["original_resid"]))
                line = fp.set_chain_and_segid(line)

                restored.append((int(entry["original_resid"]), int(entry["map_order"]), line))

            attachment = instance.definition.read_attachment()
            residue_ids = sorted(
                mapping["original_resid"].astype(int).unique(),
                reverse=attachment["5'"].resid > attachment["3'"].resid)

            restored.sort(key=lambda x: x[1])
            groups = []

            for original_resid in residue_ids:
                lines = [line for resid, _, line in restored if resid == original_resid]
                if not lines:
                    raise ValueError(
                        f"{pdb.name}: incomplete residue grouping for {instance.name}"
                    )
                groups.append((original_resid, lines))

            dye_groups[instance.name] = groups

        groups, inserted = [], set()

        for group_key, template_lines in template_groups:
            _, original_resid, _, _ = group_key
            rebuilt = []

            for template_line in template_lines:
                key = _atom_key(template_line)

                if key not in docked_dna:
                    element = template_line[76:78].strip()
                    if element != "H":
                        raise KeyError(f"{pdb.name}: missing docked DNA heavy atom {key}")
                    continue

                docked_line = docked_dna[key]
                rebuilt.append(template_line[:30] + docked_line[30:54] + template_line[54:])

            if not rebuilt:
                raise ValueError(f"{pdb.name}: no atoms remain for DNA residue {original_resid}")

            groups.append({
                "lines": rebuilt,
                "ter": original_resid in ter_after,
                "type": "dna",
                "instance": "",
                "original_resid": original_resid,
            })

            for instance in insertions.get(original_resid, []):
                for dye_resid, dye_lines in dye_groups[instance.name]:
                    groups.append({
                        "lines": dye_lines,
                        "ter": False,
                        "type": "dye",
                        "instance": instance.name,
                        "original_resid": dye_resid,
                    })
                inserted.add(instance.name)

        expected = {instance.name for instance in instances}
        if inserted != expected:
            raise RuntimeError(f"{pdb.name}: failed to insert {sorted(expected - inserted)}")

        output, serial, residue_map = [], 1, {}

        for new_resid, group in enumerate(groups, start=1):
            key = (group["type"], group["instance"], group["original_resid"])
            residue_map[key] = new_resid
            rewritten = []

            for line in group["lines"]:
                line = fp.set_chain_and_segid(line)
                line = _set_resid(line, new_resid)
                line = _set_serial(line, serial)
                rewritten.append(line)
                output.append(line)
                serial += 1

            if group["ter"]:
                output.append(_make_ter(rewritten[-1], serial))
                serial += 1

        if final_residue_map is None:
            final_residue_map = residue_map
        elif residue_map != final_residue_map:
            raise RuntimeError(f"{pdb.name}: residue numbering differs between selected models")

        output.append("END")
        pdb.write_text("\n".join(output) + "\n")
        print(f"Reformatted {pdb}")

    _write_final_bonds(bond_file, structure_dir / "bonds.csv", final_residue_map, instances)
