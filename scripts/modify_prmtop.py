#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import parmed as pmd
from netCDF4 import Dataset


DYE_A_RESNAMES = {"CY3"}
DYE_B_RESNAMES = {"CY5"}

DNA_RESNAMES = {
    "DA", "DA3", "DA5",
    "DT", "DT3", "DT5",
    "DG", "DG3", "DG5",
    "DC", "DC3", "DC5",
}


def read_forces(filename: Path) -> np.ndarray:
    with Dataset(filename, "r") as nc:
        if "forces" not in nc.variables:
            raise KeyError(
                f"{filename} does not contain a 'forces' variable. "
                f"Available variables: {list(nc.variables)}"
            )

        return np.asarray(nc.variables["forces"][:], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prmtop", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("nb_off", type=Path)
    parser.add_argument("bonded_off", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    parm = pmd.load_file(str(args.prmtop))

    full = read_forces(args.full)
    nb_off = read_forces(args.nb_off)
    bonded_off = read_forces(args.bonded_off)

    if full.shape != nb_off.shape or full.shape != bonded_off.shape:
        raise ValueError(
            "The three force arrays do not have identical shapes: "
            f"{full.shape}, {nb_off.shape}, {bonded_off.shape}"
        )

    if full.shape[1] != len(parm.atoms):
        raise ValueError(
            f"Force files contain {full.shape[1]} atoms, but topology "
            f"contains {len(parm.atoms)} atoms."
        )

    dye_a_indices = np.array(
        [
            atom.idx
            for atom in parm.atoms
            if atom.residue.name.strip() in DYE_A_RESNAMES
        ],
        dtype=int,
    )

    dna_indices = np.array(
        [
            atom.idx
            for atom in parm.atoms
            if atom.residue.name.strip() in DNA_RESNAMES
        ],
        dtype=int,
    )

    dye_nb_on_dna = full[:, dna_indices, :] - nb_off[:, dna_indices, :]

    dye_bonded_on_dna = (
        full[:, dna_indices, :]
        - bonded_off[:, dna_indices, :]
    )

    dye_total_on_dna = dye_nb_on_dna + dye_bonded_on_dna

    # Initialize everything to zero.
    output = np.zeros_like(full)

    # Dye A receives its complete full-system force.
    output[:, dye_a_indices, :] = full[:, dye_a_indices, :]

    # DNA receives only forces attributable to dyes A and B.
    output[:, dna_indices, :] = dye_total_on_dna

    with Dataset(args.output, "w", format="NETCDF4") as nc:
        nc.createDimension("frame", output.shape[0])
        nc.createDimension("atom", output.shape[1])
        nc.createDimension("spatial", 3)

        force_var = nc.createVariable(
            "forces",
            "f8",
            ("frame", "atom", "spatial"),
        )

        force_var[:] = output
        force_var.units = "unknown; copied from input force files"

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()