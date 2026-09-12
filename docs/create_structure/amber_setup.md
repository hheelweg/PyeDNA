# Amber Preparation Moved To MD

Amber/tleap preparation is no longer a `create_structure` stage.

`create_structure finalize` now ends with ranked, finalized, unsolvated models in `structures/`:

```text
structures/<system>_1.pdb
structures/<system>_2.pdb
structures/<system>_3.pdb
structures/bonds.csv
structures/amber/<dye>_<linker>_linked.mol2
structures/amber/<dye>_<linker>_linked.frcmod
```

The numeric suffix records the existing finalized-model ranking. `_1.pdb` is the highest-ranked finalized structure, `_2.pdb` is second, and so on.

The MD workflow reads those structures, `structures/bonds.csv`, and any generated dye-linker metadata in `structures/amber/`, prepares each selected structure with `tleap`, and then runs minimization, equilibration, and production. Select structures in `md.toml`:

```toml
[system]
name = "dna_CY3_CY5"
structure_directory = "structures"
structures = [1, 2, 3]
```

See [do_md](../run_md/do_md.md) for the MD-owned `[amber]` settings and run directory layout.
