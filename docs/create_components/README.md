# Create Components

Component workflows create reusable molecular and force-field inputs that later structure workflows load from `DYE_DIR` and `LNK_DIR`.

## Workflows

- [create_dye](create_dye.md) creates and parameterizes one reusable dye residue.
- [create_linker](create_linker.md) creates and parameterizes reusable 3' and 5' linker residue templates.
- [create_dyelnk](create_dyelnk.md) combines existing dye and linker templates into a dye-linker attachment component.

## Common Implementation Pattern

The dye and linker workflows:

1. read a TOML file;
2. validate mapped SMILES and required fields;
3. generate an RDKit conformer;
4. optimize geometry using PySCF/GPU4PySCF when available;
5. compute an electrostatic potential;
6. run Amber RESP fitting;
7. generate MOL2/FRCMOD/attachment metadata;
8. remove temporary files according to `output.cleanup`.

## Library Output

Use `output.directory = "library"` only when the component is ready to be written into the shared reusable library. PyeDNA refuses to overwrite an existing library output directory.
