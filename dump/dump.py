# %%
from openbabel import openbabel

# Read the molecule from a SMILES string
smiles = r"c1ccc2c(c1)C(/C(=C\C=C\C1=[N+](c3c(C1(C)C)cccc3)CCCOP(=O)(O)O)/N2CCCOP(=O)(O)O)(C)C"  # Example: ethanol

# Initialize Open Babel objects
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("smi", "xyz")

# Create an OBMol object
mol = openbabel.OBMol()

# Read the SMILES string into the OBMol object
obConversion.ReadString(mol, smiles)

# Add explicit hydrogens to the molecule
mol.AddHydrogens()

# Check hybridization of each atom
# for atom in openbabel.OBMolAtomIter(mol):
#     print(f"Atom {atom.GetIndex() + 1} ({atom.GetType()}): sp = {atom.GetHyb()}")

# Generate 3D coordinates (preserves connectivity)
builder = openbabel.OBBuilder()
builder.Build(mol)  # Generate 3D geometry without changing connectivity

# Optionally, optimize geometry using a force field (e.g., MMFF94, UFF)
forcefield = openbabel.OBForceField.FindForceField("UFF")
forcefield.Setup(mol)
forcefield.ConjugateGradients(50000, 1.0e-10)  
forcefield.GetCoordinates(mol)

# Save the molecule as an xyz file
output_file = "output.xyz"
obConversion.WriteFile(mol, output_file)





