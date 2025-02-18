# %%
import numpy as np
import quantumTools, structure
import fileProcessing as fp
import sys
import MDAnalysis as mda
import trajectory as traj
import os
from pyscf import lib

# %% [markdown]
# #### Obtaining the Input Structure 
# We first need to generate and optimize the strctures for Cy3 and Cy5. We proceed
# by constructing (unoptimized) files in Avogadro. 
# %%
# specify ChemDraw file *.cdx for molecule of interest
path = './createStructure/CY5/'
dye_name = 'CY5'
# optimization without considering 
quantumTools.optimizeStructureFF(path, dye_name, 50000)
# %% 
# now do an optimization with incorporating C2 symmetry
quantumTools.optimizeStructureSymmetryFF(path, dye_name, 5000)
# %% [markdow]n
# We can additionally run a quantum optimization of the geometry. This can be very time-consuming though.
# It seems to be more reasonable to do a good geometry optimization with molecular forcfields (e.g. with UFF).
# %%
# note : needs .xyz file (might need to use openbabbel to convert .pdb file accordingly before)
# mol = './createStructure/cy3/cy3_opt.xyz'
# quantumTools.optimizeStructureQM(mo)l
# %% [markdown]
# test to see how we can 'clean' input pdb files for chromophores.
# %%
dye_name = 'CY3'
inPath = f'./createStructure/{dye_name}/{dye_name}_opt.pdb'
outPath = f'./createStructure/{dye_name}/{dye_name}.pdb'
test = structure.cleanPDB(inPath, outPath, res_code = dye_name)
charge = 1                                                      # charge of dye in .cdx 
# %%
# run antechamber
import subprocess
dye_name = 'CY3'
charge = 1 # NOTE : should be set to one
makedir_ff = subprocess.run(f"mkdir -p ./createStructure/{dye_name}/ff", shell = True)
command = f"antechamber -i '../{dye_name}.pdb' -fi pdb -o {dye_name}.mol2 -fo mol2 -c bcc -s 2 -nc {charge} -m 1 -at gaff"
run_antechamber = subprocess.Popen(command, cwd = f'./createStructure/{dye_name}/ff', shell = True)
run_antechamber.wait()
# %%
# run parmchk2
import subprocess
dye_name = 'CY3'
command = f"parmchk2 -i {dye_name}.mol2 -f mol2 -o {dye_name}.frcmod -s gaff"
run_parmchk2 = subprocess.Popen(command, cwd = f'./createStructure/{dye_name}/ff', shell = True)
run_parmchk2.wait()
# %% [markdown]
# test DNA class
# TODO : maybe implement the a way of contruction nuc.pdb directly from this python script
# %%
path = './createStructure/nuc/'
dna = 'nuc.pdb'
# create MDAnalysis object
DNA_u = mda.Universe(path + dna, format = "PDB")


testDNA = structure.DNA(DNA_u)
# boxTest = testDNA.getDNAbox([0,0,0])
# example stationary positions for attachment at DNA
# target_pts = np.array([boxTest.bond_pos[4], boxTest.bond_pos[5]])
# target_lbs = [boxTest.bond_labels[4], boxTest.bond_labels[5]]
# target_idxs = [boxTest.res_idx[4], boxTest.res_idx[5]]
# target_resis = [boxTest.bond_resi[4], boxTest.bond_resi[5]]
# target_resns = [boxTest.bond_resn[4], boxTest.bond_resn[5]]
# com_DNA = boxTest.com



# print(target_pts)
# print(target_lbs)
# print(target_idxs)
# print(target_resis)
# print(target_resns)


# residue ID to delete (inspect from VMD)
resID = 14

testDNA.parseAttachment(resID)

# %% [markdown]
# #### Attachment information for CY3 and CY5 chromophores
# by visual inspection from the .pdb file e.g. with VMD
# %% 
dye_name = 'CY5'
path_to_dye = f'./createStructure/{dye_name}/'

# NOTE : 
# O followed by H means OH group
# OPO3_5 : this group gets linked to the 3' end of a nucleotide
# OPO3_5 : this group gets linked to the 5' end of a nucleotide
attach_groups = dict()
# (1) for CY3 (this needs to be done MANUALLY)
OPO3_5 = ['O1', 'P1', 'O2', 'O4', 'H25', 'O3', 'H24']      
OPO3_3 = ['O5', 'P2', 'O6', 'O8', 'H33', 'O7', 'H32']       
attach_groups['CY3'] = [OPO3_5, OPO3_3]
# (2) for CY5 (this needs to be done MANUALLY)
OPO3_5 = ['O1', 'P1', 'O2', 'O4', 'H27', 'O3', 'H26']      
OPO3_3 = ['O5', 'P2', 'O6', 'O8', 'H34', 'O7', 'H35']       
attach_groups['CY5'] = [OPO3_5, OPO3_3]

# write attachment information for specified dye
with open(path_to_dye + f"attach_info_{dye_name}.txt", "w") as file:
    for row in attach_groups[dye_name]:
        file.write(" ".join(row) + "\n")

# %% create force field for dye 
# we incorporate only atoms into the forcefield computation that remain based on attach_groups
dye_name = 'CY5'
inPath = f'./createStructure/{dye_name}/{dye_name}.pdb'  
dye = structure.Chromophore(mda.Universe(inPath, format = "PDB"))
dye.storeSourcePath(f'./createStructure/{dye_name}/')
dye.parseAttachment(change_atom_names = False)
dye.createFF()
# %% 
# load DNA and create CompositeStructure instance
old_dna = './createStructure/nuc/nuc.pdb'
dna = './createStructure/nuc/dna/dna.pdb'
composite = structure.CompositeStructure(dna, './createStructure/')

# (1) first attachment
dye_name = 'CY3'
repl_res_ID = 9
composite.prepareAttachment(dye_name, repl_res_ID)

# # (2) second attachment
# dye_name = 'CY3'
# repl_res_ID = 10
# composite.prepareAttachment(dye_name, repl_res_ID)

# (3) third attachment
dye_name = 'CY5'
repl_res_ID = 14
composite.prepareAttachment(dye_name, repl_res_ID)

# write AMBER input files
composite.writeAMBERinput()
# %% [markdown]
# ### DMREF Proposal figures
# create figures for DMREF proposal
# %%
# load DNA and create CompositeStructure instance
size = 60
dna = f'./createStructure/nuc/circdna/{size}/circ.pdb'
dna_out = f'./createStructure/nuc/circdna/{size}/circ_clean.pdb'

# TODO : change numbering of DNA residues
with open(dna, "r") as infile, open(dna_out, "w") as outfile:
    for line in infile:
        if line.startswith("TER"):  
            outfile.write(line)  # Keep TER lines unchanged
            continue

        if line.startswith(("ATOM", "HETATM")):
            resid = int(line[22:26].strip())  # Extract original residue number
            chain_id = line[21]  # Extract chain ID (A or B)

            # Shift Chain B residues by +50
            if chain_id == "B":
                new_resid = resid + size
            else:
                new_resid = resid  # Keep Chain A unchanged

            # Ensure correct formatting while updating residue numbers
            new_line = line[:22] + f"{new_resid:>4}" + line[26:]
            outfile.write(new_line)
        else:
            outfile.write(line)

composite = structure.CompositeStructure(dna_out, './createStructure/')

# (1) first attachment
dye_name = 'CY3'
repl_res_ID = 5
composite.prepareAttachment(dye_name, repl_res_ID, orientation = -1)

# # (2) second attachment
# dye_name = 'CY3'
# repl_res_ID = 10
# composite.prepareAttachment(dye_name, repl_res_ID)

# (3) third attachment
dye_name = 'CY5'
repl_res_ID = 112
composite.prepareAttachment(dye_name, repl_res_ID, orientation = -1)

# write output file
out = f'./createStructure/nuc/circdna/{size}/'
#composite.writeAMBERinput()
composite.writePDB(out)
# %% [markdown]
# draw spectral density function for figure
# %%
import matplotlib.pyplot as plt

# Define parameters
omega_c1 = 1  # Cutoff frequency
omega = np.linspace(0, 9, 500)  # Frequency range
J1 = (omega / omega_c1) * np.exp(-omega / omega_c1)  # Spectral density function
omega_c2 = 1.8
J2 = (omega / omega_c2) * np.exp(-omega / omega_c2)


# Set font properties globally
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 14
})

color1 = 'firebrick'
color2 = 'C02'

# Plot
plt.figure(figsize=(3, 2))
plt.plot(omega, J1, color = color1, linewidth = 2) 
#plt.plot(omega, J2, color = color2, linewidth = 2) 
plt.xlabel(r'$\omega/\omega_c$', fontname='Helvetics')
plt.ylabel(r'spectral density    $J(\omega)$', fontname='Helvetica')

# Remove tick labels on both axes
plt.tick_params(axis='both', which='both', direction='in', 
                labelbottom=False, labelleft=False, length=4)  # Keep tick marks
# save figure without background
plt.savefig("../DMREFproposal/figure/spectral_density.png", dpi=300, bbox_inches='tight', transparent=True)

plt.show()


# %% [markdown]



# #### Analyze MD trajectory data
# This is a test section to test MD trajectory data from AMBER and potentially feed them into `pyscf`.
# %%
# TODO: write class for MD simulation
params = []
MDsim = traj.MDSimulation(params)

path = './test/prod/'
name_prmtop = 'dna_test.prmtop'
name_nc = 'dna_test_prod.nc'                        # need to NetCDF3 and not NetCDF4 (use cpptraj to convert)
name_out = 'dna_test_prod.out'

data = [name_prmtop,name_nc, name_out]
dt = 10                                             # specify time step (ps)

test = traj.Trajectory(MDsim, path, data, dt)

# # analyze *.out file (e.g. for looking at energy convergence)
# path_to_perl = '/opt/homebrew/Caskroom/miniconda/base/envs/AmberTools23/bin/process_mdout.perl'
# test.analyzeOut(path_to_perl, plot = True)

# %%
# test analysis of trajectory

# input format for molecules to analyze during MD simulation: molecules = [molecule1, ...]
# with molecule1 = [resid1, resid2, ...]
donor = [9]
acceptor = [14]
molecules = [donor, acceptor]

# which information do we wish to extract from trajectory
traj_info = {'conversion': 'pyscf',
             'com': True}

# which time slice of the trajectory are we interested in?
time_slice = [0, 199]
distances = test.analyzeTrajectory(molecules, time_slice, **traj_info)

# # plot distance between chromophores:
# import matplotlib.pyplot as plt
# plt.plot(distances)
# plt.xlabel('Time Step')
# plt.ylabel(r'Distance CY3-CY5 ($\mathrm{\AA}$)')
# plt.show()

# %% [markdown]
# #### Quantum Mechanical Computations
# Since DFT and TDDFT calculations are expensive, they should be run on the cluster. Nonetheless, we show a simply sample
# case here of a stationary MD trajectory snapshot in order to provide some very first insights
# %% [markdown]
# (0) get some trajectory snapshot
# %%
molecule = [9]
idx = 10
chromophore, chromophore_conv = test.getChromophoreSnapshot(idx, molecule, conversion = 'pyscf')
# %% [markdown]
# We want to see how we can read in the output DataFrames we have analyzed from runnign classical and quantum analysis on the trajectories.
# %%
test = 'out_quant.txt'
test1 = 'out_class.txt'
# test for single index DataFrame:
df1, class_info = traj.Trajectory.readOutputFiles(path + 'out_class.txt', output_type='classical', output_info = path + 'qm_out.params')
# test for multi index DataFrame:
df2, state_dict, quant_info = traj.Trajectory.readOutputFiles(path + 'out_quant.txt', output_type='quantum', output_info = path + 'qm_out.params')
states = [0,0]

Test = df2[(state_dict[str(states)], 'coupling V_C')]
Test1 = df1['distance']
print(Test)
print(Test1)