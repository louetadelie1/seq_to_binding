import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import mdtraj as md
import itertools
import glob
import os
import sys
import shutil

### The following is fitted for alpha_synuclein from DE Shaw. Made to handle dcd files
current_dir=sys.argv[1]
protein_name=sys.argv[2]
output_folder=sys.argv[3]

xtc=(f'{current_dir}/all_concatenated.xtc')
pdb=(f'{current_dir}/template.pdb')

traj = md.load(xtc, top=pdb)
topology = traj.topology
protein = traj.top.select('protein')
ligand_indices = topology.select('not protein')

# ### CALCULATING THE AVERAGE BETWEEN COM OF LIGAND AND COM OF EACH RESIUDUE && TAKING SMALLEST DISTANCE BETWEEN LIGAND AND PROTEIN AA
com_calcualtion_all=[]
clostest_calcualtion_all=[]

com_calcualtion=[]
clostest_calcualtion=[]
for resid in range(0,traj.topology.n_residues-1): #-1 beacus -1 is ligand
    res_x=topology.select(f'resid {resid}')
    pairs = list(itertools.product(res_x, ligand_indices))
    distances=md.compute_distances(traj, pairs, periodic=True, opt=True)
    com_calcualtion.append((np.average(distances,axis=1)).tolist())
    clostest_calcualtion.append(np.min(distances,axis=1).tolist())

d_24_t=np.reshape(com_calcualtion,((traj.topology.n_residues-1), len(traj))).T
d_24_t_closest=np.reshape(clostest_calcualtion,((traj.topology.n_residues-1), len(traj))).T

dir=f'{output_folder}/{protein_name}/distances/'
os.makedirs(dir, exist_ok=True)

shutil.copyfile(xtc, os.path.join(dir, os.path.basename(xtc)))
shutil.copyfile(pdb, os.path.join(dir, os.path.basename(pdb)))

with open(f'{dir}d_24_t_closest.pkl', 'wb') as file:
          pickle.dump(d_24_t_closest, file)

with open(f'{dir}d_24_t_com_avg.pkl', 'wb') as file:
          pickle.dump(d_24_t, file)
