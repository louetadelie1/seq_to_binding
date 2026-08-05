from openff.units import unit
import numpy as np
from openff.interchange import Interchange
from openff.toolkit import ForceField
from openff.toolkit import Molecule, Topology
import glob
import sys
import os
import pandas as pd
import pickle

current_dir=sys.argv[3]
smiles = sys.argv[2]
zinc_id = sys.argv[1]

sage = ForceField("openff-2.0.0.offxml")
cubic_box = unit.Quantity(30 * np.eye(3), unit.angstrom)



try:
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers()

    interchange = Interchange.from_smirnoff(
        topology=[mol],
        force_field=sage,
        box=cubic_box
    )
    zinc_dir = os.path.join(current_dir, zinc_id)
    param_dir = os.path.join(zinc_dir, f"{zinc_id}.param")

    os.makedirs(param_dir, exist_ok=True)

    output_prefix = os.path.join(param_dir, "system")
    # output_dir = os.path.join(current_dir, zinc_id)
    # os.makedirs(output_dir, exist_ok=True)

    # output_prefix = os.path.join(output_dir, "system")
    interchange.to_gromacs(output_prefix, monolithic=False)
    print(f'Wrapped up',zinc_id)
    print(output_prefix) 
except Exception as e:
    print(f"Error processing {zinc_id}: {e}")
