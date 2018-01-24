from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.nwchem import NWChem
from ase.io import write
from ase.visualize import view
from ase.io.opls import OPLSStructure
import pandas as pd
import numpy as np
'''
export PATH="~/.local/bin:$PATH"
export PYTHONPATH=~/.local/lib/python2.7/site-packages:$PYTHONPATH
export ASE_ABINIT_COMMAND="abinit < PREFIX.files > PREFIX.log"
PP=${HOME}/abinit-pseudopotentials-2
export ABINIT_PP_PATH=$PP/LDA_FHI
export ABINIT_PP_PATH=$PP/GGA_FHI:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/LDA_HGH:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/LDA_PAW:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/LDA_TM:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/GGA_FHI:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/GGA_HGHK:$ABINIT_PP_PATH
export ABINIT_PP_PATH=$PP/GGA_PAW:$ABINIT_PP_PATH
ABINIT_PP_PATH=$PP/GGA_PBE:$ABINIT_PP_PATH
ABINIT_PP_PATH=$PP/LDA_PW:$ABINIT_PP_PATH
'''

from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from ase.build import bulk
import os
result = {}
length = {}
files = [x[0] for x in  os.walk('.') if '/' in x[0] ]
for i in sorted(files,key = lambda x : int(x[2:]))[::-1]:
    if '00' in i:
        print i
    data =pd.read_csv(i+'/geometry.xyz',skiprows=4,sep=' ',header=None)
    data[0] = data[4]
    del data[4]
    data = data.set_value(0,[0,1,2,3],[len(data)-2,np.nan,np.nan,np.nan])
    data = data.set_value(1,[0,1,2,3],[np.nan,np.nan,np.nan,np.nan])
    data.to_csv(i+'/geometry2.xyz',index=0,header=0,sep=' ')
    import ase

    s = ase.io.read(i+'/geometry2.xyz',format='xyz')
    s.set_cell(pd.read_csv(i+'/geometry.xyz',skiprows=3,sep=' ',header=None).iloc[:3][[1,2,3]].values)

    calc = LennardJones()
    s.set_calculator(calc)
    v1 = s.get_volume()
    e1 = s.get_potential_energy()
    r1 = s.get_reciprocal_cell()
    f1 = s.get_forces()
    calc = MorsePotential()
    s.set_calculator(calc)
    e2 = s.get_potential_energy()
    r2 = s.get_reciprocal_cell()
    f2 = s.get_forces()
    result[int(i[2:])] = [v1,e1,r1,f1,e2,r2,f2]
    length[int(i[2:])] = map(sum,[data[0] =='Al',data[0] =='Ga',data[0] =='In',data[0] =='O'])
np.save('../train_energy.npy',result)
np.save('../train_resi.npy',length)
