import numpy as np

atom2Id = np.load('atom2Id.npy').item()
Id2atom = {atom2Id[x]: x for x in atom2Id.keys()}
bonds2 = np.load('bonds2.npy.zip')['bonds2.npy']
