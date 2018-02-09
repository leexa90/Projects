import pandas as pd
import numpy as np

data = pd.read_csv('train_CNN.csv')
print np.mean((data['predict1']/10-data['formation_energy_ev_natom'])**2)**.5
