import pandas as pd
import numpy as np

data = pd.read_csv('train_CNN.csv')
print np.mean((data['predict1']/10-data['bandgap_energy_ev'])**2)**.5
