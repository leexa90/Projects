import pandas as pd
import numpy as np


train = pd.read_csv('train_CNN.csv')
test_pred = pd.read_csv('test_CNN.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
i=20
    a,b = np.mean((train['predict1']/i-np.log1p(train[target1]))**2)**.5, np.mean((train['predict1']/i-np.log1p(train[target2]))**2)**.5
    print a,b
