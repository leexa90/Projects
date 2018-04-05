import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
x=charge_csv[(charge_csv['atoms'] == 28) & (charge_csv['bond1'] == '[10]')\
             & (charge_csv['bond2'] == '[1, 3]') \
             & (charge_csv['charge'] >= 0.164) \
             & (charge_csv['charge'] <= 0.1652)]['origin'].values
test = pd.read_csv('../stapled_peptide_permability_features.csv')
temp=test.iloc[list(set(x))]
plt.plot(test)
