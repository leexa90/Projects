import os
import pandas as pd
import numpy as  np
models = [x for x in os.listdir('.') if 'model' in x and 'train' in x]
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train = pd.read_csv('train_v2_nan.csv')
test = pd.read_csv('test_v2_nan.csv')
train[target1] =np.log1p(train[target1])
train[target2] =np.log1p(train[target2])
for i in models:
    temp1 = pd.read_csv(i)
    if np.mean(temp1[target1]) >= 0.18 :
        print i
        temp1[target1] = np.log1p(temp1[target1])
        temp1[target2] = np.log1p(temp1[target2])
    name = i.split('_')
    name[1] = 'test'
    temp2 = pd.read_csv(name[0]+'_'+name[1]+'_'+name[2])
    train = pd.merge(train,temp1[temp1.keys()[:3]],on=['id',])
    test = pd.merge(test,temp2, on=['id',])
    print i,np.mean(temp1[target1]),temp1[target1].head()

y = ['predict1_0.05375', 'predict2_0.05375', 'predict1_0.05349',
     'predict2_0.05349', 'predict1_0.05354', 'predict2_0.05354',
     'predict1_0.05107', 'predict2_0.05107', 'predict1_0.05102',
     'predict2_0.05102']#, 'formation_energy_ev_natom', 'bandgap_energy_ev']

