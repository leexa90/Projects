import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
# calculate the volume of the structure

train  = pd.read_csv('train_v2_nan.csv')
test = pd.read_csv('test_v2_nan.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
cols = [x for x in train.keys() if (x not in ['id',target1,target2] and 'array' not in x and 'predict' not in x)]

train = train.fillna(0)
test = test.fillna(0)


train_ori = train.copy(deep = True)
cols_ori = list(np.copy(cols))
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2663,75765,2314]
print seeds,len(seeds)
for seed in seeds:
    train = train.sample(2400,random_state=seed).reset_index(drop=True)
    for i in range(0,6):
        test_id = [x for x in range(0,2400) if x%6 == i] 
        train_id = [x for x in range(0,2400) if x%6 != i] 
        scaler = StandardScaler()
        regr = linear_model.LinearRegression()
        scaler = scaler.fit(train.iloc[train_id][cols_ori].values,train[target1].values)
        train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                               train.iloc[train_id][target1].values).predict(scaler.transform(train[cols_ori].values))
        train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                               train.iloc[train_id][target2].values).predict(scaler.transform(train[cols_ori].values))
        test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                              train.iloc[train_id][target1].values).predict(scaler.transform(test[cols_ori].values))
        test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values),
                              train.iloc[train_id][target2].values).predict(scaler.transform(test[cols_ori].values))
        if 'z1' not in cols:
            cols += ['z1','z2']
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        Dense_input = keras.Input(train[cols].values.shape[1:])
        layer1 = keras.layers.Dense(train[cols].values.shape[1]//2, activation='relu', use_bias=True) (Dense_input)
        layer1 = keras.layers.Dropout(0.2)(layer1)
        layer2 = keras.layers.Dense(train[cols].values.shape[1]//4, activation='relu', use_bias=True) (layer1)
        layer2 = keras.layers.Dropout(0.2)(layer2)
        layer3 = keras.layers.Dense(train[cols].values.shape[1]//8, activation='relu', use_bias=True) (layer2)
        layer3 = keras.layers.Dropout(0.2)(layer3)
        layer_out = keras.layers.Dense(1,activation='linear',use_bias=True) (layer3)
        model = keras.models.Model(inputs=Dense_input, outputs=layer_out)
        model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                          optimizer='adam',# you can add several if needed
                         )
        model.summary();die
'''
benchmark *5
0.03250080113
0.0880216978405
0.0602612494852
with engeinner features *5
0.028781766046
0.0849339614409
0.0568578637434

engineer_features from paper*5
0.026665032169
0.0816681029669
0.054166567568

# engineer_features from paper*5 + bond angles
0.0245717966321
0.0791911408058
0.051881468719
'''

'''
https://media.nature.com/original/nature-assets/npjcompumats/2016/npjcompumats201628/extref/npjcompumats201628-s1.pdf
Average atomic mass
Average column on periodic table
Average  row  on  the  periodic  table
Average atomic radius
verage electronegativity
fraction of valence electrons
'''
