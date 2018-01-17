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
'''
convert spacegroup to one-hot-encoded
'''
all_dummy = pd.get_dummies(pd.concat([train['spacegroup'],test['spacegroup']]))

train[map(str,all_dummy.keys())] = all_dummy.iloc[:len(train)]
test[map(str,all_dummy.keys())] = all_dummy.iloc[len(train):]
cols = [x for x in train.keys() if (x not in ['id',target1,target2,'soacegroup'] and 'array' not in x and 'predict' not in x)]

for i in cols:
    train[i] = train[i].fillna(np.mean(train[i]))
    test[i] = test[i].fillna(np.mean(train[i]))
# 157 features
cols = ["('Al', 'Al')_115_125", "('Al', 'Al')_125_135", "('Al', 'Al')_85_95", "('Al', 'Al')_95_105",
        "('Al', 'Al')_A_mean", "('Al', 'Al')_A_std", "('Al', 'Al', 0)", "('Al', 'Al', 2)",
        "('Al', 'Al', 3)", "('Al', 'Ga')_115_125", "('Al', 'Ga')_125_135", "('Al', 'Ga')_85_95",
        "('Al', 'Ga')_95_105", "('Al', 'Ga')_A_mean", "('Al', 'Ga')_A_std", "('Al', 'Ga', 0)",
        "('Al', 'Ga', 1)", "('Al', 'Ga', 2)", "('Al', 'Ga', 3)", "('Al', 'In')_105_115",
        "('Al', 'In')_115_125", "('Al', 'In')_125_135", "('Al', 'In')_85_95", "('Al', 'In')_95_105",
        "('Al', 'In')_A_mean", "('Al', 'In')_A_std", "('Al', 'In', 0)", "('Al', 'In', 2)",
        "('Al', 'In', 3)", "('Al', 'O')_1.8_1.9", "('Al', 'O')_1.9_2.0", "('Al', 'O')_2.0_2.1",
        "('Al', 'O')_2.1_2.2", "('Al', 'O')_2.2_2.3", "('Al', 'O', 0)", "('Al', 'O', 1)",
        "('Al', 'O', 2)", "('Al', 'O', 3)", "('Al', 'O', 4)", "('Ga', 'Ga')_115_125", "('Ga', 'Ga')_125_135",
        "('Ga', 'Ga')_85_95", "('Ga', 'Ga')_95_105", "('Ga', 'Ga')_A_mean", "('Ga', 'Ga')_A_std",
        "('Ga', 'Ga', 0)", "('Ga', 'In')_105_115", "('Ga', 'In')_115_125", "('Ga', 'In')_125_135",
        "('Ga', 'In')_85_95", "('Ga', 'In')_95_105", "('Ga', 'In')_A_mean", "('Ga', 'In')_A_std",
        "('Ga', 'O')_1.8_1.9", "('Ga', 'O')_1.9_2.0", "('Ga', 'O')_2.0_2.1", "('Ga', 'O')_2.1_2.2",
        "('Ga', 'O', 0)", "('Ga', 'O', 2)", "('Ga', 'O', 3)", "('Ga', 'O', 4)", "('In', 'Ga', 0)",
        "('In', 'Ga', 2)", "('In', 'Ga', 3)", "('In', 'In')_105_115", "('In', 'In')_115_125",
        "('In', 'In')_125_135", "('In', 'In')_85_95", "('In', 'In')_95_105", "('In', 'In')_A_mean",
        "('In', 'In')_A_std", "('In', 'In', 0)", "('In', 'In', 1)", "('In', 'In', 3)", "('In', 'In', 4)",
        "('In', 'O')_1.7_1.8", "('In', 'O')_1.8_1.9", "('In', 'O')_1.9_2.0", "('In', 'O')_2.0_2.1",
        "('In', 'O')_2.1_2.2", "('In', 'O')_2.2_2.3", "('In', 'O', 0)", "('In', 'O', 2)", "('In', 'O', 3)",
        "('O', 'O')_105_115", "('O', 'O')_155_165", "('O', 'O')_175_185", "('O', 'O')_75_85",
        "('O', 'O')_85_95", "('O', 'O')_95_105", "('O', 'O')_A_mean", "('O', 'O')_A_std", "('O', 'O', 1)",
        "('O', 'O', 2)", "Bond_('Al', 'O')_mean", "Bond_('Al', 'O')_std", "Bond_('Ga', 'O')_mean",
        "Bond_('Ga', 'O')_std", "Bond_('In', 'O')_mean", "Bond_('In', 'O')_std", 'IonChar_mean', 'IonChar_std',
        "N('Al', 'Al', 0)", "N('Al', 'Al', 2)", "N('Al', 'Al', 3)", "N('Al', 'Ga', 0)", "N('Al', 'Ga', 1)",
        "N('Al', 'Ga', 2)", "N('Al', 'Ga', 3)", "N('Al', 'In', 0)", "N('Al', 'In', 2)", "N('Al', 'In', 3)",
        "N('Al', 'O', 0)", "N('Al', 'O', 1)", "N('Al', 'O', 2)", "N('Al', 'O', 3)", "N('Al', 'O', 4)",
        "N('Ga', 'Ga', 0)", "N('Ga', 'Ga', 2)", "N('Ga', 'Ga', 3)", "N('Ga', 'O', 0)", "N('Ga', 'O', 2)",
        "N('Ga', 'O', 3)", "N('Ga', 'O', 4)", "N('In', 'Ga', 0)", "N('In', 'Ga', 2)", "N('In', 'Ga', 3)",
        "N('In', 'In', 0)", "N('In', 'In', 1)", "N('In', 'In', 2)", "N('In', 'In', 3)", "N('In', 'In', 4)",
        "N('In', 'O', 0)", "N('In', 'O', 1)", "N('In', 'O', 2)", "N('In', 'O', 3)", "N('O', 'O', 0)",
        "N('O', 'O', 1)", "N('O', 'O', 2)", 'Norm2', 'Norm3', 'Norm5', 'lattice_angle_alpha_degree',
        'lattice_angle_beta_degree', 'lattice_angle_gamma_degree', 'lattice_vector_1_ang', 'lattice_vector_2_ang',
        'lattice_vector_3_ang', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'period_mean', 'period_std',
        'spacegroup', 'vol']
train_ori = train.copy(deep = True)
cols_ori = list(np.copy(cols))
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2663,75765,2314][:]
print seeds,len(seeds)
bad = []
train['pred1'] = 0
train['pred2'] = 0
test['pred1'] = 0
test['pred2'] = 0
num = 10
for seed in seeds:
    train = train.sample(2400,random_state=seed+1).reset_index(drop=True)
    for i in range(0,num):
        test_id = [x for x in range(0,2400) if x%num == i] 
        train_id = [x for x in range(0,2400) if x%num != i] 
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
        from keras import backend as K
        K.clear_session()
        Dense_input = keras.Input(train[cols].values.shape[1:])
        layer0 = keras.layers.BatchNormalization()(Dense_input)
        layer1 = keras.layers.Dense(train[cols].values.shape[1]*2, activation='relu', use_bias=True) (layer0)
        layer1 = keras.layers.Dropout(0.2)(layer1)
        layer1 = keras.layers.BatchNormalization()(layer1)
        layer2 = keras.layers.Dense(train[cols].values.shape[1], activation='relu', use_bias=True) (layer1)
        layer2 = keras.layers.Dropout(0.2)(layer2)
        layer2 = keras.layers.BatchNormalization()(layer2)
        layer3 = keras.layers.Dense(train[cols].values.shape[1]//2, activation='relu', use_bias=True) (layer2)
        layer3 = keras.layers.Dropout(0.2)(layer3)
        #layer3 = keras.layers.BatchNormalization()(layer3)
        layer_out = keras.layers.Dense(1,activation='linear',use_bias=True) (layer3)
        model = keras.models.Model(inputs=Dense_input, outputs=layer_out)
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        
        # prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=20, # was 10
                verbose=0),
        ]
        model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                          optimizer='adam',
                         )
        model.summary();
        history = model.fit(X_train.values,y_train1.values,128,epochs =400,verbose=2,
                  validation_data= (X_test.values,y_test1.values),callbacks=callbacks)
        train =  train.set_value(test_id,'pred1'      , train['pred1'].iloc[test_id]+model.predict(X_test.values)[:,0]/len(seeds))
        test = test.set_value(range(len(test)),'pred1', test['pred1']+model.predict(test[cols].values)[:,0]/(len(seeds)*num))
        from keras import backend as K
        K.clear_session()
        Dense_input = keras.Input(train[cols].values.shape[1:])
        layer0 = keras.layers.BatchNormalization()(Dense_input)
        layer1 = keras.layers.Dense(train[cols].values.shape[1]*2, activation='relu', use_bias=True) (layer0)
        layer1 = keras.layers.Dropout(0.2)(layer1)
        layer1 = keras.layers.BatchNormalization()(layer1)
        layer2 = keras.layers.Dense(train[cols].values.shape[1], activation='relu', use_bias=True) (layer1)
        layer2 = keras.layers.Dropout(0.2)(layer2)
        layer2 = keras.layers.BatchNormalization()(layer2)
        layer3 = keras.layers.Dense(train[cols].values.shape[1]//2, activation='relu', use_bias=True) (layer2)
        layer3 = keras.layers.Dropout(0.2)(layer3)
        layer_out = keras.layers.Dense(1,activation='linear',use_bias=True) (layer3)
        model2 = keras.models.Model(inputs=Dense_input, outputs=layer_out)
        model2.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                          optimizer='adam',
                         )
        history2 = model2.fit(X_train.values,y_train2.values,128,epochs =400,verbose=2,
                  validation_data= (X_test.values,y_test2.values),callbacks=callbacks)
        train =  train.set_value(test_id,'pred2', train['pred2'].iloc[test_id]+model2.predict(X_test.values)[:,0]/len(seeds))
        test = test.set_value(range(len(test)),'pred2',test['pred2']+model2.predict(test[cols].values)[:,0]/(len(seeds)*num))
print train[['z1','z2','pred1','pred2',target1,target2]].corr()
if True:
    a,b = np.mean((train['pred1']-train[target1])**2)**.5, np.mean((train['pred2']-train[target2])**2)**.5
    print a
    print b
    print (a+b)/2
    name1 = 'predict1_%s'%np.round((a+b)/2,5)
    name2 = 'predict2_%s'%np.round((a+b)/2,5)
    test[name1] = np.log1p(test['pred1'])
    test[name2] = np.log1p(test['pred2'])
    train[name1] = np.log1p(train['pred1'])
    train[name2] = np.log1p(train['pred2'])
    train[['id',name1,name2,target1,target2]].to_csv('modelNN_train_%s.csv'%np.round((a+b)/2,5),index=0)
    test[['id',name1,name2]].to_csv('modelNN_test_%s.csv'%np.round((a+b)/2,5),index=0)
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
