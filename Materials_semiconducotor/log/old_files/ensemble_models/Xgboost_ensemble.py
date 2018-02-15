import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
train  = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'

train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])

import os
for i in [x for x in os.listdir('.') if 'model_train' in x and np.float(x.split('_')[3][:-4]) < 0.0504]:
    name = i[11:]
    temp = pd.read_csv(i).sort_values('id').reset_index(drop=True)
    train['predict1'+name] = temp[temp.keys()[1]]
    train['predict2'+name] = temp[temp.keys()[2]]
    temp = pd.read_csv('model_test'+name).sort_values('id').reset_index(drop=True)
    test['predict1'+name] = temp[temp.keys()[1]]
    test['predict2'+name] = temp[temp.keys()[2]]
cols = [x for x in train.keys() if (x not in ['id',target1,target2] and 'predict'  in x)]
ori_cols = list(np.copy(cols))
print cols
dictt_cols1 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
dictt_cols2 = pd.DataFrame(cols + ['z1','z2'],columns=[1])
seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2664,
         75764,2314,1111,2222,3333,4444][:3]
print seeds,len(seeds)
comps=1
train['predict1'] = 0
test[target1] = 0
train['predict2'] = 0
test[target2] = 0

for seed in seeds:
    test['z1'] = 0
    test['z2'] = 0
    train['z1'] = 0
    train['z2'] = 0
    train = train.sample(2400,random_state=seed+4).reset_index(drop=True)
    for i in range(0,10):
        test_id = [x for x in range(0,2400) if x%10 == i] 
        train_id = [x for x in range(0,2400) if x%10 != i] 
        scaler = StandardScaler()
        regr = linear_model.LinearRegression()
        scaler = scaler.fit(train.iloc[train_id][ori_cols].values,train[target1].values)
        train['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                               train.iloc[train_id][target1].values).predict(scaler.transform(train[ori_cols].values))
        train['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                               train.iloc[train_id][target2].values).predict(scaler.transform(train[ori_cols].values))
        test['z1'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                              train.iloc[train_id][target1].values).predict(scaler.transform(test[ori_cols].values))
        test['z2'] = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values),
                              train.iloc[train_id][target2].values).predict(scaler.transform(test[ori_cols].values))
        xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
        a = np.mean((train['z1'].iloc[test_id]-train[target1].iloc[test_id])**2)**.5,
        b = np.mean((train['z2'].iloc[test_id]-train[target2].iloc[test_id])**2)**.5
##        print a
##        print b
        print (a+b)/2   ;

        xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
        from sklearn.decomposition import PCA
        regr = PCA(n_components=comps)
        names = []
        for i in range(comps):
            names += ['zz'+str(i),]
        for name in names :
            train[name] = 0
            test[name] = 0
        scaler = scaler.fit(train.iloc[train_id][ori_cols].values)
        temp = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values)).transform(scaler.transform(train[ori_cols].values))
        train = train.set_value(train.index,names,temp);
        temp = regr.fit(scaler.transform(train.iloc[train_id][ori_cols].values)).transform(scaler.transform(test[ori_cols].values))
        test = test.set_value(test.index,names,temp)
    for i in range(0,10):
        test_id = [x for x in range(0,2400) if x%10 == i] 
        train_id = [x for x in range(0,2400) if x%10 != i] 
        if 'z1' not in cols:
            cols += ['z1','z2'] #+names
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        

        params = {}
        params["objective"] = 'reg:linear' 
        params["eta"] = 0.03
        params["min_child_weight"] = 5
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.70 # many features here
        params["scale_pos_weight"] = 1
        params["silent"] = 0
        params["max_depth"] = 3
        params['seed']=seed
        #params['maximize'] =True
        params['eval_metric'] =  'rmse'
        if seed %2 == 0:
            None#params["eta"] = params["eta"]/3
        plst = list(params.items())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train1.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test1.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        model1=xgb.train(plst,xgtrain,5000,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=1000,
                         )
        train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(xgval)/len(seeds))
        test = test.set_value(test.index, target1, test[target1]+model1.predict(xgtest)/(10*len(seeds)))
        dictt_cols1[len(dictt_cols1.keys())+1] = dictt_cols1[1].map(model1.get_fscore())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train2.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test2.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        params["eta"] = 0.02
        params["max_depth"] = 3
        if seed %2 == 0:
            None#params["eta"] = params["eta"]/3
        plst = list(params.items())
        model2=xgb.train(plst,xgtrain,6500,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=1000,
                         )
        train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model2.predict(xgval)/len(seeds))
        test = test.set_value(test.index, target2, test[target2]+model2.predict(xgtest)/(10*len(seeds)))
        dictt_cols2[len(dictt_cols2.keys())+1] = dictt_cols2[1].map(model2.get_fscore())
if True:
    dictt_cols1['avg'] =  np.sum(dictt_cols1[range(2,len(dictt_cols1.keys()))].values,1)
    dictt_cols2['avg'] =  np.sum(dictt_cols2[range(2,len(dictt_cols1.keys()))].values,1)  
    print dictt_cols2[[1,'avg']].sort_values('avg').iloc[-50:]
    print dictt_cols1[[1,'avg']].sort_values('avg').iloc[-50:]
a,b = np.mean((train['predict1']-train[target1])**2)**.5, np.mean((train['predict2']-train[target2])**2)**.5
print a
print b
print (a+b)/2
train['avg2']= np.mean(train[[x for x in cols if 'predict2' in x]],1)
train['avg1']= np.mean(train[[x for x in cols if 'predict1' in x]],1)
a,b = np.mean((train['avg1']-train[target1])**2)**.5, np.mean((train['avg2']-train[target2])**2)**.5
print (a+b)/2
