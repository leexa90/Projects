import numpy as np
import pandas as pd
import xgboost as xgb
train  = pd.read_csv('train_v2.csv')
test = pd.read_csv('test_v2.csv')
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
test['id'] = test['id'] + 3000
train = pd.concat([train,test]).reset_index(drop=True)
cols = [x for x in train.keys() if x not in ['id',target1,target2,'predict1','predict2']]
train[target1] = np.log1p(train[target1])
train[target2] = np.log1p(train[target2])
xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
train['predict1'] = 0
test[target1] = 0
train['predict2'] = 0
test[target2] = 0
from sklearn.cross_validation import train_test_split

for seed in [1,5516,643,5235,2352]:
    for i in range(0,4):
        np.random.seed(seed)
        train = train.sample(frac=1).reset_index(drop=True)
        test_id = [x for x in range(0,3000) if x%4 == i]
        train_id = [x for x in range(0,3000) if x%4 != i]
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]

        
        params = {}
        params["objective"] = 'reg:linear' 
        params["eta"] = 0.09
        params["min_child_weight"] = 10
        params["subsample"] = 0.6
        params["colsample_bytree"] = 0.6
        params["scale_pos_weight"] = 0.6
        params["silent"] = 0
        params["max_depth"] = 6
        params['seed']=seed
        #params['maximize'] =True
        params['eval_metric'] =  'rmse'
        plst = list(params.items())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train1.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test1.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        model1=xgb.train(plst,xgtrain,1000,watchlist,early_stopping_rounds=50,
                         evals_result=model1_a,maximize=False,verbose_eval=1000)
        train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(xgval)/5)
        test = test.set_value(test.index, target1, test[target1]+model1.predict(xgtest)/20)
        
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train2.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test2.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        params["eta"] = 0.01
        params["max_depth"] = 6
        plst = list(params.items())
        model1=xgb.train(plst,xgtrain,1000,watchlist,early_stopping_rounds=50,
                         evals_result=model1_a,maximize=False,verbose_eval=1000)
        train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model1.predict(xgval)/5)
        test = test.set_value(test.index, target2, test[target2]+model1.predict(xgtest)/20)
train2 = train[train['id'] < 3000]
print np.mean((train2['predict1']-train2[target1])**2)**.5
print np.mean((train2['predict2']-train2[target2])**2)**.5
test[target1] = np.exp(test[target1])-1
test[target2] = np.exp(test[target2])-1
train[target1] = np.exp(train[target1])-1
train[target2] = np.exp(train[target2])-1
train['predict1'] = np.exp(train['predict1'])-1
train['predict2'] = np.exp(train['predict2'])-1
test[['id',target1,target2]].to_csv('test3.csv',index=0)
#print train[[target1,'predict1',target2,'predict2']]
if False:
    train.to_csv('train_v2.csv',index=0)
    test.to_csv('test_v2.csv',index=0)
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
'''
