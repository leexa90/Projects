import os
import pandas as pd
import numpy as  np
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
models = [x for x in os.listdir('.') if 'model' in x and 'train' in x]
target1 = 'formation_energy_ev_natom'
target2 = 'bandgap_energy_ev'
train = pd.read_csv('train_v2_nan.csv')
test = pd.read_csv('test_v2_nan.csv')

train = train.fillna(0)
test = test.fillna(0)
for i in models:
    temp1 = pd.read_csv(i)
    name = i.split('_')
    name[1] = 'test'
    temp2 = pd.read_csv(name[0]+'_'+name[1]+'_'+name[2])
    train = pd.merge(train,temp1[temp1.keys()[:3]],on=['id',])
    test = pd.merge(test,temp2, on=['id',])
    print i,np.mean(temp1[target1])#,temp1[target1].head()
cols = [x for x in train.keys() if (x not in ['id',target1,target2,'predict1','predict2'] and 'predict'  in x)]
cols_ori = list(np.copy(cols))
train['predict1'] = 0
test['predict1'] = 0
train['predict2'] = 0
test['predict2'] = 0

seeds = [1,5516,643,5235,2352,12,5674,19239,41241,1231,151,34,1235,2663,75765,2314][:2]
print seeds,len(seeds)
comps=18
for seed in seeds:
    train = train.sample(2400,random_state=seed).reset_index(drop=True)
    for i in range(0,10):
        test_id = [x for x in range(0,2400) if x%10 == i] 
        train_id = [x for x in range(0,2400) if x%10 != i]
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
        from sklearn.decomposition import PCA
        regr = PCA(n_components=comps)
        names = []
        for i in range(comps):
            names += ['zz'+str(i),]
        for name in names :
            train[name] = 0
            test[name] = 0
        scaler = scaler.fit(train.iloc[train_id][cols_ori].values)
        temp = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values)).transform(scaler.transform(train[cols_ori].values))
        train = train.set_value(train.index,names,temp);
        temp = regr.fit(scaler.transform(train.iloc[train_id][cols_ori].values)).transform(scaler.transform(test[cols_ori].values))
        test = test.set_value(test.index,names,temp)
        if 'z1' not in cols:
            cols = ['z1','z2']+names
            dictt_cols1 = pd.DataFrame(cols ,columns=[1])
            dictt_cols2 = pd.DataFrame(cols ,columns=[1])
        X_train, y_train1,y_train2 = train.iloc[train_id][cols], train.iloc[train_id][target1],\
                                     train.iloc[train_id][target2]
        X_test, y_test1,y_test2 = train.iloc[test_id][cols], train.iloc[test_id][target1],\
                                  train.iloc[test_id][target2]
        xgtest = xgb.DMatrix(test[cols].values,missing=np.NAN,feature_names=cols)
        params = {}
        params["objective"] = 'reg:linear' 
        params["eta"] = 0.03
        params["min_child_weight"] = 10
        params["subsample"] = 0.6
        params["colsample_bytree"] = 0.4
        params["scale_pos_weight"] = 1
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
        model1=xgb.train(plst,xgtrain,1800,watchlist,early_stopping_rounds=50,
                         evals_result=model1_a,maximize=False,verbose_eval=1000)
        train = train.set_value(test_id,'predict1',train.iloc[test_id]['predict1']+model1.predict(xgval)/len(seeds))
        test = test.set_value(test.index, 'predict1', test['predict1']+model1.predict(xgtest)/(10*len(seeds)))
        dictt_cols1[len(dictt_cols1.keys())+1] = dictt_cols1[1].map(model1.get_fscore())
        xgtrain = xgb.DMatrix(X_train[cols].values, label=y_train2.values,missing=np.NAN,feature_names=cols)
        xgval = xgb.DMatrix(X_test[cols].values, label=y_test2.values,missing=np.NAN,feature_names=cols)
        watchlist  = [ (xgtrain,'train'),(xgval,'test')]
        model1_a = {}
        params["eta"] = 0.01
        params["max_depth"] = 6
        plst = list(params.items())
        model2=xgb.train(plst,xgtrain,2800,watchlist,early_stopping_rounds=50,
                         evals_result=model1_a,maximize=False,verbose_eval=1000)
        train = train.set_value(test_id,'predict2',train.iloc[test_id]['predict2']+model2.predict(xgval)/len(seeds))
        test = test.set_value(test.index, 'predict2', test['predict2']+model2.predict(xgtest)/(10*len(seeds)))
        dictt_cols2[len(dictt_cols2.keys())+1] = dictt_cols2[1].map(model2.get_fscore())
dictt_cols1 = dictt_cols1.fillna(0)
dictt_cols2 = dictt_cols2.fillna(0)
for i in range(2,1+len(dictt_cols1.keys())):
    dictt_cols1[i] = 100*dictt_cols1[i] / np.sum(dictt_cols1[i])
    dictt_cols2[i] = 100*dictt_cols2[i] / np.sum(dictt_cols2[i])
#print np.sum(dictt_cols1[range(2,len(dictt_cols1.keys()))].values,1)
#print np.sum(dictt_cols2[range(2,len(dictt_cols1.keys()))].values,1)
dictt_cols1['avg'] =  np.sum(dictt_cols1[range(2,len(dictt_cols1.keys()))].values,1)
dictt_cols2['avg'] =  np.sum(dictt_cols2[range(2,len(dictt_cols1.keys()))].values,1)
print dictt_cols2[[1,'avg']].sort_values('avg').iloc[-20:]
print dictt_cols1[[1,'avg']].sort_values('avg').iloc[-20:]
a,b = np.mean((train['predict1']-train[target1])**2)**.5, np.mean((train['predict2']-train[target2])**2)**.5
print a
print b
print (a+b)/2
if True:
    test[target1] = np.exp(test['predict1'])-1
    test[target2] = np.exp(test['predict2'])-1
    test[['id',target1,target2]].to_csv('test_v2%s.csv'%np.round((a+b)/2,4),index=0)

