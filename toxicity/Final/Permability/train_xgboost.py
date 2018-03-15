import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv('permability_features.csv')

plt.plot((train['TPSA']-np.mean(train['TPSA']))/np.std(train['TPSA']),train['permability'],'bo');
plt.plot(train['Pred-Boosting'],train['permability'],'go');
train['bias'] = 1
features = ['HBA','HBD','SlogP_VSA3','SMR_VSA5','PEOE_VSA2','LabuteASA','TPSA','SMR_VSA1','PEOE_VSA1','VSA_EState9','bias']
features += list([x for x in train.keys() if 'atom' in x])[::1]
features += list([x for x in train.keys() if 'bond' in x])[::1]
X = train[train['set'] == 'Tr'][features].values

coef_ = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),train[train['set'] == 'Tr']['permability'].values)
train['pred'] = np.matmul(train[features].values,coef_)
plt.plot(train['pred'],train['permability'],'ro')
def R2_score(truth,predict):
    truth = np.array(truth)
    predict = np.array(predict)
    square_total = np.sum((truth - np.mean(truth))**2)
    square_pred = np.sum((truth - predict)**2)
    return 1-(square_pred/square_total)
print R2_score(train['permability'],train['pred'])
print R2_score(train['permability'],train['Pred-Boosting'])
val = train[train['set'] == 'val'].reset_index(drop=True)
te = train[train['set'] == 'Te'].reset_index(drop=True)
tr = train[train['set'] == 'Tr'].sort_values('permability').reset_index(drop=True)
if True:
    plt.clf()
    plt.plot(train[train['set'] == 'Tr']['permability'],
             train[train['set'] == 'Tr']['pred'],'ro')
    plt.plot(train[train['set'] == 'Te']['permability'],
             train[train['set'] == 'Te']['pred'],'bo')

    #plt.show()
print tr.corr()['permability']
target = 'permability'
print te[features+['pred','permability']].corr()
params = {}
params["objective"] = 'reg:linear' 
params["eta"] = 0.1
params["min_child_weight"] = 15
params["subsample"] = 1
params["featuresample_bytree"] = 1 # many features here
params["scale_pos_weight"] = 1
params["silent"] = 0
params["max_depth"] = 5
params['seed']=0
params['tree_method'] = 'hist'
#params['maximize'] =True
params['eval_metric'] =  'rmse'
params['eval_metric'] =  'rmse'
features = ['HBA','HBD','LabuteASA','TPSA']#['HBA','HBD','SlogP_VSA3','SMR_VSA5','PEOE_VSA2','LabuteASA','TPSA','SMR_VSA1','PEOE_VSA1','VSA_EState9','bias']
features += list([x for x in train.keys() if 'atom' in x])[::]
features += list([x for x in train.keys() if 'bond' in x])[::]
features += list([x for x in train.keys() if '_' in x if (x not in features and np.std(train[x]) >= 0.01)])
plst = list(params.items())
folds_id= []
if False:
    import xgbfir
    xgbfir.saveXgbFI(model1,MaxTrees=300,MaxInteractionDepth=5)
for repeat in range(0,1):
    for fold in range(5):
        train_id = [x for x in range(len(tr)) ]#if x%5!=fold]
        val_id = [x for x in range(len(tr)) if x%5==fold]
        xgtest = xgb.DMatrix(val[features].values,label=val[target].values,missing=np.NAN,feature_names=features)
        xgtrain = xgb.DMatrix(tr[features].iloc[train_id].values,
                              label=tr[target].iloc[train_id].values,
                              missing=np.NAN,feature_names=features)
        xgstop = xgb.DMatrix(tr[features].iloc[val_id].values,
                              label=tr[target].iloc[val_id].values,
                              missing=np.NAN,feature_names=features)
        xgval = xgb.DMatrix(te[features].values, label=te[target].values,missing=np.NAN,feature_names=features)
        model1_a = {}
        watchlist  = [ (xgtrain,'train'),(xgtest,'test'),(xgval,'val'),(xgstop,'stop')]
        def r2(y_pred,y_true):
            #negative AUC used because AUC is maximised while objective is minimized.
            #xgboost only maximises or minimise the obj function and stops when feval is ooptmized, cannot do one max,one min
            y_true =y_true.get_label()
            y_pred = np.array(y_pred)
            total_squares = np.sum((y_true - np.mean(y_true))**2)
            pred_squares = np.sum((y_true - y_pred)**2)
            return str(len(y_true))+'auc error',pred_squares/total_squares-1
        model1=xgb.train(plst,xgtrain,2700,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=100,feval=r2)
        te['fold_%s_%s'%(fold,repeat)] = model1.predict(xgval,ntree_limit=model1.best_ntree_limit)
        folds_id += ['fold_%s_%s'%(fold,repeat),]
    print r2(np.mean(te[folds_id],1),xgval) 
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
if True:
    D = np.abs(train[features].corr().values)**2
    Y = sch.linkage(D,method='single')
    Z = sch.dendrogram(Y,orientation='right')
    index = Z['leaves']
    plt.close()
    D = D[index,:] #reindex heatmap
    D = D[:,index] #reindex heatmap
    sorted_keys = [features[x] for x in index]
    plt.imshow(D)
    plt.legend()
    plt.xticks(range(len(index)),sorted_keys,rotation='vertical', fontsize=3)
    plt.yticks(range(len(index)),sorted_keys, fontsize=3)
    plt.savefig('Corr.png',dpi=300,bbox_inches='tight')
    die
    plt.show()
    Z = sch.dendrogram(Y,orientation='right')
    plt.yticks(np.array(range(1,1+len(index)))*10-5,sorted_keys, fontsize=4)
    plt.savefig('cluster.png',dpi=500)


from scipy.cluster.hierarchy import inconsistent
if True:
    Z = Y
    depth = 5
    incons = inconsistent(Z, depth)
    incons[-10:]
    depth = 3
    incons = inconsistent(Z, depth)
    incons[-10:]
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print "clusters:", k
cluster= fcluster(Z,25,criterion='maxclust')

def function(x):
    try:
        return a[a['Interaction']==x]['Gain'].values[0]
    except : return 0
for i in range(max(cluster)):
	if np.sum(cluster==i) >= 1:
		print(np.array(sorted_keys)[cluster==i]),
		temp = map(function,np.array(sorted_keys)[cluster==i])
		print (np.mean(temp),np.std(temp),temp)

[['bond_sum_0' 'bond_sum_1' 'atom_sum_1' 'LabuteASA_norm'],
['atom_sum_6' 'bond_1' 'SMR_VSA7' 'SlogP_VSA6'],
['PEOE_VSA4' 'PEOE_VSA3' 'SlogP_VSA10' 'VSA_EState8' 'atom_9' 'atom_sum_9']
,['PEOE_VSA11' 'SMR_VSA9' 'SlogP_VSA11'],
['VSA_EState10' 'atom_17' 'atom_sum_17'],
['SlogP_VSA7' 'SlogP_VSA3' 'SMR_VSA10' 'PEOE_VSA12' 'bond_2' 'PEOE_VSA2'
 'atom_sum_7' 'atom_7' 'SMR_VSA3'],
['atom_sum_8' 'charge_std' 'HBD_norm' 'TPSA_norm' 'HBA_norm' 'charge_skew'
 'charge_kurtosis'],
['PEOE_VSA7' 'PEOE_VSA8' 'SlogP_VSA2' 'SMR_VSA5' 'SMR_VSA1' 'bond_0'
 'atom_1' 'VSA_EState5' 'atom_6' 'atom_len' 'LabuteASA' 'atom_size'
 'SlogP_VSA5' 'VSA_EState9'],
['TPSA' 'HBA' 'atom_8' 'PEOE_VSA1' 'PEOE_VSA10' 'HBD' 'SMR_VSA2']]
a=pd.read_excel('XgbFeatureInteractions.xlsx')
