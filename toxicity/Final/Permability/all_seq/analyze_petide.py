import pandas as pd
import xgboost as xgb
import numpy as np
target = 'permability'
train = pd.read_csv('peptide_permability_features.csv')
def remove_highly_pos(seq):
    total = len(seq)
    count = 0.0
    for i in seq:
        if i.lower() in ['r','k']:
            count += 1
    return count/total
train = train[train['ID'].apply(remove_highly_pos) < 0.34]
from scipy import histogram, digitize, stats, mean, std
import matplotlib.pyplot as plt
features = [u'charge_skew',
       u'charge_kurtosis', u'charge_std',  u'atom_8',
       u'atom_sum_8', u'atom_1', u'atom_sum_1', u'atom_16', u'atom_sum_16',
       u'atom_6', u'atom_sum_6', u'atom_7', u'atom_sum_7', u'TPSA',
       u'TPSA_norm', u'LabuteASA', u'LabuteASA_norm', u'PEOE_VSA1',
       u'PEOE_VSA2', u'PEOE_VSA3', u'PEOE_VSA4', u'PEOE_VSA5', u'PEOE_VSA6',
       u'PEOE_VSA7', u'PEOE_VSA8', u'PEOE_VSA9', u'PEOE_VSA10', u'PEOE_VSA11',
       u'PEOE_VSA12', u'PEOE_VSA13', u'PEOE_VSA14', u'SMR_VSA1', u'SMR_VSA2',
       u'SMR_VSA3', u'SMR_VSA4', u'SMR_VSA5', u'SMR_VSA6', u'SMR_VSA7',
       u'SMR_VSA8', u'SMR_VSA9', u'SMR_VSA10', u'SlogP_VSA1', u'SlogP_VSA2',
       u'SlogP_VSA3', u'SlogP_VSA4', u'SlogP_VSA5', u'SlogP_VSA6',
       u'SlogP_VSA7', u'SlogP_VSA8', u'SlogP_VSA9', u'SlogP_VSA10',
       u'SlogP_VSA11', u'VSA_EState1', u'VSA_EState2', u'VSA_EState3',
       u'VSA_EState4', u'VSA_EState5', u'VSA_EState6', u'VSA_EState7',
       u'VSA_EState8', u'VSA_EState9', u'VSA_EState10', u'HBA', u'HBD',
       u'Rotatable_num', u'HBA_norm', u'HBD_norm', u'atom_len', u'atom_size']
params = {}
params["objective"] = 'binary:logistic' 
params["eta"] = 0.1
params["min_child_weight"] = 5
params["subsample"] = 1
params["featuresample_bytree"] = 1 # many features here
params["scale_pos_weight"] = 5
params["silent"] = 0
params["max_depth"] = 5
params['seed']=0
params['tree_method'] = 'hist'
#params['maximize'] =True
params['eval_metric'] =  'auc'
train = train.sort_values([target,'size']).reset_index(drop=True)
plst = list(params.items())
import scipy
features2 = []
for i in features+[target,]:
    print i
    train[i+'percentile'] = ((scipy.stats.rankdata(train[i].values)-1)/len(train))*100//1 #into 100 bins
def mutual_information(x,y):
        #discritize to 100 bins
    #   binx = histogram(x,100,density=True)
    #   biny = histogram(y,100,density=True)
        temp = plt.hist2d(x,y,99)
        plt.close()
        prob,x_bin,y_bin = temp[0],temp[1],temp[2]
        prob = temp[0]/np.sum(temp[0])
        result = 0.0
        for i in range(99):
            sumI = np.sum(prob[i,:])
            if sumI != 0:
                for j in range(99):
                    sumJ = np.sum(prob[:,j])
                    if sumJ != 0 and prob[i][j] != 0:
                        result += prob[i][j]*np.log(prob[i][j]/(sumI*sumJ))
        return result
if True:
    features = [u'SMR_VSA9', u'SlogP_VSA11', u'PEOE_VSA3', u'SlogP_VSA8',
                u'PEOE_VSA13', u'PEOE_VSA14', u'SMR_VSA7', u'SlogP_VSA6',
                u'atom_16', u'PEOE_VSA4', u'VSA_EState4', u'SMR_VSA4',
                u'SlogP_VSA4', u'atom_7', u'SlogP_VSA7', u'HBD', u'atom_8',
                u'HBA', u'PEOE_VSA6', u'atom_sum_6', u'atom_sum_1', u'LabuteASA_norm',
                u'atom_sum_7', u'charge_skew', u'HBD_norm', u'charge_kurtosis',
                u'atom_sum_8', u'HBA_norm', u'charge_std', u'TPSA_norm',
                u'PEOE_VSA9', u'PEOE_VSA10', u'SlogP_VSA3', u'SMR_VSA10',
                u'PEOE_VSA7', u'SMR_VSA3', u'PEOE_VSA8', u'VSA_EState9',
                u'SMR_VSA5', u'SlogP_VSA5', u'Rotatable_num', u'atom_6',
                u'atom_1', u'VSA_EState5', u'atom_len', u'SMR_VSA1', u'LabuteASA',
                u'atom_size', u'PEOE_VSA12', u'SlogP_VSA2', u'SlogP_VSA1', u'TPSA',
                u'PEOE_VSA1', u'PEOE_VSA2', u'PEOE_VSA11', u'SMR_VSA2', u'SMR_VSA6',
                u'atom_sum_16', u'VSA_EState10']
    features = sorted(features)
    corr = np.zeros((len(features),len(features)))
    for idI in range(len(features)):
        for idJ in range(idI,len(features)):
            i,j = features[idI],features[idJ]
            val = mutual_information(train[j+'percentile'],train[i+'percentile'])
            corr[idI,idJ] = val
            corr[idJ,idI] = val
            if val > 1:
                print i,j,val
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch
xgbfile= pd.read_csv('XgbFeatureInteractions-20tree.csv')
for i in range(len(features)):
    print features[i],
    print np.mean(xgbfile[xgbfile['Interaction'].isin(np.array(features)[np.argsort(corr[i])][-5:])]['Gain']),
    print np.mean(xgbfile[xgbfile['Interaction'].isin(np.array(features)[corr[i] > 1.5])]['Gain'])
for i in range(len(features)):
    print features[i],
    print 
    
if True:
    D = corr
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
die
from scipy import stats
for i in features:
    x= train.groupby(target)[i].apply(list)
    pval = stats.ks_2samp(x[0], x[1])[1]
    y1= list(plt.hist(x[0],bins=20,normed=True))
    y2= list(plt.hist(x[1],bins=20,normed=True))
    y1[0] = np.append(y1[0],0)
    y2[0] = np.append(y2[0],0)
    plt.close()
    plt.boxplot((x[0],x[1]))
    plt.plot([0.0,]*20,y2[1][:20],'c')
    plt.plot([0.0,]*20,y1[1][:20],'c')
    plt.plot(0.0+0.5*(y1[0]/max(y1[0])),y1[1][:],'b')
    plt.plot(0.0+0.5*(y2[0]/max(y2[0])),y2[1][:],'r')
    
    plt.plot([1.1,]*20,y1[1][:20],'b')
    plt.plot(1.1+0.5*(y1[0]/max(y1[0])),y1[1][:],'b')
    plt.plot(2.1+0.5*(y2[0]/max(y2[0])),y2[1][:],'r')
    plt.plot([2.1,]*20,y2[1][:20],'r')
    plt.title(pval)
    plt.ylabel(i)
    plt.xlim(0,3)
    plt.savefig(str(i)+'_.png')
    plt.close()
for repeat in range(0,1):
    for fold in range(1):
        train_id = [x for x in range(len(train)) if x%5!=fold]
        val_id = [x for x in range(len(train)) if x%5==fold]
        xgtest = xgb.DMatrix(train[features].iloc[val_id].values,
                             label=train[target].iloc[val_id].values,
                             missing=np.NAN,feature_names=features)
        xgtrain = xgb.DMatrix(train[features].iloc[train_id].values,
                              label=train[target].iloc[train_id].values,
                              missing=np.NAN,feature_names=features)
        model1_a = {}
        watchlist  = [ (xgtrain,'train'),(xgtest,'test')]
        model1=xgb.train(plst,xgtrain,200,watchlist,early_stopping_rounds=200,
                         evals_result=model1_a,maximize=False,verbose_eval=100)
feature1_score = {}
for i in range(len(features)):
    for repeat in range(0,1):
        for fold in range(1):
            print features[i],
            train_id = [x for x in range(len(train)) if x%5!=fold]
            val_id = [x for x in range(len(train)) if x%5==fold]
            xgtest = xgb.DMatrix(train[features[i:i+1]].iloc[val_id].values,
                                 label=train[target].iloc[val_id].values,
                                 missing=np.NAN)
            xgtrain = xgb.DMatrix(train[features[i:i+1]].iloc[train_id].values,
                                  label=train[target].iloc[train_id].values,
                                  missing=np.NAN)
            model1_a = {}
            watchlist  = [ (xgtrain,'train'),(xgtest,'test')]
            model1=xgb.train(plst,xgtrain,300,watchlist,early_stopping_rounds=50,
                             evals_result=model1_a,maximize=False,verbose_eval=500)
            model1=xgb.train(plst,xgtrain,model1.best_ntree_limit,watchlist,early_stopping_rounds=50,
                             evals_result=model1_a,maximize=False,verbose_eval=500)
            feature1_score[features[i]] = (model1.best_score,model1,xgtest)
feature2_score = {}
from sklearn.metrics import roc_auc_score

for i in range(len(features)):
    for j in range(i+1,len(features)):
        for fold in range(1):
            print features[i],features[j],
            temp_f = [features[i],features[j]]
            train_id = [x for x in range(len(train)) if x%5!=fold]
            val_id = [x for x in range(len(train)) if x%5==fold]
            xgtest = xgb.DMatrix(train[temp_f].iloc[val_id].values,
                                 label=train[target].iloc[val_id].values,
                                 missing=np.NAN)
            xgtrain = xgb.DMatrix(train[temp_f].iloc[train_id].values,
                                  label=train[target].iloc[train_id].values,
                                  missing=np.NAN)
            model1_a = {}
            watchlist  = [ (xgtrain,'train'),(xgtest,'test')]
            model1=xgb.train(plst,xgtrain,300,watchlist,early_stopping_rounds=50,
                             evals_result=model1_a,maximize=False,verbose_eval=500)
            pred1 =  feature1_score[features[i]][1].predict(feature1_score[features[i]][2])
            pred1 +=  feature1_score[features[j]][1].predict(feature1_score[features[j]][2])
            pred1 = pred1/2
            score1= roc_auc_score(xgtest.get_label(),pred1)
            score2= roc_auc_score(xgtest.get_label(),model1.predict(xgtest,ntree_limit=model1.best_ntree_limit))
            print score1,score2
            feature2_score[(features[i],features[j])] = (model1.best_score,model1,xgtest,(score1,score2))
plt.plot(range(59),map(lambda x : feature1_score[x][0],sorted(feature1_score.keys(),key = lambda x : feature1_score[x])),'ro');plt.show()
plt.plot([ feature2_score[x][-1][0] for x in  feature2_score.keys()],
	 [feature2_score[x][-1][-1] for x in feature2_score.keys()],'ro');plt.show()
WC = pd.read_csv('wildmen_crippen.csv')
WC[(WC['MR'] > 1.29) & (WC['MR'] < 1.82) ][['type','descriptions']].values

WC[(WC['MR'] > 1.82) & (WC['MR'] < 2.24) ][['type','descriptions']].values

WC[(WC['MR'] >2.24) & (WC['MR'] < 2.45) ][['type','descriptions']].values

WC[(WC['Log P'] > -999) & (WC['Log P'] < -0.4) ][['type','descriptions']].values

WC[(WC['Log P'] > 0) & (WC['Log P'] < .1) ][['type','descriptions']].values
'''
charge_skew charge_skew 4.591354836255298
charge_skew charge_kurtosis 1.0774090707837758
charge_skew charge_std 1.142039725227622
charge_skew atom_sum_8 1.1332070258456124
charge_skew TPSA_norm 1.0709828282302822
charge_skew HBA_norm 1.138341895679312
charge_kurtosis charge_kurtosis 4.591354836255298
charge_kurtosis charge_std 1.2035161401242287
charge_kurtosis TPSA_norm 1.360946358356288
charge_kurtosis HBA_norm 1.0875864030656517
charge_kurtosis HBD_norm 1.2211057770806504
charge_std charge_std 4.591352602169483
charge_std atom_sum_8 1.3335060982339704
charge_std atom_sum_1 1.0375837118914144
charge_std TPSA_norm 1.517429890001562
charge_std VSA_EState9 1.0502775219103158
charge_std HBA_norm 1.4268842693365427
charge_std HBD_norm 1.1525378645400555
atom_8 atom_8 3.2233744400022943
atom_8 TPSA 1.1914026749080817
atom_8 PEOE_VSA1 1.2130845690786136
atom_8 PEOE_VSA2 1.3266293745265454
atom_8 PEOE_VSA12 1.3179180115540376
atom_8 SMR_VSA1 1.0942778553289318
atom_8 SMR_VSA10 1.0668561306309539
atom_8 SlogP_VSA2 1.1423965714313162
atom_8 SlogP_VSA3 1.362386735918537
atom_8 HBA 1.5622455125680943
atom_sum_8 atom_sum_8 4.584610318962971
atom_sum_8 atom_sum_1 1.0087928095704768
atom_sum_8 atom_sum_7 1.0240647719310076
atom_sum_8 TPSA_norm 1.0869754527782698
atom_sum_8 VSA_EState9 1.0261595330051432
atom_sum_8 HBA_norm 1.503867864833924
atom_sum_8 HBD_norm 1.16569534556088
atom_sum_8 atom_len 1.6145143289333972
atom_1 atom_1 4.525926173516317
atom_1 atom_sum_1 1.3512212692829744
atom_1 atom_6 1.6587218888701334
atom_1 atom_7 1.3050433526208214
atom_1 TPSA 1.2692880861282976
atom_1 LabuteASA 1.8886688099003193
atom_1 PEOE_VSA1 1.3263691584987676
atom_1 PEOE_VSA2 1.1759573434352322
atom_1 PEOE_VSA7 1.276586026790564
atom_1 PEOE_VSA8 1.4110008751283238
atom_1 PEOE_VSA10 1.479925327156997
atom_1 PEOE_VSA11 1.0576616706455255
atom_1 PEOE_VSA12 1.343157535981614
atom_1 SMR_VSA1 1.822587819870768
atom_1 SMR_VSA3 1.489435476917676
atom_1 SMR_VSA5 1.7320140489032143
atom_1 SMR_VSA10 1.3229104899940494
atom_1 SlogP_VSA1 1.3754052549136295
atom_1 SlogP_VSA2 1.2977737666399771
atom_1 SlogP_VSA3 1.211917865168816
atom_1 SlogP_VSA5 1.698275314450741
atom_1 VSA_EState5 3.122048001326359
atom_1 VSA_EState9 1.3909513267225055
atom_1 Rotatable_num 1.7686919180968919
atom_1 atom_len 2.222960258441413
atom_1 atom_size 1.711425348441909
atom_sum_1 atom_sum_1 4.547066355354242
atom_sum_1 atom_sum_7 1.0485416252166986
atom_sum_1 LabuteASA 1.2331734717754907
atom_sum_1 LabuteASA_norm 1.67302500225411
atom_sum_1 SMR_VSA5 1.0205615962675043
atom_sum_1 VSA_EState5 1.1806098426526834
atom_sum_1 HBA_norm 1.0300793697607546
atom_sum_1 atom_len 1.363861953814469
atom_sum_1 atom_size 1.152164045083726
atom_16 atom_16 1.2027073321127129
atom_16 PEOE_VSA4 1.1943796757865432
atom_16 VSA_EState10 1.1510267278277833
atom_sum_16 atom_sum_16 2.2928934268064607
atom_sum_16 VSA_EState10 1.1304174497803658
atom_sum_16 atom_len 1.1499832526896336
atom_6 atom_6 4.335590240151673
atom_6 atom_sum_6 1.1983389883360382
atom_6 TPSA 1.1569324860190895
atom_6 LabuteASA 2.081711965731031
atom_6 PEOE_VSA1 1.2306798898910531
atom_6 PEOE_VSA2 1.1077762782186131
atom_6 PEOE_VSA7 1.117788353264615
atom_6 PEOE_VSA8 1.3363504813344462
atom_6 PEOE_VSA10 1.3965446979058411
atom_6 PEOE_VSA12 1.275716533187881
atom_6 SMR_VSA1 1.5829059379351216
atom_6 SMR_VSA3 1.3854191629155383
atom_6 SMR_VSA5 1.3324955185049234
atom_6 SMR_VSA10 1.2396083314985133
atom_6 SlogP_VSA1 1.1816062093183024
atom_6 SlogP_VSA2 1.2222603716389013
atom_6 SlogP_VSA3 1.3440423053682253
atom_6 SlogP_VSA5 1.2777220173315604
atom_6 VSA_EState5 1.7340049179841563
atom_6 VSA_EState9 1.1088316320598595
atom_6 Rotatable_num 1.3209340729939791
atom_6 atom_len 1.9956540137645697
atom_6 atom_size 1.92300304445686
atom_sum_6 atom_sum_6 4.5705707467765935
atom_sum_6 atom_sum_7 1.0185212506086294
atom_sum_6 PEOE_VSA6 1.0202065896499994
atom_sum_6 SlogP_VSA3 1.0580676399161542
atom_sum_6 HBD_norm 1.0344884113730513
atom_sum_6 atom_len 1.4187241229299719
atom_7 atom_7 3.2330536109912327
atom_7 TPSA 1.2187012791862024
atom_7 LabuteASA 1.0921198733565352
atom_7 PEOE_VSA1 1.3302720892012423
atom_7 PEOE_VSA2 1.2835791990781855
atom_7 PEOE_VSA11 1.4155969425487345
atom_7 PEOE_VSA12 1.122477057792186
atom_7 SMR_VSA3 1.4340727572726233
atom_7 SMR_VSA10 1.1195865377593979
atom_7 SlogP_VSA1 1.482118183398314
atom_7 SlogP_VSA7 1.497094305061348
atom_7 VSA_EState5 1.0017692837155825
atom_7 HBD 1.117073375305552
atom_7 atom_len 1.0841872530353707
atom_7 atom_size 1.1101157647757531
atom_sum_7 atom_sum_7 4.579007029565158
atom_sum_7 HBA_norm 1.0142790784580655
atom_sum_7 HBD_norm 1.152885216801129
atom_sum_7 atom_len 1.725456839310728
TPSA TPSA 4.591222443942684
TPSA LabuteASA 1.4053127870329098
TPSA PEOE_VSA1 2.4042988733753234
TPSA PEOE_VSA2 1.6629427742004421
TPSA PEOE_VSA7 1.0218682489317081
TPSA PEOE_VSA8 1.1064033294220277
TPSA PEOE_VSA9 1.1029340455165295
TPSA PEOE_VSA10 1.3534937403932152
TPSA PEOE_VSA11 1.3716249209042395
TPSA PEOE_VSA12 1.3900748225528663
TPSA SMR_VSA1 1.5080751800773031
TPSA SMR_VSA3 1.4896872742141205
TPSA SMR_VSA5 1.2388814366815795
TPSA SMR_VSA10 1.5281211098189917
TPSA SlogP_VSA1 1.5966529302671144
TPSA SlogP_VSA2 1.5591835318872136
TPSA SlogP_VSA3 1.4802635890913567
TPSA SlogP_VSA5 1.1184563292555838
TPSA SlogP_VSA7 1.187429586160255
TPSA VSA_EState5 1.2714032905540027
TPSA HBA 1.49028560821727
TPSA HBD 1.6705847460817127
TPSA Rotatable_num 1.2973423110777216
TPSA atom_len 1.375530524340579
TPSA atom_size 1.4667680369706664
TPSA_norm TPSA_norm 4.591345899912038
TPSA_norm VSA_EState9 1.0042302181417624
TPSA_norm HBA_norm 1.1948684707953448
TPSA_norm HBD_norm 1.2800119058553352
LabuteASA LabuteASA 4.591348100145496
LabuteASA PEOE_VSA1 1.4735680168591156
LabuteASA PEOE_VSA2 1.3291959451121442
LabuteASA PEOE_VSA7 1.2024232704524143
LabuteASA PEOE_VSA8 1.4314038767436845
LabuteASA PEOE_VSA9 1.0283579766328366
LabuteASA PEOE_VSA10 1.608735009727778
LabuteASA PEOE_VSA11 1.1516026372562909
LabuteASA PEOE_VSA12 1.4912483729334327
LabuteASA SMR_VSA1 1.969488896593528
LabuteASA SMR_VSA3 1.642174875121004
LabuteASA SMR_VSA5 1.4830742576469516
LabuteASA SMR_VSA10 1.456982855436134
LabuteASA SlogP_VSA1 1.4242263181170007
LabuteASA SlogP_VSA2 1.4644977816265425
LabuteASA SlogP_VSA3 1.4447496033221796
LabuteASA SlogP_VSA5 1.3794957092452336
LabuteASA VSA_EState5 1.9002051988453037
LabuteASA VSA_EState9 1.1897542861800188
LabuteASA HBA 1.0472814418514973
LabuteASA HBD 1.031473953141147
LabuteASA Rotatable_num 1.5884696120667117
LabuteASA atom_len 2.7071984009167207
LabuteASA atom_size 2.883187261432441
LabuteASA_norm LabuteASA_norm 4.591352602169483
PEOE_VSA1 PEOE_VSA1 4.590246781374056
PEOE_VSA1 PEOE_VSA2 1.5669161585056803
PEOE_VSA1 PEOE_VSA7 1.0617696833174208
PEOE_VSA1 PEOE_VSA8 1.1408362416002311
PEOE_VSA1 PEOE_VSA9 1.1652834520701563
PEOE_VSA1 PEOE_VSA10 1.4372292110820386
PEOE_VSA1 PEOE_VSA11 1.4427248570079192
PEOE_VSA1 PEOE_VSA12 1.4624788563544715
PEOE_VSA1 SMR_VSA1 1.5572337818628024
PEOE_VSA1 SMR_VSA3 1.6908017015694619
PEOE_VSA1 SMR_VSA4 1.0905668913674502
PEOE_VSA1 SMR_VSA5 1.2916375281342718
PEOE_VSA1 SMR_VSA10 1.500232526529309
PEOE_VSA1 SlogP_VSA1 2.156802631749816
PEOE_VSA1 SlogP_VSA2 1.8189224664434334
PEOE_VSA1 SlogP_VSA3 1.4412299622098312
PEOE_VSA1 SlogP_VSA5 1.1492271863936434
PEOE_VSA1 SlogP_VSA7 1.5215808813961311
PEOE_VSA1 VSA_EState5 1.3403385846574887
PEOE_VSA1 VSA_EState9 1.0225552644787854
PEOE_VSA1 HBA 1.47068272381451
PEOE_VSA1 HBD 1.8219078375916475
PEOE_VSA1 Rotatable_num 1.3354691503118237
PEOE_VSA1 atom_len 1.4388534828604018
PEOE_VSA1 atom_size 1.5415725093657813
PEOE_VSA2 PEOE_VSA2 3.9434411479603586
PEOE_VSA2 PEOE_VSA10 1.2939165237836523
PEOE_VSA2 PEOE_VSA11 1.884750692551366
PEOE_VSA2 PEOE_VSA12 1.6858732175754145
PEOE_VSA2 SMR_VSA1 1.5100329975697415
PEOE_VSA2 SMR_VSA2 1.2137289749785742
PEOE_VSA2 SMR_VSA3 1.4459570688534182
PEOE_VSA2 SMR_VSA5 1.110675403271242
PEOE_VSA2 SMR_VSA10 2.017507339817892
PEOE_VSA2 SlogP_VSA1 1.5175202253569806
PEOE_VSA2 SlogP_VSA2 1.391503816565715
PEOE_VSA2 SlogP_VSA3 2.3232307960907423
PEOE_VSA2 SlogP_VSA4 1.2027052257868291
PEOE_VSA2 SlogP_VSA7 1.0760879574886175
PEOE_VSA2 VSA_EState5 1.155697789148933
PEOE_VSA2 HBA 1.3310455040467637
PEOE_VSA2 HBD 1.2216802821601727
PEOE_VSA2 Rotatable_num 1.1586271194828401
PEOE_VSA2 atom_len 1.2833157515308364
PEOE_VSA2 atom_size 1.4041770096557074
PEOE_VSA3 PEOE_VSA3 1.1478160713234695
PEOE_VSA4 PEOE_VSA4 1.4769072377905124
PEOE_VSA4 SMR_VSA6 1.1377848388000436
PEOE_VSA4 VSA_EState10 1.1717820174562394
PEOE_VSA6 PEOE_VSA6 4.349016211021568
PEOE_VSA6 PEOE_VSA7 1.090075431588149
PEOE_VSA6 SMR_VSA4 1.6093919220016089
PEOE_VSA6 SMR_VSA5 1.1007932231073327
PEOE_VSA6 SMR_VSA7 1.519488624316711
PEOE_VSA6 SlogP_VSA3 1.0767998210720733
PEOE_VSA6 SlogP_VSA4 1.6169966417952975
PEOE_VSA6 SlogP_VSA6 1.5295592708275043
PEOE_VSA6 VSA_EState9 1.0261715977102466
PEOE_VSA7 PEOE_VSA7 4.5907762256753335
PEOE_VSA7 PEOE_VSA8 1.2693191315574435
PEOE_VSA7 PEOE_VSA10 1.2068682982684644
PEOE_VSA7 SMR_VSA1 1.1518289086955935
PEOE_VSA7 SMR_VSA3 1.0964403994703904
PEOE_VSA7 SMR_VSA5 1.3243101340593897
PEOE_VSA7 SMR_VSA10 1.0821636765614993
PEOE_VSA7 SlogP_VSA1 1.024982174444465
PEOE_VSA7 SlogP_VSA2 1.0746491699226544
PEOE_VSA7 SlogP_VSA5 1.3358545218783215
PEOE_VSA7 VSA_EState5 1.3056452644804017
PEOE_VSA7 VSA_EState9 1.3643331674766823
PEOE_VSA7 Rotatable_num 1.0618047112238835
PEOE_VSA7 atom_len 1.225327887257014
PEOE_VSA7 atom_size 1.1678503598879144
PEOE_VSA8 PEOE_VSA8 4.591227069926198
PEOE_VSA8 PEOE_VSA10 1.2244941815235746
PEOE_VSA8 PEOE_VSA12 1.1013486430984645
PEOE_VSA8 SMR_VSA1 1.2805928519999679
PEOE_VSA8 SMR_VSA3 1.326520461370928
PEOE_VSA8 SMR_VSA5 1.4007670506955336
PEOE_VSA8 SMR_VSA10 1.208818999400239
PEOE_VSA8 SlogP_VSA1 1.0926690222121163
PEOE_VSA8 SlogP_VSA2 1.168217576875339
PEOE_VSA8 SlogP_VSA3 1.0612512581899003
PEOE_VSA8 SlogP_VSA5 1.4042393574369205
PEOE_VSA8 VSA_EState5 1.4494112023530532
PEOE_VSA8 VSA_EState9 1.2902064534933404
PEOE_VSA8 Rotatable_num 1.138993334175053
PEOE_VSA8 atom_len 1.4399470527551324
PEOE_VSA8 atom_size 1.3834316192105838
PEOE_VSA9 PEOE_VSA9 4.590303472170754
PEOE_VSA9 PEOE_VSA10 1.0621244523358797
PEOE_VSA9 PEOE_VSA12 1.0378502407436179
PEOE_VSA9 SMR_VSA1 1.0285869229527245
PEOE_VSA9 SMR_VSA3 1.0675095855655024
PEOE_VSA9 SMR_VSA10 1.0416118270308752
PEOE_VSA9 SlogP_VSA2 1.0978436516450458
PEOE_VSA9 SlogP_VSA3 1.1333824522875269
PEOE_VSA9 atom_len 1.0085963384673065
PEOE_VSA9 atom_size 1.0376537723192003
PEOE_VSA10 PEOE_VSA10 4.472862283986104
PEOE_VSA10 PEOE_VSA11 1.0568921781112932
PEOE_VSA10 PEOE_VSA12 1.460919756435927
PEOE_VSA10 SMR_VSA1 1.651590578481104
PEOE_VSA10 SMR_VSA3 1.4824440840173516
PEOE_VSA10 SMR_VSA5 1.4487085012838137
PEOE_VSA10 SMR_VSA10 1.4005295168991851
PEOE_VSA10 SlogP_VSA1 1.3485953743832586
PEOE_VSA10 SlogP_VSA2 1.5045237514876009
PEOE_VSA10 SlogP_VSA3 1.3857742317122441
PEOE_VSA10 SlogP_VSA5 1.2719897750389255
PEOE_VSA10 VSA_EState5 1.4795546941546407
PEOE_VSA10 VSA_EState9 1.0855198353347832
PEOE_VSA10 HBA 1.0951615628809221
PEOE_VSA10 Rotatable_num 1.4123395167667983
PEOE_VSA10 atom_len 1.589824423532997
PEOE_VSA10 atom_size 1.6381533039144345
PEOE_VSA11 PEOE_VSA11 4.013228284863734
PEOE_VSA11 PEOE_VSA12 1.3584391035174248
PEOE_VSA11 SMR_VSA1 1.13687994195167
PEOE_VSA11 SMR_VSA2 1.1391230681849804
PEOE_VSA11 SMR_VSA3 1.6707960304752072
PEOE_VSA11 SMR_VSA10 1.3221721896956367
PEOE_VSA11 SlogP_VSA1 1.8293988374105563
PEOE_VSA11 SlogP_VSA2 1.0821856823042897
PEOE_VSA11 SlogP_VSA3 1.1914584527272358
PEOE_VSA11 SlogP_VSA4 1.1958176966209602
PEOE_VSA11 SlogP_VSA7 2.174341950947814
PEOE_VSA11 VSA_EState5 1.0383512059307132
PEOE_VSA11 HBD 1.1804472374990416
PEOE_VSA11 Rotatable_num 1.0855959737503513
PEOE_VSA11 atom_len 1.12885552342741
PEOE_VSA11 atom_size 1.1791895820699
PEOE_VSA12 PEOE_VSA12 4.201179962870731
PEOE_VSA12 SMR_VSA1 1.5682209121103539
PEOE_VSA12 SMR_VSA3 1.4994112741963233
PEOE_VSA12 SMR_VSA5 1.2291717414803094
PEOE_VSA12 SMR_VSA10 1.6978608285185837
PEOE_VSA12 SlogP_VSA1 1.4512688133531528
PEOE_VSA12 SlogP_VSA2 1.4945456269264388
PEOE_VSA12 SlogP_VSA3 1.7271479227056712
PEOE_VSA12 SlogP_VSA5 1.1363483612558982
PEOE_VSA12 VSA_EState5 1.3144719269216807
PEOE_VSA12 HBA 1.1052002944946209
PEOE_VSA12 Rotatable_num 1.2806203190786356
PEOE_VSA12 atom_len 1.4569419980748826
PEOE_VSA12 atom_size 1.532004797312305
PEOE_VSA13 PEOE_VSA13 1.6237675778237493
PEOE_VSA14 PEOE_VSA14 1.4058092857280036
PEOE_VSA14 SlogP_VSA8 1.3655925089852283
SMR_VSA1 SMR_VSA1 4.591348133997853
SMR_VSA1 SMR_VSA3 1.5297484251597284
SMR_VSA1 SMR_VSA5 1.5757447921634666
SMR_VSA1 SMR_VSA10 1.5045816466134558
SMR_VSA1 SlogP_VSA1 1.43715282078732
SMR_VSA1 SlogP_VSA2 1.5075409644564663
SMR_VSA1 SlogP_VSA3 1.5193084092219156
SMR_VSA1 SlogP_VSA5 1.4149206168500448
SMR_VSA1 VSA_EState5 1.7689775555685952
SMR_VSA1 VSA_EState9 1.1518544752238074
SMR_VSA1 HBA 1.198067600115514
SMR_VSA1 HBD 1.0541721478757777
SMR_VSA1 Rotatable_num 1.7527944666457291
SMR_VSA1 atom_len 2.0157415443981894
SMR_VSA1 atom_size 1.9825098117358806
SMR_VSA2 SMR_VSA2 3.572176851153755
SMR_VSA2 SlogP_VSA4 1.0239356509937692
SMR_VSA3 SMR_VSA3 4.416385422288541
SMR_VSA3 SMR_VSA5 1.2829131509741385
SMR_VSA3 SMR_VSA10 1.5683009750557906
SMR_VSA3 SlogP_VSA1 2.4083986215409623
SMR_VSA3 SlogP_VSA2 1.5144218831692977
SMR_VSA3 SlogP_VSA3 1.4048579626686113
SMR_VSA3 SlogP_VSA5 1.1957390678965536
SMR_VSA3 SlogP_VSA7 1.1645499750804162
SMR_VSA3 VSA_EState5 1.4570810259248272
SMR_VSA3 VSA_EState9 1.055519259890057
SMR_VSA3 HBA 1.0284655138007297
SMR_VSA3 HBD 1.2819331673615235
SMR_VSA3 Rotatable_num 1.453407495748588
SMR_VSA3 atom_len 1.6022317569041693
SMR_VSA3 atom_size 1.6543889399668623
SMR_VSA4 SMR_VSA4 3.7402477033932064
SMR_VSA4 SMR_VSA5 1.1967274020157521
SMR_VSA4 SlogP_VSA1 1.8508709658533276
SMR_VSA4 SlogP_VSA4 2.1756098277607725
SMR_VSA4 SlogP_VSA5 1.0302032478502532
SMR_VSA4 VSA_EState9 1.0023815061376316
SMR_VSA5 SMR_VSA5 4.591090190993973
SMR_VSA5 SMR_VSA10 1.2523984995643045
SMR_VSA5 SlogP_VSA1 1.207036545157847
SMR_VSA5 SlogP_VSA2 1.2912744346341758
SMR_VSA5 SlogP_VSA3 1.1825430415799625
SMR_VSA5 SlogP_VSA5 2.1791723249511437
SMR_VSA5 VSA_EState5 1.7109540929648002
SMR_VSA5 VSA_EState9 1.407835524364708
SMR_VSA5 Rotatable_num 1.4650168498263605
SMR_VSA5 atom_len 1.5743864651015542
SMR_VSA5 atom_size 1.4281588963412775
SMR_VSA6 SMR_VSA6 4.11521189251
SMR_VSA6 SMR_VSA10 1.0338621582542895
SMR_VSA6 VSA_EState10 1.1039920451658518
SMR_VSA7 SMR_VSA7 2.9613764554424686
SMR_VSA7 SlogP_VSA3 1.3883601983490959
SMR_VSA7 SlogP_VSA6 2.8722932178821723
SMR_VSA10 SMR_VSA10 4.583992314991691
SMR_VSA10 SlogP_VSA1 1.4365453767489242
SMR_VSA10 SlogP_VSA2 1.4959266600625745
SMR_VSA10 SlogP_VSA3 1.7261165341953189
SMR_VSA10 SlogP_VSA5 1.1649144163442975
SMR_VSA10 VSA_EState5 1.306002106320265
SMR_VSA10 VSA_EState9 1.0469644291174134
SMR_VSA10 HBA 1.2886973377136948
SMR_VSA10 HBD 1.0948193711644745
SMR_VSA10 Rotatable_num 1.2511082222218013
SMR_VSA10 atom_len 1.4061806928421612
SMR_VSA10 atom_size 1.5266978501245787
SlogP_VSA1 SlogP_VSA1 4.349181896309055
SlogP_VSA1 SlogP_VSA2 1.3062667947312627
SlogP_VSA1 SlogP_VSA3 1.2757687035342706
SlogP_VSA1 SlogP_VSA5 1.132046799379927
SlogP_VSA1 SlogP_VSA7 2.083833450409027
SlogP_VSA1 VSA_EState5 1.3356677388133054
SlogP_VSA1 VSA_EState9 1.0112340612092137
SlogP_VSA1 HBA 1.0068832218578092
SlogP_VSA1 HBD 1.4366667090505605
SlogP_VSA1 Rotatable_num 1.4691666728911714
SlogP_VSA1 atom_len 1.4235132468088372
SlogP_VSA1 atom_size 1.435255039150362
SlogP_VSA2 SlogP_VSA2 4.591305544980676
SlogP_VSA2 SlogP_VSA3 1.4042771037979478
SlogP_VSA2 SlogP_VSA5 1.1492663521952786
SlogP_VSA2 VSA_EState5 1.2993433293462753
SlogP_VSA2 VSA_EState9 1.0100860089509645
SlogP_VSA2 HBA 1.439937524100712
SlogP_VSA2 HBD 1.173686005793349
SlogP_VSA2 Rotatable_num 1.2123928398121917
SlogP_VSA2 atom_len 1.417956168577019
SlogP_VSA2 atom_size 1.5155441035519714
SlogP_VSA3 SlogP_VSA3 4.307988050594554
SlogP_VSA3 SlogP_VSA5 1.0747375739176042
SlogP_VSA3 SlogP_VSA6 1.3874108482435679
SlogP_VSA3 VSA_EState5 1.2067287102870765
SlogP_VSA3 HBA 1.3469546441635314
SlogP_VSA3 Rotatable_num 1.1838500688622882
SlogP_VSA3 atom_len 1.360520120191305
SlogP_VSA3 atom_size 1.5287443144869237
SlogP_VSA4 SlogP_VSA4 3.1771688791580277
SlogP_VSA5 SlogP_VSA5 4.591296440282138
SlogP_VSA5 VSA_EState5 1.6994464863782324
SlogP_VSA5 VSA_EState9 1.534001242499668
SlogP_VSA5 Rotatable_num 1.3609997496573139
SlogP_VSA5 atom_len 1.474043858058968
SlogP_VSA5 atom_size 1.3070466258113842
SlogP_VSA6 SlogP_VSA6 2.972298257014037
SlogP_VSA7 SlogP_VSA7 3.3666004519064203
SlogP_VSA7 HBD 1.1302916341712532
SlogP_VSA8 SlogP_VSA8 2.0145265905812324
VSA_EState4 VSA_EState4 1.083657561332355
VSA_EState5 VSA_EState5 4.591354836255298
VSA_EState5 VSA_EState9 1.4217333753054835
VSA_EState5 Rotatable_num 1.6736752992708046
VSA_EState5 atom_len 2.23388314519224
VSA_EState5 atom_size 1.7167222998011005
VSA_EState9 VSA_EState9 4.591354836255298
VSA_EState9 Rotatable_num 1.0779853498497824
VSA_EState9 HBA_norm 1.0626565495364606
VSA_EState9 atom_len 1.2489201101801295
VSA_EState9 atom_size 1.1372370687998343
VSA_EState10 VSA_EState10 2.2932657935370835
HBA HBA 3.3352466593574803
HBA HBD 1.1923181169229025
HBA Rotatable_num 1.0073000689454776
HBA atom_size 1.1323292562870797
HBD HBD 3.3650264614406393
HBD Rotatable_num 1.0460688752366092
HBD atom_size 1.094080216427391
Rotatable_num Rotatable_num 4.331408797163757
Rotatable_num atom_len 1.7129734565406662
Rotatable_num atom_size 1.520427437640296
HBA_norm HBA_norm 4.582522478770452
HBA_norm HBD_norm 1.2992823909342377
HBA_norm atom_len 1.5802045769127537
HBD_norm HBD_norm 4.584331825649317
HBD_norm atom_len 1.5510453333741703
atom_len atom_len 4.573351157061623
atom_len atom_size 2.26409230306171
atom_size atom_size 4.586303935225087
'''
