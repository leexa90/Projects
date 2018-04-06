import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
x=charge_csv[(charge_csv['atoms'] == 28) & (charge_csv['bond1'] == '[10]')\
             & (charge_csv['bond2'] == '[1, 3]') \
             & (charge_csv['charge'] >= 0.164) \
             & (charge_csv['charge'] <= 0.1652)]['origin'].values
test = pd.read_csv('../stapled_peptide_permability_features.csv')
temp=test.iloc[list(set(x))]

def process_string(str,len=len):
    result = []
    special_resi = False
    for i in str:
        if i =='(':
            return len(result)
        elif special_resi and i !='-': #inside special residue
            temp += i
        elif i =='-':
            if special_resi : #closing of special residue
                result += [temp,]
                special_resi = False
                temp  = ''
            else: #starting of special residue
                special_resi = True 
                temp = ''
        else: #normal residue
            result += [i,]
            #if i in ['Q','N']:
                #result += [i,]
    return len(result)
if True:
    test['len']=test['1'].apply(process_string)
    test['res_list']=test['1'].apply(lambda x : process_string(x,list))
    test['list']=test['1'].apply(lambda x : np.array(process_string(x,list)))
    test['res_list_QN']=test['res_list'].apply(lambda x : collections.Counter(x)['Q']+collections.Counter(x)['N'])
    test['res_list']=test['res_list'].apply(len)
    dictt = collections.Counter(np.concatenate(test['list'].values))
print (dictt)
dictt_counter = {}
counter =0
string='abcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()'
for i in sorted(dictt,key=lambda x : dictt[x])[::-1]:
	dictt_counter[i] = string[counter]
	counter += 1
def func(x):
    result = ''
    for i in x:
        result += dictt_counter[i]
    return result
test['string'] = test['list'].apply(func)

corr_mat = np.zeros((217,217))

# Import pairwise2 module
from Bio import pairwise2

# Import format_alignment method
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned
for i in range(217):
    for j in range(i,217):
        alignments = pairwise2.align.globalms(test.iloc[i]['string'],
                                             test.iloc[j]['string'],
                                              2, -1, -0.5, -0.1)
        score = max([0.5*a[-3]/a[-1] for a in alignments])
        corr_mat[i,j] = score
        corr_mat[j,i] = score
if True:
    heat_map = corr_mat
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.hierarchy import fcluster
    Y = sch.linkage(heat_map,method='single')
    Z = sch.dendrogram(Y,orientation='right')
    index = Z['leaves']
    D = heat_map[index,:]
    D = D[:,index]
    sorted_keys = [test.iloc[x]['1'] for x in index]
    plt.close()
    plt.xticks(range(len(index)),sorted_keys,rotation='vertical', fontsize=2)
    plt.yticks(range(len(index)),sorted_keys, fontsize=2)
    plt.imshow(D);plt.savefig('cluster_peptide.png',dpi=500, bbox_inches='tight');plt.show()
    Z = sch.dendrogram(Y,orientation='right')
    plt.yticks(np.array(range(1,1+len(index)))*10-5,sorted_keys, fontsize=2)
    plt.savefig('cluster_peptide.png',dpi=500)

# get which cluster is it in as a intercept #
for i in [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]: 
    test['cluster_%s'%i] = test.set_value(index,'cluster_%s'%i,fcluster(Y,i, criterion='distance'))['cluster_%s'%i]
    print (i,max(fcluster(Y,i, criterion='distance')))
    #class the small clusters into -1 
    temp = test.groupby('cluster_%s'%i).apply(len).reset_index()
    id_minus1 = test[test['cluster_%s'%i].isin(temp[temp[0] <5]['cluster_%s'%i].values)].index
    test = test.set_value(id_minus1,'cluster_%s'%i,-1)
    print (test.groupby('cluster_%s'%i).apply(len).reset_index())
    
        
        

def process_string(str,len=len):
    result = []
    special_resi = False
    for i in str:
        if i =='(':
            return len(result)
        elif special_resi and i !='-': #inside special residue
            temp += i
        elif i =='-':
            if special_resi : #closing of special residue
                result += [temp,]
                special_resi = False
                temp  = ''
            else: #starting of special residue
                special_resi = True 
                temp = ''
        else: #normal residue
            result += [i,]
            #if i in ['Q','N']:
                #result += [i,]
    return len(result)
if True:
    test['len']=test['1'].apply(process_string)
    test['res_list']=test['1'].apply(lambda x : process_string(x,list))
    test['list']=test['1'].apply(lambda x : np.array(process_string(x,list)))
    test['res_list_QN']=test['res_list'].apply(lambda x : collections.Counter(x)['Q']+collections.Counter(x)['N'])
    test['res_list']=test['res_list'].apply(len)
    dictt = collections.Counter(np.concatenate(test['list'].values))
print (dictt)
impt = []
#single res
for res in dictt.keys():
    if dictt[res] >= 20:
        test['%s_num' %res] = test['list'].apply(lambda x : len(x[x==res]))
        test['%s_norm' %res] = 1.0*test['%s_num'%res]/test['res_list']
        impt += ['%s_norm' %res,'%s_num' %res]
#double res
def func_compare2(x,gap,res):
    for i in range(len(res)):
        if i==0:
            value = (x[:-gap]==res[i])*1
        else:
            value = value * ([x[gap:]==res[i]])
    return np.sum(value)
    
for gap in [1,2,3,4,5]:
    for res1 in dictt.keys():
        if dictt[res1] >= 20:
            for res2 in dictt.keys():
                if dictt[res2] >= 20:
                    test['%s_%s_%s_num' %(res1,res2,gap)] = test['list'].apply(lambda x : func_compare2(x,gap,(res1,res2)))
                    test['%s_%s_%s_norm' %(res1,res2,gap)] = test['%s_%s_%s_num' %(res1,res2,gap)]/test['res_list']
                    impt += ['%s_%s_%s_num' %(res1,res2,gap),'%s_%s_%s_norm' %(res1,res2,gap)]
for i in tuple(impt):
    if 'norm' in i :
        test[i+'_cat'] = (test[i] != 0)*1
        impt  +=  [i+'_cat',]
for i in impt:
    if np.std(test[i])==0:
        del test[i]
if True:
    clusters = [x for x in test.keys() if 'cluster_' in x]
    results = []
    from sklearn.metrics import r2_score
    for i in impt:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
        for cluster in clusters:
            try:
                X,Y1,Y2 = [],[],[]
                from sklearn import linear_model,preprocessing ,decomposition
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import Lasso
                alpha =0.00#alpha#10**alpha
                clf = linear_model.LinearRegression()
                temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
                dummie = pd.get_dummies(test[cluster].values,drop_first= True).values
                temptrain = np.concatenate([dummie,temptrain],1)
                preds = clf.fit(temptrain,np.log10(test['10'])).predict(temptrain)
                se = np.sum((preds-np.log10(test['10']))**2)/np.matmul(temptrain.T,temptrain)[-1,-1]
                se = 3*(se/215)**.5
                ids_non_zero = test[test[i]!=0].index
                if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
                    results += [(i+'_'+cluster,r2_score(np.log10(test['10']),preds),
                                 clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)]
            except : print (i)
    print ([x[:3] for x in sorted(results,key =  lambda x : x[1])[-20:]])
reg_coef = pd.DataFrame(results,columns= \
                        ['name','simple LR model R2','corr coef','Se','CI','present in ids']).\
                        sort_values('simple LR model R2').reset_index(drop=True)



