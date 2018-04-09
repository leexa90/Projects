import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from scipy import histogram, digitize, stats, mean, std
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
    plt.imshow(D);plt.savefig('cluster_peptide.png',dpi=500, bbox_inches='tight');#plt.show()
    Z = sch.dendrogram(Y,orientation='right')
    plt.yticks(np.array(range(1,1+len(index)))*10-5,sorted_keys, fontsize=2)
    plt.savefig('cluster_peptide.png',dpi=500)

def get_surface_area(smile):
    print (smile[-25:])
    mol0 = Chem.MolFromSmiles(smile)
    mol = Chem.AddHs(mol0)
    AllChem.Compute2DCoords(mol)
    adj = (Chem.GetDistanceMatrix(mol)==1)*1
    adj2 = (Chem.GetDistanceMatrix(mol)==2)*1
    molMMFF = AllChem.MMFFGetMoleculeProperties(mol)
    # Chem.MolSurf._LabuteHelper(mol) indiv contribution of surface area
    atoms = list(
                            map(lambda x: molMMFF.GetMMFFAtomType(x),
                                range(len(mol.GetAtoms()))
                                 )
                            )
    AllChem.ComputeGasteigerCharges(mol)
    charges = np.array([float(mol.GetAtomWithIdx(x).GetProp('_GasteigerCharge')) for x in range(len(atoms))])
    surf= np.array(Chem.MolSurf._LabuteHelper(mol))
    return (charges,surf[1:],atoms)
test['charge_surf'] = test['SMILES'].apply( get_surface_area)
bins = [-999,-0.3 , -0.25, -0.2 , -0.15, -0.1 , -0.05,  0.  ,  0.05,  0.1 ,
        0.15,  0.2 ,  0.25,  0.3 ,999]
if True:
    MMFF_dictt = pd.read_csv('res_charge.csv')
    charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
    for atom in [35, 71, 15, 11, 66, 39, 23, 34, 64, 16, 21, 63,  2, 29, 41, 57, 32,
            6, 56, 36,  7, 10,  3, 28, 37,  1,  5]:
        x=plt.hist(charge_csv[charge_csv['atoms']==atom]['charge'].values,50)
        plt.xticks(x[1], rotation='vertical',fontsize=7)
        plt.savefig('1_'+str(atom)+'.png',bbox_inches='tight',dpi3=500);plt.close()
        for num in [3,4,5,8,11,15,50,]:
            if num ==50:
                bins = histogram(charge_csv[charge_csv['atoms']==atom]['charge'].values,num)
                def process_bins(bins):
                    num = bins[0]
                    gaps = bins[1]
                    gaps[-1] = gaps[-1]+0.001
                    gaps = list(map(lambda x :np.round(x,6),gaps))
                    result,temp = [],[]
                    for i in range(len(num)):
                        if num[i] != 0:
                            temp += [gaps[i],gaps[i+1],]
                        else:
                            if temp != []:
                                result += [[temp[0],temp[-1]],]
                            temp = []
                    return result
                bins2= process_bins(bins)
            else:
                bins = histogram(charge_csv[charge_csv['atoms']==atom]['charge'].values,num)[1]
                bins[-1] = bins[-1] + 0.001
                bins = list(map(lambda x :np.round(x,4),bins))
                bins2 = []
                for i in range(len(bins)-1):
                    bins2 += [[bins[i],bins[i+1]],]
            for i in bins2:
                def func(x,i=i,atom=atom):
                    sum_ =  np.sum(x[1][(x[0]  >= i[0] )&(x[0]  <= i[1] ) &(np.array(x[2])==atom)])
                    if sum_ >= 0:
                        return sum_
                    else :
                        return 0.0
                def func_norm(x,i=i,atom=atom):
                    sum_ =  np.sum(x[1][(x[0]  >= i[0] )&(x[0]  <= i[1] ) &(np.array(x[2])==atom)])
                    if sum_ >= 0:
                        return sum_/np.sum(x[1])
                    else :
                        return 0.0
                test['charge_%s_%s_%s'%(atom,i[0],i[1])] = test['charge_surf'].apply(func)
                test['charge_norm_%s_%s_%s'%(atom,i[0],i[1])] = test['charge_surf'].apply(func_norm)
                if np.mean(test['charge_%s_%s_%s'%(atom,i[0],i[1])])==0:
                    del test['charge_%s_%s_%s'%(atom,i[0],i[1])]
                    print ('charge_%s_%s_%s'%(atom,i[0],i[1]))

#weight matrix
if True:
    cluster = fcluster(Y, .66, criterion='distance')
    test = test.set_value(test.index,'weight0',cluster)
    test['weight'] = 1.0/test['weight0'].map(collections.Counter(fcluster(Y, .66, criterion='distance')))

# get which cluster is it in as a intercept #
for i in [3,]:#[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]: 
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
for i in [('W','F','Y','pff'),('N','Q'),('L','I','V','NL'),
          ('S8','R8','B8'),('R5','S5'),('PEG2','PEG5','PEG1'),
          ('H','R','K'),('S','T'),('S8','R8'),('S8','B8'),('R8','B8')]:
    dictt[i] = 999
          
#single res
for res in dictt.keys():
    if dictt[res] >= 20:
        test['%s_num' %str(res)] = test['list'].apply(lambda x : sum(np.in1d(x,np.array(res))))
        test['%s_norm' %str(res)] = 1.0*test['%s_num'%str(res)]/test['res_list']
        impt += ['%s_norm' %str(res),'%s_num' %str(res)]
#double res
def func_compare2(x,gap,res): #find motiffs
    for i in range(len(res)):
        if i==0:
            value = np.in1d(x,np.array(res[0]))*1
        else:
            value = np.in1d(x,np.array(res[1]))*1
    return np.sum(value)
    
##for gap in [1,2,3,4,5]:
##    for res1 in dictt.keys():
##        if dictt[res1] >= 20:
##            for res2 in dictt.keys():
##                if dictt[res2] >= 20:
##                    test['%s__%s_%s_num' %(res1,res2,gap)] = test['list'].apply(lambda x : func_compare2(x,gap,(res1,res2)))
##                    test['%s__%s_%s_norm' %(res1,res2,gap)] = test['%s__%s_%s_num' %(res1,res2,gap)]/test['res_list']
##                    impt += ['%s__%s_%s_num' %(res1,res2,gap),'%s__%s_%s_norm' %(res1,res2,gap)]
##for i in tuple(impt):
##    if 'norm' in i :
##        test[i+'_cat'] = (test[i] != 0)*1
##        impt  +=  [i+'_cat',]
for i in impt:
    if np.std(test[i])==0:
        del test[i]
dictt_name = {}
if True:
    clusters = [x for x in test.keys() if 'cluster_' in x]
    results = []
    from sklearn.metrics import r2_score
##    def r2_score(y_true,y_pred,sample_weight):
##        mean = sum(y_true*sample_weight)/sum(sample_weight)
##        r2_model = sum(sample_weight*(y_true-y_pred)**2)
##        r2_all = sum(sample_weight*(y_true-mean)**2)
##        return 1-r2_model/r2_all
    for i in impt+list(test.keys()):#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
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
                preds = clf.fit(temptrain,np.log10(test['10']),sample_weight=test['weight']).predict(temptrain)
                se = np.sum((preds-np.log10(test['10']))**2)/np.matmul(temptrain.T,temptrain)[-1,-1]
                se = 2*(se/np.sum(test['weight']))**.5
                ids_non_zero = test[test[i]!=0].index
                if len(i.split('_')) == 2 and (clf.coef_[0]-se >0 or clf.coef_[0]+se <0):
                    dictt_name[i] = (i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
                             clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)
                    results += [(i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
                                 clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)]
                if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
                    if '__' in i:
                        if sum(
                            np.array((dictt_name[i.split('_')[0]+'_norm'][2],
                                         dictt_name[i.split('_')[2]+'_norm'][2]))
                            >= r2_score(np.log10(test['10']),preds)
                            ):
                            results += [(i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
                                     clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)]
            except : print (i)
    print ([x[:3] for x in sorted(results,key =  lambda x : x[1])[-20:]])
#print (dictt_name)
reg_coef = pd.DataFrame(results,columns= \
                        ['name','simple LR model R2','corr coef','Se','CI','present in ids']).\
                        sort_values('simple LR model R2').reset_index(drop=True)

if True:
    for i in [x[:3] for x in sorted(results,key =  lambda x : x[1])[-160:]]:
            if i[2] >= 0:
                    print (i[0][:-9]+str(i[2])[0:5])
    for i in [x[:3] for x in sorted(results,key =  lambda x : x[1])[-160:]]:
            if i[2] <= 0:
                    print (i[0][:-9]+str(i[2])[0:6])
def log12(x):
	return np.log10(x)/np.log10(1.5)
test['int']=(log12(test['10']).values-13).astype(np.int32)
if False:
    f1=open('/media/leexa/97ba6a6b-3f4d-4528-84ca-50200ba4594f/Projects/toxicity/Final/Permability/all_seq/staple/j3/miserables2.json','w')
    f1.write("""{\n "nodes":[\n""")
    for i in range(0,len(test['1'])):
        if i != len(test)-1:
            text = '''    {"name":"%s_%s","chemName":"%s","group":%s},\n''' %(str(i),test['1'].iloc[i],test['1'].iloc[i],test['int'].iloc[i])
        else: text = '''    {"name":"%s_%s","chemName":"%s","group":%s}\n''' %(str(i),test['1'].iloc[i],test['1'].iloc[i],test['int'].iloc[i])
        f1.write(text)
    f1.write('  ],\n  "links":[\n')
    for i in range(len(test)-1):
        for j in range(i+1,len(test)):
            if i == len(test)-2:
                f1.write('''    {"source":%s,"target":%s,"value":%s}\n'''%(i,j,corr_mat[i,j]))
            else:
                f1.write('''    {"source":%s,"target":%s,"value":%s},\n'''%(i,j,corr_mat[i,j]))
    f1.write('''  ]\n}''')
    f1.close()
