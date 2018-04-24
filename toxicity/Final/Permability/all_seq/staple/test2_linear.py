import pandas as pd
pd.options.display.max_rows = 20
import numpy as np
import matplotlib.pyplot as plt
import collections
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from scipy import histogram, digitize, stats, mean, std
import sys
sys.path.append('/home/leexa/anaconda2/envs/deepchem/lib/python3.5/site-packages')
import statsmodels.api as sm
charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
x=charge_csv[(charge_csv['atoms'] == 28) & (charge_csv['bond1'] == '[10]')\
             & (charge_csv['bond2'] == '[1, 3]') \
             & (charge_csv['charge'] >= 0.164) \
             & (charge_csv['charge'] <= 0.1652)]['origin'].values
test = pd.read_csv('../stapled_peptide_permability_features.csv')
temp=test.iloc[x]
test = test[test['7'] == 1].reset_index(drop=True)
test = test[test['1'] != '-AC-AHL-R8-LCLEKL-S5-GLV-(K-PEG1--FITC-)'].reset_index(drop=True)
test['20'] = np.log10(test['10'])
#test['err'] = 0.5*np.abs(np.log10(test['8'])-np.log10(test['10']))
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
    test['Aro_Ccycle_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAromaticCarbocycles(Chem.MolFromSmiles(x)))
    test['Aro_Hcycle_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAromaticHeterocycles(Chem.MolFromSmiles(x)))
    test['Aro_Ring_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAromaticRings(Chem.MolFromSmiles(x)))
    test['Ali_Ccycle_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAliphaticCarbocycles(Chem.MolFromSmiles(x)))
    test['Ali_Hcycle_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAliphaticHeterocycles(Chem.MolFromSmiles(x)))
    test['Ali_Ring_num'] = test['SMILES'].apply(lambda x :\
        Descriptors.NumAliphaticRings(Chem.MolFromSmiles(x)))
    for i in [x for x in test.keys() if ('Ali' in x or 'Aro' in x)]:
        test[i[:-4]+'_norm'] = test[i]/test['res_list']
    test['TPSA-DIV-LabuteASA'] = test['TPSA']/test['LabuteASA'] #ratio of polar SA
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

corr_mat = np.zeros((len(test),len(test)))

# Import pairwise2 module
from Bio import pairwise2

# Import format_alignment method
from Bio.pairwise2 import format_alignment

# Define two sequences to be aligned
for i in range(len(test)):
    for j in range(i,len(test)):
        alignments = pairwise2.align.globalms(test.iloc[i]['string'],
                                             test.iloc[j]['string'],
                                              2, -1, -0.5, -0.1)
        alignments = sorted(alignments,key= lambda x : x[-3])[-1]
        score = 2*np.sum(np.array(list(map(lambda x : x,alignments[0])))==np.array(list(map(lambda x : x,alignments[1]))))
        corr_mat[i,j] = score/(len(test.iloc[i]['string'])+len(test.iloc[i]['string']))
        corr_mat[j,i] = score/(len(test.iloc[i]['string'])+len(test.iloc[i]['string']))
if True:
    import scipy.spatial.distance as ssd
    heat_map = abs(corr_mat-1)
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.hierarchy import fcluster
    Y = sch.linkage(ssd.squareform(heat_map),method='centroid')
    Z = sch.dendrogram(Y,orientation='right')
    index = Z['leaves']
    D = heat_map[index,:]
    D = D[:,index]
    sorted_keys = [test.iloc[x]['1'] for x in index]
    plt.close()
    plt.xticks(range(len(index)),sorted_keys,rotation='vertical', fontsize=2)
    plt.yticks(range(len(index)),sorted_keys, fontsize=2)
    plt.imshow(D);plt.savefig('cluster_peptide.png',dpi=500, bbox_inches='tight');#plt.show()
    plt.close()
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
test['charge_num'] = test['charge_surf'].apply(lambda x : np.sum(x[0]))
test['charge_mean'] = test['charge_surf'].apply(lambda x : np.mean(x[0]))
bins = [-999,-0.3 , -0.25, -0.2 , -0.15, -0.1 , -0.05,  0.  ,  0.05,  0.1 ,
        0.15,  0.2 ,  0.25,  0.3 ,999]
if False:
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
                bins = list(map(lambda x :np.round(x,5),bins))
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

for cutoff in [0.1,0.15,0.2,0.3]:
    def makePlot(i) :
        plt.close()
        temp = test[['20',i,'weight0']].groupby('weight0')[i].apply(np.mean).reset_index()
        temp = pd.merge(temp, test[['20',i,'weight0']].groupby('weight0')['20'].apply(np.mean).reset_index(),
                        on = ['weight0'])
        temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
        if 'num' in i or 'res_list' in i or i in ['atom_1', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'atom_16', 'atom_len', 'atom_size']: #do not scale count features. 
            temptrain = test[[i,]].values
            print  (i)
        temptrain2 = np.concatenate([temptrain*0+1,temptrain],1)
        W=np.diag(test['weight'])
        X=temptrain2
        y=np.log10(test['10'])
        B=np.matmul(np.linalg.inv(np.matmul(np.matmul(X.T,W),X)) ,np.matmul(np.matmul(X.T,W),y))
        preds = np.matmul(B,X.T)
        se = np.sum(
            (test['weight']*(preds-np.log10(test['10']))**2)
            )*np.linalg.inv(
                 np.matmul(np.matmul(temptrain2.T,np.diag(test['weight'])),temptrain2)
                 )
        se = (se/(sum(test['weight'])-2))
        err2=np.sum(
            (test['weight']*(preds-np.log10(test['10']))**2)
            )/(sum(test['weight'])-2)
        err2 = err2**.5
        err=np.diag(np.matmul(np.matmul(temptrain2,se),temptrain2.T))**.5
        B = np.round(B,3)
        plt.plot(test[i],preds,'r',label='y = %sX + %s'%(B[1],B[0]));
        plt.plot(temp[i],temp['20'],'bo',label='data points from clusters');
        plt.ylabel('Log flourescence');
        plt.errorbar(test[i],preds,color='green',yerr=err2*2,label='regresion sigma');
        plt.errorbar(test[i],preds,color='red',yerr=err*2);
        axes = plt.axes()
        xlim = axes.get_xlim()
        # example of how to zoomout by a factor of 0.1
        factor = 0.1
        plt.title(i,fontsize=32)
        new_xlim = (xlim[0] + xlim[1])/2 + np.array((-0.5, 0.5)) * (xlim[1] - xlim[0]) * (1 + factor) 
        axes.set_xlim(new_xlim)
        plt.xlabel(i);plt.legend();plt.savefig('%s.png'%(str(i)+str(cutoff)),bbox_inches='tight' );plt.close()#plt.show()
    #weight matrix
    if True:
        cluster = fcluster(Y, cutoff, criterion='distance')
        test = test.set_value(test.index,'weight0',cluster)
        test['weight'] = 1.0/test['weight0'].map(collections.Counter(fcluster(Y, cutoff, criterion='distance')))
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
        #test['res_list_QN']=test['res_list'].apply(lambda x : collections.Counter(x)['Q']+collections.Counter(x)['N'])
        test['res_list']=test['res_list'].apply(len)
        dictt = collections.Counter(np.concatenate(test['list'].values))
    print (dictt)
    impt = []
    dictt_vol = {'FITC': 163.32732360530238, 'PEG2': 75.972, 'pff': 84.691, 'Y': 68.658, 'I': 47.901,
                 'E': 50.492, 'A': 28.806, 'P': 40.74, 'L': 47.901, 'AC': 16.891, 'B8': 71.665, 'B88': 146.863,
                 'S5': 53.418, 'BAla': 28.806, 'PEG': 58.128, 'S': 33.6, 'D': 44.127, 'EEA': 93.299, 'F': 63.863,
                 'R8': 72.513, 'T': 39.965, 'R5': 50.236, 'NL': 47.901, 'N': 44.673, 'Q': 51.038, 'R88': 115.722,
                 'S8': 72.513, 'C': 39.961, 'B5': 71.665, 'G': 22.441, 'K': 53.241, 'R': 63.476, 'PEG1': 58.128,
                 'W': 79.754, 'M': 51.659, 'V': 35.171, 'PEG5': 135.867, 'H': 56.292}
    for i in [('W','F','Y','pff'),('N','Q'),('L','I','V','NL'),('D','E'),
              ('S8','R8','B8'),('R5','S5'),('PEG2','PEG5','PEG1'),
              ('R','K'),('H','R','K'),('S','T'),('S8','R8'),('S8','B8'),('R8','B8'),
              ('W','F','pff'),('F','pff'),('F','Y')]:
        dictt[i] = 999
              
    #single res
    for res in dictt.keys():
        #if dictt[res] >= 20:
            test['%s_num' %str(res)] = test['list'].apply(lambda x : sum(np.in1d(x,np.array(res))))
            test['%s_norm' %str(res)] = 1.0*test['%s_num'%str(res)]/test['res_list']
            #test['%s_normLG' %str(res)] = test['%s_norm' %str(res)].apply(np.log1p)
            #test['%s_normSQ' %str(res)] = test['%s_norm' %str(res)].apply(lambda x : x**2)
            test['%s_normSA' %str(res)] = test['list'].apply(lambda y: np.array(list(map(lambda x : dictt_vol[x],y)))).apply(np.sum) #total area
            test['%s_normSA' %str(res)] = test['list'].apply(lambda y : sum(np.array(list(map(lambda x : dictt_vol[x],y)))[np.in1d(y,np.array(res))]))\
                                          /test['list'].apply(lambda y: np.array(list(map(lambda x : dictt_vol[x],y)))).apply(np.sum)
            impt += ['%s_norm' %str(res),'%s_num' %str(res),'%s_normSA' %str(res),]#'%s_normSQ' %str(res),'%s_normLG' %str(res)]

    #double res
    #test = test[test['R_num'] <= 4].reset_index(drop=True)
    dictt = collections.Counter(np.concatenate(test['list'].values))
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
    test['MW'] = test['len']
    del test['len']
    del test['FITC_norm']
    class StandardScaler(object):
        #my own scaler with weights, verified to give idential values for unweights values to original function
        def fit(self,X,weights=test['weight']):
            if type(weights)==type(None):
                mean = np.mean(X, 0).values
                std = np.std(X, 0).values
            else:
                weights= np.array([weights,]*X.shape[1]).T
                mean = np.sum(np.array(weights)*np.array(X),0)/np.sum(np.array(weights),0)
                std = np.sum(np.array(weights)*((np.array(X)-mean)**2),0)/np.sum(np.array(weights),0)
            self.mean = mean
            self.std = std
            return self
        def transform(self,X):
            return (X.values-self.mean)/self.std
    if True:
        clusters = [x for x in test.keys() if 'cluster_' in x]
        results = []
        from sklearn.metrics import r2_score
    ##    def r2_score(y_true,y_pred,sample_weight):
    ##        mean = sum(y_true*sample_weight)/sum(sample_weight)
    ##        r2_model = sum(sample_weight*(y_true-y_pred)**2)
    ##        r2_all = sum(sample_weight*(y_true-mean)**2)
    ##        return 1-r2_model/r2_all
        for i in list(test.keys())[15:]:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
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
                    ids_non_zero = test[test[i]!=0].index
                    if 'num' in i or 'res_list' in i or i in ['atom_1', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'atom_16', 'atom_len', 'atom_size']: #do not scale count features. 
                        temptrain = test[[i,]].values
                        print  (i)
                        #i='num_' + i
                    temptrain2 = np.concatenate([temptrain*0+1,temptrain],1)
                    #dummie = pd.get_dummies(test[cluster].values,drop_first= True).values
                    #temptrain = np.concatenate([dummie,temptrain],1)
                    preds = clf.fit(temptrain,np.log10(test['10']),sample_weight=test['weight']).predict(temptrain)
                    # equailvalent ~ inv(X.T*W*X)*X.T*W*Y
                    W=np.diag(test['weight'])
                    X=temptrain2
                    y=np.log10(test['10'])
                    B=np.matmul(np.linalg.inv(np.matmul(np.matmul(X.T,W),X)) ,np.matmul(np.matmul(X.T,W),y))
                    preds = np.matmul(B,X.T)
                    # sm.WLS(y, X, weights=test['weight']).fit().summary()
                    se = np.sum(
                        (test['weight']*(preds-np.log10(test['10']))**2)
                        )*np.linalg.inv(
                             np.matmul(np.matmul(temptrain2.T,np.diag(test['weight'])),temptrain2)
                             )[-1,-1]
                    se = 2.0*(se/(sum(test['weight'])-2))**.5
        
                    if  (clf.coef_[0]-1.67*se/2 >0 or clf.coef_[0]+1.67*se/2  <0) and r2_score(np.log10(test['10']),preds,sample_weight=test['weight']) < 0.7:
                        dictt_name[i] = (i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
                                 clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)
                        results += [(i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
                                     clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)]
    ##                if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
    ##                    if '__' in i:
    ##                        if sum(
    ##                            np.array((dictt_name[i.split('_')[0]+'_norm'][2],
    ##                                         dictt_name[i.split('_')[2]+'_norm'][2]))
    ##                            >= r2_score(np.log10(test['10']),preds)
    ##                            ):
    ##                            results += [(i+'_'+cluster,r2_score(np.log10(test['10']),preds,sample_weight=test['weight']),
    ##                                     clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se],ids_non_zero)]
                except : None#print (i)
        print ([x[:3] for x in sorted(results,key =  lambda x : x[1])[-20:]])
    #print (dictt_name)
    reg_coef = pd.DataFrame(results,columns= \
                            ['name','simple LR model R2','corr coef','Se','CI','present in ids']).\
                            sort_values('simple LR model R2').reset_index(drop=True)

    if True:
        for i in [x[:3] for x in sorted(results,key =  lambda x : x[0])[-250:]]:
                if i[2] > 0.01:
                        print (i[0][:-10],str(i[2])[:6],str(i[1])[0:5])
        for i in [x[:3] for x in sorted(results,key =  lambda x : x[0])[-250:]]:
                if i[2] < -0.01:
                        print (i[0][:-10],str(i[2])[:6],str(i[1])[0:6])
    features = []
    if True:
        plt.close()
        counter = 1
        plots = []
        for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
                if i[2] < -0.01 and 'num' in i[0] and '.(' not in i[0]:
                        #makePlot(i[0][:-10])
                        features += [i[0][:-10],]
                        plots += [i,]
                        plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='r')
                        plt.plot([i[-4],],[counter,]*1,'ro')
                        if i[-4]-1.19*i[-3] > 0 or  i[-4]+1.19*i[-3] < 0:
                            txt = i[0][:-10]+'***'
                        elif i[-4]-i[-3] > 0 or i[-4]+i[-3] < 0:
                            txt = i[0][:-10]+'**'
                        else:
                            txt = i[0][:-10]+'*'
                        plt.text(i[-4],counter+.25,txt,horizontalalignment='center',fontsize=7)
                        counter += 1
        try:
            i = [x for x in sorted(results,key =  lambda x : x[-4]) if 'res_list_cluster_3' in x[0]][0]
            #makePlot(i[0][:-10])
            plots += [i,]
            plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='g')
            plt.plot([i[-4],],[counter,]*1,'go')
            if i[-4]-1.19*i[-3] > 0 or  i[-4]+1.19*i[-3] < 0:
                txt = i[0][:-10]+'***'
            elif i[-4]-i[-3] > 0 or i[-4]+i[-3] < 0:
                txt = i[0][:-10]+'**'
            else:
                txt = i[0][:-10]+'*'
            plt.text(i[-4],counter+.25,'Number of residues**',horizontalalignment='center',fontsize=7)
            counter += 1
        except : None
        for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
                if i[2] > 0.01 and 'num' in i[0] and '.(' not in i[0]:
                        #makePlot(i[0][:-10])
                        features += [i[0][:-10],]
                        plots += [i,]
                        plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='b')
                        plt.plot([i[-4],],[counter,]*1,'bo')
                        if i[-4]-1.19*i[-3] > 0 or  i[-4]+1.19*i[-3] < 0:
                            txt = i[0][:-10]+'***'
                        elif i[-4]-i[-3] > 0 or i[-4]+i[-3] < 0:
                            txt = i[0][:-10]+'**'
                        else:
                            txt = i[0][:-10]+'*'
                        plt.text(i[-4],counter+.25,txt,horizontalalignment='center',fontsize=7)
                        counter += 1
        plt.title('Number Features')
        plt.xlabel('Coefficient and 95% CI of feature')
        plt.ylabel('Features')
        plt.yticks([],[])
        plt.ylim([0,counter])
        plt.savefig('B_coef'+str(cutoff)+'.png',dpi=300),plt.show()
        plt.close()
        counter = 1
        plots = []
        for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
                if i[2] < -0.01 and 'num' not in i[0] and 'list' not in i[0] and 'normSA' not in i[0] and 'VSA' not in i[0]:
                        features += [i[0][:-10],]
                        #makePlot(i[0][:-10])
                        plots += [i,]
                        plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='r')
                        plt.plot([i[-4],],[counter,]*1,'ro')
                        if i[-4]-1.19*i[-3] > 0 or  i[-4]+1.19*i[-3] < 0:
                            txt = i[0][:-10]+'***'
                        elif i[-4]-i[-3] > 0 or i[-4]+i[-3] < 0:
                            txt = i[0][:-10]+'**'
                        else:
                            txt = i[0][:-10]+'*'
                        plt.text(i[-4],counter+.25,txt,horizontalalignment='center',fontsize=7)
                        counter += 1
        for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
                if i[2] > 0.01 and 'num' not in i[0] and 'list' not in i[0] and 'normSA' not in i[0] and 'VSA' not in i[0]:
                        features += [i[0][:-10],]
                        #makePlot(i[0][:-10])
                        plots += [i,]
                        plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='b')
                        plt.plot([i[-4],],[counter,]*1,'bo')
                        if i[-4]-1.19*i[-3] > 0 or  i[-4]+1.19*i[-3] < 0:
                            txt = i[0][:-10]+'***'
                        elif i[-4]-i[-3] > 0 or i[-4]+i[-3] < 0:
                            txt = i[0][:-10]+'**'
                        else:
                            txt = i[0][:-10]+'*'
                        plt.text(i[-4],counter+.25,txt,horizontalalignment='center',fontsize=7)
                        counter += 1
        plt.ylim([0,counter])
        plt.title('Normalized (by seq length) Features')
        plt.xlabel('Coefficient and 95% CI of feature')
        plt.ylabel('Percentage Features')
        plt.yticks([],[])
        plt.savefig('B_norm'+str(cutoff)+'.png',dpi=300),plt.show()    


                        
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
    if False:
        plt.close()
        test['res_list2'] = test['res_list']
        bins = [[8,9,10,11,12],[13,14,15],[16,],[17,],[18,],[19,],[21,23,24,25],[26,28,31,38,39]]
        for i in bins:
            test = test.set_value(test[test['res_list2'].isin(i)].index,
                                  'res_list2',
                                  np.mean(i))
            
        test['20'] = np.log10(test['10'])
        val = test.groupby('res_list2')['20'].apply(np.array).reset_index()
        weight = test.groupby('res_list2')['weight'].apply(np.array).reset_index()
        temp = pd.merge(val,weight,on='res_list2')
        temp['mean'] = (temp['20']*temp['weight']).apply(sum)/temp['weight'].apply(sum)
        temp['std'] = (((temp['20']-temp['mean'])**2)*temp['weight']).apply(sum)\
                      /temp['weight'].apply(sum)
        temp['num'] = temp['weight'].apply(sum)
        temp['std'] = temp['std']**.5
        temp = temp[temp['num'] ].reset_index(drop=True)
        temp['se'] = temp['std']/temp['weight'].apply(sum).apply(lambda x : (x-1)**.5).astype(np.float32)
        plt.errorbar(temp['res_list2'].values,
                     temp['mean'].values,
                     yerr=temp['se'].values);
        plt.plot(temp['res_list2'].values,
                 temp['mean'].values,
                 'go')
        plt.ylabel('flourescence')
        plt.xlabel('length of peptide (includes FITC and linker)')
        axes = plt.gca()
        plt.title('Flourscence vs peptide length')
        plt.xlim(min(test['res_list']),max(test['res_list']))
        plt.yticks(axes.get_yticks(),(10**axes.get_yticks()).astype(np.int32))
        plt.xticks([np.mean(x) for x in  bins],[str(x) for x in bins], rotation='vertical')
        print (axes.get_yticks())
        plt.savefig('length'+str(cutoff)+'.png',dpi=300,bbox_inches='tight')
        plt.show()

    if False:
        plt.close()
        test['res_list2'] = (test['5']+0.5)//1
        bins = [[-7,-6,-5,-4,-3], [-2],
                [-1], [0], [1,2], [3,4],  [6], [7,8,9,10]]
        for i in bins:
            test = test.set_value(test[test['res_list2'].isin(i)].index,
                                  'res_list2',
                                  np.mean(i))
            
        test['20'] = np.log10(test['10'])
        val = test.groupby('res_list2')['20'].apply(np.array).reset_index()
        weight = test.groupby('res_list2')['weight'].apply(np.array).reset_index()
        temp = pd.merge(val,weight,on='res_list2')
        temp['mean'] = (temp['20']*temp['weight']).apply(sum)/temp['weight'].apply(sum)
        temp['std'] = (((temp['20']-temp['mean'])**2)*temp['weight']).apply(sum)\
                      /temp['weight'].apply(sum)
        temp['num'] = temp['weight'].apply(sum)
        temp['std'] = temp['std']**.5
        temp = temp[temp['num']].reset_index(drop=True)
        temp['se'] = temp['std']/temp['weight'].apply(sum).apply(lambda x : (x-1)**.5).astype(np.float32)
        plt.errorbar(temp['res_list2'].values,
                     temp['mean'].values,
                     yerr=temp['se'].values);
        plt.plot(temp['res_list2'].values,
                 temp['mean'].values,
                 'go')
        plt.ylabel('flourescence')
        plt.xlabel('length of peptide (includes FITC and linker)')
        axes = plt.gca()
        plt.title('Flourescence vs peptide length')
        plt.xlim(min(test['res_list']),max(test['res_list']))
        plt.yticks(axes.get_yticks(),(10**axes.get_yticks()).astype(np.int32))
        plt.xticks([np.mean(x) for x in  bins],[str(x) for x in bins], rotation='vertical')
        print (axes.get_yticks())
        plt.xlim([np.mean(x)-0.5 for x in  bins][0:1]+[np.mean(x)+0.5 for x in  bins][-1:])
        plt.savefig('charge'+str(cutoff)+'.png',dpi=300,bbox_inches='tight')
        plt.show()

    test['20'] = np.log10(test['10'])
    # DCOR(METRIC) Amitava Roy

    if False:
        def dist_covar(x,y):
            assert len(x)==len(y)
            a=1.0*np.zeros(shape=(len(x),len(x)))
            b=1.0*np.zeros(shape=(len(x),len(x)))
            for i in range(len(x)):
                for j in range(i,len(x)):
                    a[i,j]=np.sqrt(np.sum(x[i]-x[j])**2)
                    a[j,i]=a[i,j]
                    b[i,j]=np.sqrt(np.sum(y[i]-y[j])**2)
                    b[j,i]=b[i,j]
            ma1,ma2,ma3 = np.mean(a,0),np.mean(a,1),np.mean(a)
            mb1,mb2,mb3 = np.mean(b,0),np.mean(b,1),np.mean(b)
            #print (mb1==mb2)
            val = (a-ma1-ma2+ma3)*(b-mb1-mb2+mb3)/(len(x)**2)
            return np.sum(val)
        corr = np.zeros(shape=(len(impt),len(impt)))            
        for i in range(len(impt)):
            for j in range(len(impt)):
                corr[i,j]=dist_covar(test[impt[i]].values,test[impt[j]].values)
            

    dataA = []
    all_pred = []
    for seed in range(1,3):
        if seed ==1 :
            a,b = 0,1
        else:
            a,b = 1,0
        replace = False
        test1=test.sort_values(['weight','10']).iloc[a::2].reset_index(drop=True)
        #test1 = test1.sample(len(test)//2,replace=replace, random_state=seed).reset_index(drop=True)
        test2=test.sort_values(['weight','10']).iloc[b::2].reset_index(drop=True)
        #features = [x[0][:-10] for x in results]
        X,Y1,Y2 = [],[],[]
        from sklearn import linear_model,preprocessing ,decomposition
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Lasso
        test = test.iloc[::-1]
        for alpha in np.linspace(-5,-2,5)[-2:-1]:# [0.03,0.1,0.3,1.0,3.0]:
            alpha =10**alpha
            from scipy.optimize import minimize
            temptrain  = StandardScaler().fit(test1[features]).transform(test1[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            B = np.zeros((X.shape[1],))
            W= test1[['weight',]].values
            y = test1[['20',]].values
            def weighted_lasso(B,X=X,y=y,W=W,alpha=alpha):
                B = np.reshape(B,(B.shape[0],1))
                score = W*(y-np.matmul(X,B))**2
                score = np.sum(score) /np.sum(W)
                score = score + np.sum(np.abs(B*alpha))
                return score
            B = minimize(weighted_lasso,B)['x']
            temptrain  = StandardScaler().fit(test1[features]).transform(test1[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            preds = np.matmul(X,np.reshape(B,(43,1)))[:,0]
            temptrain  = StandardScaler().fit(test1[features]).transform(test2[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            preds2 = np.matmul(X,np.reshape(B,(43,1)))[:,0]
            a,b=np.mean((preds -np.log10(test1['10']))**2)**.5,np.mean((preds2 -np.log10(test2['10']))**2)**.5
    ##        temptrain  = StandardScaler().fit(test[features]).transform(test[features])
    ##        preds3 = clf.fit(StandardScaler().fit(test[features]).transform(test[features]),
    ##                         np.log10(test['10'])).predict(temptrain)
    ##        test['pred_linear'] = preds3
            print (alpha,a,b)
            X += [alpha,]
            Y1 += [a,]
            Y2 += [b,]    
            pd.DataFrame(features)['a']=clf.coef_
            D = pd.DataFrame(features);D['a']=clf.coef_
            print (D[D['a'] != 0][0].values)
            all_pred += [D[D['a'] != 0][0].values,]
            r1 = 1-np.mean((preds -test1['20'])**2)/np.mean((np.mean(test1['20']) -test1['20'])**2)
            r2 = str(np.corrcoef(preds2 ,np.log10(test2['10']))[0,1])[:4]#str(np.mean((preds2 -np.log10(test2['10']))**2)**.5)[:4]#str(1-np.mean((10**preds2 -test2['10'])**2)/np.mean((np.mean(test2['10']) -test2['10'])**2))[:4]
            dataA += [r2,]
            print (r2)
            print (alpha,a,b,r1,r2)
            if True:
                plt.plot([2.4,3.8],[2.4,3.8],'g',label='y = x');
                #plt.errorbar(preds,np.log10(test1['10']),yerr=test1['err'],ecolor='r',fmt='o')
                plt.plot(preds,np.log10(test1['10']),'ro',label='train predictions');
                plt.ylim([2.4,3.8]);plt.xlim([2.4,3.8]);
                plt.title('Stapled Peptide model predict stapled peptide\ncorr coef = %s'%r2);
                plt.xlabel('predictions of log flourescence');plt.ylabel('ground truth of log flourescence');
                #plt.errorbar(preds2,np.log10(test2['10']),yerr=test2['err'],ecolor='b',fmt='o')
                plt.plot(preds2,np.log10(test2['10']),'bo',label='test predictions');#plt.legend()
                plt.savefig('StapledPeptide_model_predict_StapledPeptide'+str(cutoff)+'.png');
                plt.show()
    collections.Counter(np.concatenate(all_pred))
    if True:
        cluster = fcluster(Y, .45, criterion='distance')
        test = test.set_value(test.index,'weight2',1*(cluster==11))
    if True:
        from scipy.optimize import minimize
        temptrain  = StandardScaler().fit(test[features]).transform(test[features])
        X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
        B = np.zeros((X.shape[1],))
        W= test[['weight',]].values
        y = test[['20',]].values
        def weighted_lasso(B,X=X,y=y,W=W,alpha=alpha):
            B = np.reshape(B,(B.shape[0],1))
            score = W*(y-np.matmul(X,B))**2
            score = np.sum(score) /np.sum(W)
            score = score + np.sum(np.abs(B*alpha))
            return score
        minimize(weighted_lasso,B)
    for seed in range(1,3): #weighted alsso clusters 
        if seed ==1 :
            a,b = 0,1
        else:
            a,b = 1,0
        replace = False
        test1=test[test['weight2']==a].reset_index(drop=True)
        #test1 = test1.sample(len(test)//2,replace=replace, random_state=seed).reset_index(drop=True)
        test2=test[test['weight2']==b].reset_index(drop=True)
        #features = [x[0][:-10] for x in results]
        X,Y1,Y2 = [],[],[]
        from sklearn import linear_model,preprocessing ,decomposition
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Lasso
        test = test.iloc[::-1]
        for alpha in np.linspace(-5,-2,5)[-2:-1]:# [0.03,0.1,0.3,1.0,3.0]:
            alpha =10**alpha
            from scipy.optimize import minimize
            temptrain  = StandardScaler().fit(test1[features]).transform(test1[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            B = np.zeros((X.shape[1],))
            W= test1[['weight',]].values
            y = test1[['20',]].values
            def weighted_lasso(B,X=X,y=y,W=W,alpha=alpha):
                B = np.reshape(B,(B.shape[0],1))
                score = W*(y-np.matmul(X,B))**2
                score = np.sum(score) /np.sum(W)
                score = score + np.sum(np.abs(B*alpha))
                return score
            B = minimize(weighted_lasso,B)['x']
            temptrain  = StandardScaler().fit(test1[features]).transform(test1[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            preds = np.matmul(X,np.reshape(B,(len(features),1)))[:,0]
            temptrain  = StandardScaler().fit(test1[features]).transform(test2[features])
            X = np.concatenate([temptrain[:,0:1]*0+1,temptrain],1)
            preds2 = np.matmul(X,np.reshape(B,(len(features),1)))[:,0]
            a,b=np.mean((preds -np.log10(test1['10']))**2)**.5,np.mean((preds2 -np.log10(test2['10']))**2)**.5
    ##        temptrain  = StandardScaler().fit(test[features]).transform(test[features])
    ##        preds3 = clf.fit(StandardScaler().fit(test[features]).transform(test[features]),
    ##                         np.log10(test['10'])).predict(temptrain)
    ##        test['pred_linear'] = preds3
            print (alpha,a,b)
            X += [alpha,]
            Y1 += [a,]
            Y2 += [b,]    
            pd.DataFrame(features)['a']=B[1:]
            D = pd.DataFrame(features);D['a']=B[1:]
            print (D[D['a'] != 0][0].values)
            all_pred += [D[D['a'] != 0][0].values,]
            r1 = 1-np.mean((preds -test1['20'])**2)/np.mean((np.mean(test1['20']) -test1['20'])**2)
            r2 = str(np.corrcoef(preds2 ,np.log10(test2['10']))[0,1])[:4]#str(np.mean((preds2 -np.log10(test2['10']))**2)**.5)[:4]#str(1-np.mean((10**preds2 -test2['10'])**2)/np.mean((np.mean(test2['10']) -test2['10'])**2))[:4]
            dataA += [r2,]
            print (r2)
            print (alpha,a,b,r1,r2)
            if True:
                plt.plot([2.4,3.8],[2.4,3.8],'g',label='y = x');
                #plt.errorbar(preds,np.log10(test1['10']),yerr=test1['err'],ecolor='r',fmt='o')
                plt.plot(preds,np.log10(test1['10']),'ro',label='train predictions');
                plt.ylim([2.4,3.8]);plt.xlim([2.4,3.8]);
                plt.title('Stapled Peptide model predict stapled peptide\ncorr coef = %s'%r2);
                plt.xlabel('predictions of log flourescence');plt.ylabel('ground truth of log flourescence');
                #plt.errorbar(preds2,np.log10(test2['10']),yerr=test2['err'],ecolor='b',fmt='o')
                plt.plot(preds2,np.log10(test2['10']),'bo',label='test predictions');#plt.legend()
                plt.savefig('WORSTCASE'+str(cutoff)+'.png');
                plt.show()

'''
L_num         -0.0411      0.014     -2.913      0.004      -0.069      -0.013
NL_num        -0.0857      0.052     -1.642      0.103      -0.189       0.017
K_num          0.9926      0.012     80.829      0.000       0.968       1.017
F_num         -0.0081      0.020     -0.401      0.689      -0.048       0.032
T_num         -0.1181      0.023     -5.160      0.000      -0.163      -0.073
B8_num         0.3055      0.105      2.908      0.004       0.098       0.513
C_num          0.1361      0.062      2.203      0.029       0.014       0.258
G_num          0.0805      0.024      3.423      0.001       0.034       0.127
M_num         -0.0106      0.048     -0.222      0.825      -0.105       0.084
R5_num        -0.5665      0.050    -11.300      0.000      -0.665      -0.467
N_num          0.0799      0.021      3.843      0.000       0.039       0.121
V_num          0.0889      0.021      4.331      0.000       0.048       0.129
S8_num        -0.6650      0.056    -11.834      0.000      -0.776      -0.554
I_num         -0.0245      0.018     -1.366      0.174      -0.060       0.011
R8_num        -0.4440      0.041    -10.799      0.000      -0.525      -0.363
Q_num         -0.0046      0.013     -0.366      0.715      -0.029       0.020
pff_num       -0.2225      0.054     -4.087      0.000      -0.330      -0.115
R_num          1.0097      0.007    135.127      0.000       0.995       1.024
D_num         -0.9477      0.021    -44.593      0.000      -0.990      -0.906
H_num          0.1875      0.029      6.564      0.000       0.131       0.244
E_num         -0.9900      0.017    -57.135      0.000      -1.024      -0.956
B5_num         0.0227      0.046      0.495      0.621      -0.068       0.113
S5_num        -0.6078      0.030    -20.451      0.000      -0.666      -0.549
Y_num          0.0104      0.025      0.410      0.682      -0.040       0.060
P_num         -0.1294      0.031     -4.227      0.000      -0.190      -0.069
A_num         -0.0415      0.016     -2.670      0.008      -0.072      -0.011
W_num          0.0789      0.039      2.044      0.043       0.003       0.155
S_num         -0.0114      0.020     -0.572      0.568      -0.051       0.028
'''
