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
test = pd.read_csv('../peptide_permability_features.csv')
temp=test.iloc[x]
test['1'] = test['ID']
#test = test[test['7'] == 1].reset_index(drop=True)
#test = test[test['1'] != '-AC-AHL-R8-LCLEKL-S5-GLV-(K-PEG1--FITC-)'].reset_index(drop=True)
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
    test['TPSA/LabuteASA'] = test['TPSA']/test['LabuteASA'] #ratio of polar SA
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
##for i in range(len(test)):
##    for j in range(i,len(test)):
##        alignments = pairwise2.align.globalms(test.iloc[i]['string'],
##                                             test.iloc[j]['string'],
##                                              2, -1, -0.5, -0.1)
##        score = max([0.5*a[-3]/a[-1] for a in alignments])
##        corr_mat[i,j] = score
##        corr_mat[j,i] = score
if False:
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
#test['charge_surf'] = test['SMILES'].apply( get_surface_area)
#test['charge_num'] = test['charge_surf'].apply(lambda x : np.sum(x[0]))
#test['charge_mean'] = test['charge_surf'].apply(lambda x : np.mean(x[0]))
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

#weight matrix
if True:
    #cluster = fcluster(Y, .66, criterion='distance')
    #test = test.set_value(test.index,'weight0',cluster)
    test['weight'] = 1#.0/test['weight0'].map(collections.Counter(fcluster(Y, .66, criterion='distance')))
# get which cluster is it in as a intercept #
for i in [3,]:#[1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]: 
    test['cluster_%s'%i] = test.set_value(test.index,'cluster_%s'%i,1)#fcluster(Y,i, criterion='distance'))['cluster_%s'%i]
    #print (i,max(fcluster(Y,i, criterion='distance')))
    #class the small clusters into -1 
    #temp = test.groupby('cluster_%s'%i).apply(len).reset_index()
    #id_minus1 = test[test['cluster_%s'%i].isin(temp[temp[0] <5]['cluster_%s'%i].values)].index
    #test = test.set_value(id_minus1,'cluster_%s'%i,-1)
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
dictt_vol = {'FITC': 163.32732360530238, 'PEG2': 75.972, 'pff': 84.691, 'Y': 68.658, 'I': 47.901,
             'E': 50.492, 'A': 28.806, 'P': 40.74, 'L': 47.901, 'AC': 16.891, 'B8': 71.665, 'B88': 146.863,
             'S5': 53.418, 'BAla': 28.806, 'PEG': 58.128, 'S': 33.6, 'D': 44.127, 'EEA': 93.299, 'F': 63.863,
             'R8': 72.513, 'T': 39.965, 'R5': 50.236, 'NL': 47.901, 'N': 44.673, 'Q': 51.038, 'R88': 115.722,
             'S8': 72.513, 'C': 39.961, 'B5': 71.665, 'G': 22.441, 'K': 53.241, 'R': 63.476, 'PEG1': 58.128,
             'W': 79.754, 'M': 51.659, 'V': 35.171, 'PEG5': 135.867, 'H': 56.292}
for i in [('W','F','Y','pff'),('N','Q'),('L','I','V','NL'),('D','E'),
          ('S8','R8','B8'),('R5','S5'),('PEG2','PEG5','PEG1'),
          ('H','R','K'),('S','T'),('S8','R8'),('S8','B8'),('R8','B8'),
          ('W','F','pff'),('F','pff'),('F','Y')]:
    dictt[i] = 999
          
#single res
for res in dictt.keys():
    if dictt[res] >= 20:
        test['%s_num' %str(res)] = test['list'].apply(lambda x : sum(np.in1d(x,np.array(res))))
        test['%s_norm' %str(res)] = 1.0*test['%s_num'%str(res)]/test['res_list']
        #test['%s_normSA' %str(res)] = test['list'].apply(lambda y: np.array(list(map(lambda x : dictt_vol[x],y)))).apply(np.sum) #total area
        #test['%s_normSA' %str(res)] = test['list'].apply(lambda y : sum(np.array(list(map(lambda x : dictt_vol[x],y)))[np.in1d(y,np.array(res))]))\
        #                              /test['list'].apply(lambda y: np.array(list(map(lambda x : dictt_vol[x],y)))).apply(np.sum)
        impt += ['%s_norm' %str(res),'%s_num' %str(res),]#'%s_normSA' %str(res)]
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
    test['10'] = (test['source']==2)*9 +1
    clusters = [x for x in test.keys() if 'cluster_' in x]
    results = []
    from sklearn.metrics import r2_score
    def log_likelihood(X,B,y): #loglihood of logistic regresion parameters
        temp1 = np.sum(y * np.matmul(X,B))
        temp2 = np.sum(np.log(1+np.exp(np.matul(X,B))))
        return temp1-temp2
    
    for i in list(test.keys())[15:]:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
        for cluster in clusters:
            try:
                X,Y1,Y2 = [],[],[]
                from sklearn import linear_model,preprocessing ,decomposition
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import Lasso
                alpha =0.00#alpha#10**alpha
                clf = linear_model.LogisticRegression()
                temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
                ids_non_zero = test[test[i]!=0].index
                if 'num' in i or 'res_list' in i or i in ['atom_1', 'atom_6', 'atom_7', 'atom_8', 'atom_9', 'atom_16', 'atom_len', 'atom_size']: #do not scale count features. 
                    temptrain = test[[i,]].values
                    print  (i)
                    i='num_' + i
                temptrain2 = np.concatenate([temptrain*0+1,temptrain],1)
                #dummie = pd.get_dummies(test[cluster].values,drop_first= True).values
                #temptrain = np.concatenate([dummie,temptrain],1)
                preds = clf.fit(temptrain,np.log10(test['10']),sample_weight=test['weight']).predict(temptrain)
                # equailvalent ~ inv(X.T*W*X)*X.T*W*Y
                W=np.diag(test['weight'])
                X=temptrain2
                y=np.log10(test['10'])
                B=np.concatenate([[clf.intercept_],clf.coef_])
                preds = np.matmul(X,B)
                # sm.WLS(y, X, weights=test['weight']).fit().summary()
                se = np.sum(
                    (test['weight']*(preds-np.log10(test['10']))**2)
                    )*np.linalg.inv(
                         np.matmul(np.matmul(temptrain2.T,np.diag(test['weight'])),temptrain2)
                         )[-1,-1]
                se = 1.67*(se/(sum(test['weight'])-2))**.5 #effective sample size
    
                if  (clf.coef_[0]-2*se/2 >0 or clf.coef_[0]+2*se/2  <0) and r2_score(np.log10(test['10']),preds,sample_weight=test['weight']) < 0.7:
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
if True:
    plt.close()
    counter = 1
    plots = []
    for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
            if i[2] < -0.01 and 'num' in i[0] and '.(' not in i[0]:
                    plots += [i,]
                    plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='r')
                    plt.plot([i[-4],],[counter,]*1,'ro')
                    plt.text(i[-4],counter+.25,i[0][:-10],horizontalalignment='center',fontsize=7)
                    counter += 1
    try:
        i = [x for x in sorted(results,key =  lambda x : x[-4]) if 'res_list_cluster_3' in x[0]][0]
        plots += [i,]
        plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='g')
        plt.plot([i[-4],],[counter,]*1,'go')
        plt.text(i[-4],counter+.25,'Number of residues',horizontalalignment='center',fontsize=7)
        counter += 1
    except : None
    for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
            if i[2] > 0.01 and 'num' in i[0] and '.(' not in i[0]:
                    plots += [i,]
                    plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='b')
                    plt.plot([i[-4],],[counter,]*1,'bo')
                    plt.text(i[-4],counter+.25,i[0][:-10],horizontalalignment='center',fontsize=7)
                    counter += 1
    plt.title('Number Features')
    plt.xlabel('Coefficient and 95% CI of feature')
    plt.ylabel('Features')
    plt.yticks([],[])
    plt.ylim([0,counter])
    plt.savefig('B_coef_linear.png',dpi=300),plt.show()
    plt.close()
    counter = 1
    plots = []
    for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
            if i[2] < -0.01 and 'num' not in i[0] and 'list' not in i[0] and 'normSA' not in i[0] and 'VSA' not in i[0]:
                    plots += [i,]
                    plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='r')
                    plt.plot([i[-4],],[counter,]*1,'ro')
                    plt.text(i[-4],counter+.25,i[0][:-10],horizontalalignment='center',fontsize=7)
                    counter += 1
    for i in [x for x in sorted(results,key =  lambda x : x[-4])[-250:]]:
            if i[2] > 0.01 and 'num' not in i[0] and 'list' not in i[0] and 'normSA' not in i[0] and 'VSA' not in i[0]:
                    plots += [i,]
                    plt.errorbar([i[-4],],[counter,]*1,xerr=i[-3],fmt='b')
                    plt.plot([i[-4],],[counter,]*1,'bo')
                    plt.text(i[-4],counter+.25,i[0][:-10],horizontalalignment='center',fontsize=7)
                    counter += 1
    plt.ylim([0,counter])
    plt.title('Percentage Features')
    plt.xlabel('Coefficient and 95% CI of feature')
    plt.ylabel('Percentage Features')
    plt.yticks([],[])
    plt.savefig('B_norm_linear.png',dpi=300),plt.show()    


                    
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
if True:
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
    temp = temp[temp['num'] > 5].reset_index(drop=True)
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
    plt.savefig('length_linear.png',dpi=300,bbox_inches='tight')
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
        
        
