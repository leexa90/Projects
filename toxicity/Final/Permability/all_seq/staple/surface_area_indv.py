import pandas as pd
pd.options.display.max_rows = 20
import numpy as np
import sklearn
import xgboost
import scipy
from scipy import histogram, digitize, stats, mean, std
all  = []
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import collections
for line in open('../bond_lengths.csv','r'):
    all += [line.split(),]
bond_L= pd.DataFrame(all[1:],columns=all[0])
all = []
for line in open('../VDW.csv','r'):
    all += [line.split(),]
VDW= pd.DataFrame(all[1:],columns=all[0])

test = pd.read_csv('../stapled_peptide_permability_features.csv')
for feats in ['SMR_VSA','PEOE_VSA','SlogP_VSA']:
    feat_all = [ x for x in test.keys() if feats in x]
    for feat in feat_all:
        test[feat+'_norm']=test[feat]/np.sum(test[feat_all],1)
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
    test['res_list_QN']=test['res_list'].apply(lambda x : collections.Counter(x)['Q']+collections.Counter(x)['N'])
    test['res_list']=test['res_list'].apply(len)
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
    charge_csv =pd.read_csv('../stapled_peptide_geisteiger_charge.csv')
    for atom in [35, 71, 15, 11, 66, 39, 23, 34, 64, 16, 21, 63,  2, 29, 41, 57, 32,
            6, 56, 36,  7, 10,  3, 28, 37,  1,  5]:
        bins = histogram(charge_csv[charge_csv['atoms']==atom]['charge'].values,4)[1]
        bins[-1] = bins[-1]+0.001
        for i in range(len(bins)-1):
            def func(x,i=i,atom=atom):
                sum_ =  np.sum(x[1][(x[0]  >= bins[i] )&(x[0]  <= bins[i+1] ) &(np.array(x[2])==atom)])
                if sum_ >= 0:
                    return sum_
                else :
                    return 0.0
            def func_norm(x,i=i,atom=atom):
                sum_ =  np.sum(x[1][(x[0]  >= bins[i] )&(x[0]  <= bins[i+1] ) &(np.array(x[2])==atom)])
                if sum_ >= 0:
                    return sum_/np.sum(x[1])
                else :
                    return 0.0
            test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])] = test['charge_surf'].apply(func)
            test['charge_norm_%s_%s_%s'%(atom,bins[i],bins[i+1])] = test['charge_surf'].apply(func_norm)
            if np.mean(test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])])==0:
                del test['charge_%s_%s_%s'%(atom,bins[i],bins[i+1])]
                print ('charge_%s_%s_%s'%(atom,bins[i],bins[i+1]))

# 28,[10],[3, 28]
if True:
    test['id'] = test.index
    temp = charge_csv[(charge_csv['atoms']==atom) & (charge_csv['bond2'] == '[1, 3]') & (charge_csv['bond1'] == '[10]')].groupby(['origin'])['charge'].apply(len).reset_index()
    temp.columns = ['id','num']
    test = pd.merge(test,temp,how='left',on='id')
    test['num'] = test['num'].fillna(0)
    plt.plot(test['num'],np.log10(test['permability']),'ro');plt.show()
    del test['num']

charge_csv[(charge_csv['atoms'] == 28) & (charge_csv['bond1'] == '[1, 7, 10]') & (charge_csv['bond2'] == '[1, 1, 5, 10, 28] ')].groupby(['bond1','bond2'])['charge'].apply(
    lambda x : list(map(lambda x : round(x,4),(len(x),min(x),max(x),mean(x),std(x))))).reset_index()

#print reuslts
if True:
    results = []
    from sklearn.metrics import r2_score
    for i in test.keys()[10:]:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
        try:
            X,Y1,Y2 = [],[],[]
            from sklearn import linear_model,preprocessing ,decomposition
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Lasso
            alpha =0.00#alpha#10**alpha
            clf = linear_model.LinearRegression()
            temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
            preds = clf.fit(temptrain,np.log10(test['10'])).predict(temptrain)
            se = np.sum((preds-np.log10(test['10']))**2)/np.sum((temptrain-np.mean(temptrain))**2)
            se = 2*(se/215)**.5
            if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
                results += [(i,r2_score(np.log10(test['10']),preds),
                             clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se]),]
        except : print (i)
    print ([x[:2] for x in sorted(results,key =  lambda x : x[1])[-20:-2]])
#test[''] = 
#plt.plot(test['charge_norm_7_-0.27398123963107096_-0.27251541490535375'],np.log10(test['permability']),'ro');plt.show()
'''

for i in sorted(results,key =  lambda x : x[1])[-17:-2]:
	print (i[0:1]+tuple(map(lambda x:np.round(x,3),i[1:4])))


# RELATING TO SIZE BEING SMALLER molecules (total surface area)
('charge_norm_1_0.10431742603390584_0.11737113959967935', 0.374, -0.093, 0.032)
('charge_41_0.06415049310865131_0.06666999388438306', 0.375, 0.093, 0.032)
('PEOE_VSA12_norm', 0.384, -0.096, 0.031)
('PEOE_VSA1_norm', 0.385, -0.096, 0.031)
('charge_norm_3_0.2306185636697955_0.24619849559912854', 0.386, -0.096, 0.031)
('SMR_VSA10_norm', 0.388, -0.097, 0.031)
('charge_norm_41_0.041474986127065495_0.04399448690279725', 0.396, -0.099, 0.031)
('charge_norm_32_-0.5501702476405683_-0.549912333580245', 0.396, -0.099, 0.031)
('charge_norm_32_-0.5501702476405683_-0.5500412906104066', 0.396, -0.099, 0.031)
('charge_norm_41_0.06415049310865131_0.06666999388438306', 0.398, 0.099, 0.031)
('charge_norm_3_0.21503863174046248_0.24619849559912854', 0.408, -0.102, 0.031)
('charge_norm_28_0.16300020569822835_0.1639752966596217', 0.445, -0.111, 0.03)
('charge_norm_7_-0.27398123963107096_-0.27251541490535375', 0.45, -0.112, 0.03)
('charge_norm_28_0.16348775117892503_0.1639752966596217', 0.45, -0.112, 0.03)
('charge_norm_7_-0.27544706435678823_-0.27251541490535375', 0.453, -0.113, 0.03)
'''
MMFF_dictt = pd.read_csv('res_charge.csv')
import collections
for ii in sorted(results,key =  lambda x : x[1])[-17:-2]:
    def func3(ii):
        if 'charge_norm_41' in ii[0]:
            print (ii),
            i = list(map(float,ii[0].split('_')[-3:]))
            atom=int(i[0])
            print (i)
            temp = charge_csv[(charge_csv['atoms'] == atom)].groupby(['bond1','bond2'])['charge'].apply(
            lambda x : list(map(lambda x : round(x,6),(len(x),min(x),max(x),mean(x),std(x))))).reset_index()
            def func2(x,i=i[1:]): # function to get records within max and min of the feature range
                x =x[1:]
                if (x[1] >= i[0] and x[1] <= i[1]) or (x[0] >= i[0] and x[0] <= i[1]) or\
                   (i[1] >= x[0] and i[1] <= x[1]) or (i[0] >= x[0] and i[0] <= x[1]):
                    return 1
                else: return 0
            temp =  temp[temp['charge'].apply(func2)==1]
            #print (pd.merge(temp,MMFF_dictt,'left',on=['bond1','bond2']).sort_values('origin'))
            print (collections.Counter(
                pd.merge(temp,MMFF_dictt,'left',on=['bond1','bond2']).sort_values('origin')['origin'].values)
                   ,'\n')
            return pd.merge(temp,MMFF_dictt,'left',on=['bond1','bond2']).sort_values('origin')
    func3(ii=ii)
    
    
MMFF_dictt = pd.read_csv('res_charge.csv')
dictt = {'AC':'CC(=O)',
         'A':'N[C@H](C)C(=O)',
         'C':'N[C@H](CS)C(=O)',
         'D':'N[C@H](CC(=O)[O-])C(=O)',
         'E':'N[C@H](CCC(=O)[O-])C(=O)',
         'F':'N[C@H](Cc1ccccc1)C(=O)',
         'G':'NCC(=O)',
         'H':'N[C@H](Cc1c[nH]cn1)C(=O)',
         'I':'N[C@H]([C@@H](C)CC)C(=O)',#NC([C@@H](CC)C)C(=O)
         'K':'N[C@H](CCCC[NH3+])C(=O)',
         'L':'N[C@H](CC(C)C)C(=O)',
         'M':'N[C@H](CCSC)C(=O)',
         'N':'N[C@H](CC(N)=O)C(=O)',
         'NL':'N[C@@H](CCCC)C(=O)',
         'P':'N1CCCC1C(=O)',
         'Q':'N[C@H](CCC(N)=O)C(=O)',
         'R':'N[C@H](CCCNC(=[NH2+])N)C(=O)',
         'S':'N[C@H](CO)C(=O)',
         'T':'N[C@H]([C@H](O)C)C(=O)', #NC([C@H](O)C)C(=O)
         'V':'N[C@H](C(C))C(=O)',
         'W':'N[C@H](Cc1c[nH]c2ccccc12)C(=O)',
         'Y':'N[C@H](Cc1ccc(O)cc1)C(=O)',
         'S5':'N[C@](CCCC=3)(C)C(=O)',
         'S8':'N[C@](CCCCCCC=3)(C)C(=O)',
         'R5':'N[C@@](CCCC=3)(C)C(=O)',
         'R8':'N[C@@](CCCCCCC=3)(C)C(=O)',
         'S53':'N[C@](CCCC=4)(C)C(=O)',
         'S83':'N[C@](CCCCCCC=4)(C)C(=O)',
         'R53':'N[C@@](CCCC=4)(C)C(=O)',
         'R83':'N[C@@](CCCCCCC=4)(C)C(=O)',
         'B8':'NC(CCCC=3)(CCCC=4)C(=O)', #???
         'B5':'NC(CCCC=3)(CCCC=4)C(=O)',#'S8':'N[C@](CCCCCCC=3)(C)C(=O)' for S8
         'BAla':'NCCC(=O)',
         'PEG1':'NCCOCCOCC(=O)',
         'PEG':'NCCOCCOCC(=O)',
         'PEG2':'NCCOCOCCOCCC(=O)',
         'PEG5':'NCCOCCOCCOCCOCCOCCOCCC(=O)',
         'EEA':'NCCOCCOCCNC(=O)CCC(=O)',
         'pff':'N[C@@H](Cc1c(F)c(F)c(F)c(F)c1(F))C(=O)',
         'FITC':'C(NC1=CC=C2C(=C1)C4(OC2=O)C3=C(C=C(C=C3)O)OC5=C4C=CC(=C5)O)(=S)'}
         
data_dictt=pd.DataFrame(None)
data_dictt['res']=dictt.key()
data_dictt['SMILES']=dictt.values()
