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
def process_string(str):
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
            if i in ['Q','N']:
                result += [i,]
    return len(result)
test['len']=test['1'].apply(process_string)
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
        bins = histogram(charge_csv[charge_csv['atoms']==atom]['charge'].values,10)[1]
        for i in range(len(bins)-1):
            def func(x,atom=atom):
                sum_ =  np.sum(x[1][(x[0]  >= bins[i] )&(x[0]  <= bins[i+1] ) &(np.array(x[2])==atom)])
                if sum_ >= 0:
                    return sum_
                else :
                    return 0.0
            def func_norm(x,atom=atom):
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
charge_csv[(charge_csv['atoms'] == 28)].groupby(['bond1','bond2'])['charge'].apply(
    lambda x : list(map(lambda x : round(x,4),(len(x),min(x),max(x),mean(x),std(x))))).reset_index()

#print reuslts
if True:
    results = []
    for i in test.keys()[10:]:#(0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3):
        try:
            X,Y1,Y2 = [],[],[]
            from sklearn import linear_model,preprocessing ,decomposition
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Lasso
            alpha =0.00#alpha#10**alpha
            clf = linear_model.Lasso(alpha=alpha)
            temptrain  = StandardScaler().fit(test[[i,]]).transform(test[[i,]])
            preds = clf.fit(temptrain,np.log10(test['10'])).predict(temptrain)
            se = np.sum((preds-np.log10(test['10']))**2)/np.sum((temptrain-np.mean(temptrain))**2)
            se = 2*(se/215)**.5
            if  clf.coef_[0]-se >0 or clf.coef_[0]+se <0 :
                results += [(i,np.corrcoef(preds,np.log10(test['10']))[0,1],
                             clf.coef_[0],se,[clf.coef_[0]-se,clf.coef_[0]+se]),]
        except : print (i)
    print (sorted(results,key =  lambda x : x[1]))
#test[''] = 
plt.plot(test['charge_norm_7_-0.27398123963107096_-0.27251541490535375'],np.log10(test['permability']),'ro');plt.show()
'''
[('charge_norm_32_-0.5501702476405683_-0.549912333580245', 0.3963576040491148, -0.09874573632951715),

# RELATING TO SIZE BEING SMALLER molecules (total surface area)
('charge_norm_41_0.06415049310865131_0.06666999388438306', 0.3979022597040567, 0.09913056093856876),
('charge_norm_3_0.21503863174046248_0.24619849559912854', 0.40792828210874027, -0.10162837340562553),
('charge_norm_28_0.16300020569822835_0.1639752966596217', 0.44530923007222073, -0.11094119897940165),
('charge_norm_7_-0.27398123963107096_-0.27251541490535375', 0.44950482365016764, -0.11198645955460311),
('charge_norm_28_0.16348775117892503_0.1639752966596217', 0.4495856145482003, -0.11200658722878785),
('charge_norm_7_-0.27544706435678823_-0.27251541490535375', 0.45333406533731424, -0.11294044980511977)]
'''

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
         
